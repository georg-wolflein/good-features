from pathlib import Path
from typing import Sequence, Tuple, Optional, Mapping
import sys
import pandas as pd
import pytorch_lightning as pl
import torch
from torch import nn
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, Callback, TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.loggers.wandb import WandbLogger
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf
import hydra
import os
from textwrap import indent
from loguru import logger
import numpy as np
import functools

os.environ["HYDRA_FULL_ERROR"] = "1"

from .utils import (
    make_dataset_df,
    pathlist,
    DummyBiggestBatchFirstCallback,
    flatten_batched_dicts,
    make_preds_df,
)
from ..data import FeatureDataset
from .targets import TargetEncoder
from .metrics import create_metrics_for_target
from .utils import summarize_dataset
import histaug

# from hydra.core.global_hydra import GlobalHydra
# GlobalHydra().clear()
# hydra.initialize(config_path="conf")
# cfg = hydra.compose(
#     "config.yaml"#, overrides=["+experiment=mnist_collage", "+model=gnn_gat"]
# )


class LitMilTransformer(pl.LightningModule):
    def __init__(
        self,
        cfg: DictConfig,
    ) -> None:
        super().__init__()

        # This actually overwrites wandb config, but we don't want that
        # self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))

        self.model: nn.Module = hydra.utils.instantiate(cfg.model)
        self.targets = cfg.dataset.targets
        self.cfg = cfg

        metric_goals = dict()
        for step_name in ["train", "val", "test"]:
            metrics_and_goals = {target.column: create_metrics_for_target(target) for target in self.targets}
            metric_goals.update(
                {
                    f"{step_name}/{column}/{name}": goal
                    for column, (metrics, goals) in metrics_and_goals.items()
                    for (name, goal) in goals.items()
                }
            )
            metrics = nn.ModuleDict({c: metrics for c, (metrics, goals) in metrics_and_goals.items()})
            setattr(self, f"{step_name}_target_metrics", metrics)
        self.metric_goals = metric_goals

        self.losses = nn.ModuleDict(
            {
                target.column: nn.CrossEntropyLoss(
                    weight=(
                        w := torch.tensor(target.weights, device=self.device, dtype=torch.float)
                        if target.weights
                        else torch.ones(len(target.classes), device=self.device, dtype=torch.float)
                    )
                    / w.sum(),
                    reduction="sum",
                )
                if target.type == "categorical"
                else nn.MSELoss()
                for target in self.targets
            }
        )
        self.loss_weights = {target.column: target.get("weight", 1.0) for target in self.targets}

    def step(self, batch, step_name=None):
        feats, coords, mask, targets, *_ = batch
        logits = self(feats, coords, mask)

        # Calculate the CE or MSE loss for each target, then sum them
        losses = {
            column: loss(logits[column], targets[column]) * self.loss_weights[column]
            for column, loss in self.losses.items()
        }
        loss = sum(losses.values())

        if step_name:
            self.log(
                f"{step_name}/loss",
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )
            self.log_dict(
                {f"{step_name}/loss/{column}": l for column, l in losses.items()},
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

            # Update target-wise metrics
            for target in self.targets:
                target_metrics = getattr(self, f"{step_name}_target_metrics")[target.column]
                y_true = targets[target.column]
                target_metrics.update(logits[target.column], y_true)
                self.log_dict(
                    {f"{step_name}/{target.column}/{name}": metric for name, metric in target_metrics.items()},
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )

        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, step_name="train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, step_name="val")

    def test_step(self, batch, batch_idx):
        return self.step(batch, step_name="test")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        feats, coords, mask, *_ = batch
        logits = self(feats, coords, mask)

        softmaxed = {
            t.column: (torch.softmax(logits[t.column], -1) if t.type == "categorical" else logits[t.column])
            for t in self.targets
        }
        return softmaxed

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(self.cfg.optimizer, params=self.parameters())
        scheduler = (
            hydra.utils.instantiate(self.cfg.scheduler, optimizer=optimizer)
            if self.cfg.get("scheduler", None)
            else None
        )
        return optimizer if scheduler is None else ([optimizer], [scheduler])

    @property
    def learning_rate(self):
        return self.optimizers().param_groups[0]["lr"]

    @learning_rate.setter
    def learning_rate(self, value):
        for param_group in self.optimizers().param_groups:
            param_group["lr"] = value

    def forward(self, *args):
        return self.model(*args)


class DefineWandbMetricsCallback(Callback):
    def __init__(self, model: LitMilTransformer, run) -> None:
        super().__init__()
        self.metric_goals = model.metric_goals
        self.run = run

    def on_train_start(self, *args, **kwargs) -> None:
        for name, goal in self.metric_goals.items():
            self.run.define_metric(name, goal=f"{goal}imize", step_metric="epoch", summary="best")


def make_trainer(
    cfg: DictConfig,
    dummy_batch: torch.Tensor,
    crossval_id: Optional[str] = None,
    crossval_fold: Optional[int] = None,
    augmentation_keys: Optional[Sequence[str]] = (),
    run_prefix: str = "",
    callbacks: Sequence[Callback] = (),
) -> Tuple[LitMilTransformer, pl.Trainer, Path, WandbLogger]:
    model = LitMilTransformer(cfg)

    name = cfg.name if crossval_fold is None else f"{cfg.name}-fold{crossval_fold}"
    name = name if run_prefix == "" else f"{run_prefix}-{name}"

    wandb_logger = WandbLogger(
        name,
        project=cfg.project,
        job_type="cv" if crossval_id else "train",
    )
    wandb_logger.experiment.config.update(OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))
    wandb_logger.experiment.config["overrides"] = " ".join(sys.argv[1:])
    wandb_logger.experiment.log_code(
        ".",
        include_fn=lambda path: Path(path).suffix in {".py", ".yaml", ".yml"} and "env" not in Path(path).parts,
    )
    if crossval_id is not None:
        if crossval_id == "":
            crossval_id = wandb_logger.experiment.id
        wandb_logger.experiment.config.update({"crossval_id": crossval_id, "crossval_fold": crossval_fold})
    wandb_logger.experiment.tags = (*wandb_logger.experiment.tags, *augmentation_keys)

    out_dir = Path(cfg.output_dir) / (wandb_logger.version or "")
    if crossval_id is not None:
        out_dir = Path(cfg.output_dir) / crossval_id / f"fold{crossval_fold}_{wandb_logger.version}"

    define_metrics_callback = DefineWandbMetricsCallback(model, wandb_logger.experiment)

    callbacks = [
        DummyBiggestBatchFirstCallback(dummy_batch),
        TQDMProgressBar(refresh_rate=10),
        define_metrics_callback,
        *callbacks,
    ]

    if cfg.early_stopping.enabled:
        callbacks.append(
            EarlyStopping(
                monitor=cfg.early_stopping.metric,
                mode=cfg.early_stopping.goal,
                patience=cfg.early_stopping.patience,
            )
        )

    trainer = pl.Trainer(
        # profiler="simple",
        default_root_dir=out_dir,
        callbacks=callbacks,
        max_epochs=cfg.max_epochs,
        accelerator="gpu",
        devices=cfg.device or 1,
        accumulate_grad_batches=cfg.accumulate_grad_samples // cfg.dataset.batch_size
        if cfg.accumulate_grad_samples
        else 1,
        gradient_clip_val=cfg.grad_clip,
        logger=[CSVLogger(save_dir=out_dir), wandb_logger],
    )

    return model, trainer, out_dir, wandb_logger


def load_dataset_df(dataset_cfg: DictConfig):
    dataset_df = make_dataset_df(
        clini_tables=pathlist(dataset_cfg.clini_tables),
        slide_tables=pathlist(dataset_cfg.slide_tables),
        feature_dirs=pathlist(dataset_cfg.feature_dirs),
        patient_col=dataset_cfg.patient_col,
        filename_col=dataset_cfg.filename_col,
        target_labels=[label.column for label in dataset_cfg.targets],
    )

    # Remove patients with no target labels
    to_delete = pd.Series(False, index=dataset_df.index)
    for target in dataset_cfg.targets:
        to_delete |= dataset_df[target.column].isna()
        if target.type == "categorical":
            to_delete |= ~dataset_df[target.column].isin(target.classes)
    if to_delete.any():
        print(
            f"Removing {to_delete.sum()} patients with missing target labels (or unsupported classes); {(~to_delete).sum()} remaining"
        )
    dataset_df = dataset_df[~to_delete]
    return dataset_df


def get_folds(cfg: DictConfig, dataset_df: pd.DataFrame) -> pd.Series:
    folds_file = Path(cfg.dataset.folds_table)
    if not folds_file.exists():
        n_folds = 5
        patients = dataset_df.index.unique()
        n_patients = len(patients)
        logger.info(f"Folds table {folds_file} missing, so creating {n_folds} folds for {n_patients} patients")
        folds = np.arange(n_patients) % n_folds
        np.random.shuffle(folds)
        folds_df = pd.DataFrame({cfg.dataset.patient_col: patients, "fold": folds}).set_index(cfg.dataset.patient_col)
        folds_df.to_csv(folds_file)

    folds_df = pd.read_csv(folds_file).set_index(cfg.dataset.patient_col)

    # Ensure folds_df contains only patients in dataset_df; remove extra patients
    folds_df = folds_df.loc[folds_df.index.isin(dataset_df.index.unique())]

    return folds_df.fold


def train_fold(
    cfg: DictConfig,
    dataset_df: pd.DataFrame,
    folds: pd.Series,
    crossval_id: Optional[str] = None,
    crossval_fold: Optional[int] = None,
    run_prefix: str = "",
):
    logger.info(
        f"Using fold {crossval_fold} for validation, contains {(folds == crossval_fold).mean()*100:.1f}% of patients"
    )

    valid_mask = folds == crossval_fold
    train_items, valid_items = folds.index[~valid_mask], folds.index[valid_mask]
    train_df, valid_df = dataset_df.loc[train_items], dataset_df.loc[valid_items]

    print("Train dataset:")
    print(indent(summarize_dataset(cfg.dataset.targets, train_df), "  "))

    print("Validation dataset:")
    print(indent(summarize_dataset(cfg.dataset.targets, valid_df), "  "))

    assert not (
        overlap := set(train_df.index) & set(valid_df.index)
    ), f"unexpected overlap between training and testing set: {overlap}"

    encoders = {target.column: TargetEncoder.for_target(target) for target in cfg.dataset.targets}
    train_targets = {t: encoder.fit(train_df) for t, encoder in encoders.items()}
    valid_targets = {t: encoder(valid_df) for t, encoder in encoders.items()}

    train_ds = FeatureDataset(
        patient_ids=train_df.index,
        bags=train_df.path.values,
        targets=train_targets,
        instances_per_bag=cfg.dataset.instances_per_bag,
        augmentations=cfg.dataset.augmentations.train,
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.dataset.batch_size,
        num_workers=cfg.dataset.num_workers,
        shuffle=True,
        pin_memory=True,
        collate_fn=train_ds.collate_fn,
    )

    valid_ds = FeatureDataset(
        patient_ids=valid_df.index,
        bags=valid_df.path.values,
        targets=valid_targets,
        instances_per_bag=cfg.dataset.instances_per_bag,
        augmentations=cfg.dataset.augmentations.val,
    )
    valid_dl = DataLoader(
        valid_ds,
        batch_size=cfg.dataset.batch_size,
        num_workers=cfg.dataset.num_workers,
        pin_memory=True,
        collate_fn=valid_ds.collate_fn,
    )

    model_checkpoint_callback = ModelCheckpoint(
        monitor=cfg.early_stopping.metric,
        mode=cfg.early_stopping.goal,
        filename="checkpoint-{epoch:02d}-{" + cfg.early_stopping.metric + ":0.3f}",
        save_last=True,
    )

    model, trainer, out_dir, wandb_logger = make_trainer(
        cfg,
        dummy_batch=train_ds.dummy_batch(cfg.dataset.batch_size),
        crossval_fold=crossval_fold,
        crossval_id=crossval_id,
        run_prefix=run_prefix,
        callbacks=[model_checkpoint_callback],
    )
    print(model)

    if cfg.tune_lr:
        tuner = pl.tuner.tuning.Tuner(trainer)
        tuner.lr_find(model=model, train_dataloaders=train_dl, val_dataloaders=valid_dl)
        logger.info(f"Best learning rate: {model.learning_rate}")

    trainer.fit(model=model, train_dataloaders=train_dl, val_dataloaders=valid_dl)

    torch.save(model_checkpoint_callback.state_dict(), out_dir / "checkpoints.pth")

    if cfg.restore_best_checkpoint:
        model = model.load_from_checkpoint(model_checkpoint_callback.best_model_path, cfg=cfg)

    predictions = flatten_batched_dicts(trainer.predict(model=model, dataloaders=valid_dl, return_predictions=True))

    preds_df = make_preds_df(
        predictions=predictions,
        base_df=valid_df,
        categories={target.column: target.classes for target in cfg.dataset.targets},
    )
    preds_df.to_csv(out_dir / "valid-patient-preds.csv")

    if cfg.test.enabled:
        test(cfg, model, trainer, encoders, out_dir)

    wandb_logger.experiment.finish()

    return model, trainer, out_dir, wandb_logger


def test(
    cfg: DictConfig,
    model: LitMilTransformer,
    trainer: pl.Trainer,
    target_encoders: Mapping[str, TargetEncoder],
    out_dir: Path,
):
    test_df = load_dataset_df(cfg.test.dataset)
    test_targets = {t: encoder(test_df) for t, encoder in target_encoders.items()}
    test_ds = FeatureDataset(
        patient_ids=test_df.index,
        bags=test_df.path.values,
        targets=test_targets,
        instances_per_bag=cfg.dataset.instances_per_bag,
        augmentations=cfg.dataset.augmentations.test,
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=cfg.dataset.batch_size,
        num_workers=cfg.dataset.num_workers,
        pin_memory=True,
        collate_fn=test_ds.collate_fn,
    )

    trainer.test(model=model, dataloaders=test_dl)

    predictions = flatten_batched_dicts(trainer.predict(model=model, dataloaders=test_dl, return_predictions=True))
    preds_df = make_preds_df(
        predictions=predictions,
        base_df=test_df,
        categories={target.column: target.classes for target in cfg.test.dataset.targets},
    )
    preds_df.to_csv(out_dir / "test-patient-preds.csv")


def setup(func):
    @functools.wraps(func)
    def wrapper(cfg: DictConfig):
        pl.seed_everything(cfg.seed)
        torch.set_float32_matmul_precision("medium")

        dataset_df = load_dataset_df(cfg.dataset)
        folds = get_folds(cfg, dataset_df)
        return func(cfg, dataset_df=dataset_df, folds=folds)

    return wrapper


@hydra.main(config_path=str(Path(histaug.__file__).parent.with_name("conf")), config_name="config", version_base="1.3")
@setup
def train_crossval(cfg: DictConfig, dataset_df: pd.DataFrame, folds: pd.Series) -> None:
    crossval_id = ""
    for fold in sorted(folds.unique()):
        model, trainer, out_dir, wandb_logger = train_fold(
            cfg, dataset_df, folds, crossval_id=crossval_id, crossval_fold=fold, run_prefix="crossval"
        )
        if not crossval_id:
            crossval_id = wandb_logger.experiment.config["crossval_id"]


@hydra.main(config_path=str(Path(histaug.__file__).parent.with_name("conf")), config_name="config", version_base="1.3")
@setup
def train_nocrossval(cfg: DictConfig, dataset_df: pd.DataFrame, folds: pd.Series) -> None:
    train_fold(cfg, dataset_df, folds, crossval_id=None, crossval_fold=None, run_prefix="train")


@hydra.main(config_path=str(Path(histaug.__file__).parent.with_name("conf")), config_name="config", version_base="1.3")
@setup
def train_onecrossval(cfg: DictConfig, dataset_df: pd.DataFrame, folds: pd.Series) -> None:
    train_fold(cfg, dataset_df, folds, crossval_id=None, crossval_fold=0, run_prefix="train")
