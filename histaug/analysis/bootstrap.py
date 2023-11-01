import wandb
import pandas as pd
from loguru import logger
from tqdm import tqdm
from pathlib import Path
import numpy as np
import torch
from torchmetrics.classification import MulticlassAUROC

from histaug.utils import cached_df

TRAIN_DIR = Path("/data/histaug/train")
BOOTSTRAPS_DIR = Path("/data/histaug/bootstraps")
N_BOOTSTRAPS = 1000


def filter_runs(runs, filters: dict):
    return [run for run in runs if all(getattr(run, key, None) == value for key, value in filters.items())]


def get_test_preds(run) -> pd.DataFrame:
    return pd.read_csv(next(iter(TRAIN_DIR.glob(f"*/{run.id}"))) / "test-patient-preds.csv").set_index("PATIENT")


def load_bootstraps(sample_run, n_bootstraps: int = 1000):
    # Load test-patient-preds.csv for a run and generate bootstraps for it
    dataset = sample_run.config["dataset"]["name"]
    column = sample_run.config["dataset"]["targets"][0]["column"]
    classes = sample_run.config["dataset"]["targets"][0]["classes"]

    bootstrap_csv = BOOTSTRAPS_DIR / f"{dataset}_{column}_{n_bootstraps}.csv"
    if not bootstrap_csv.exists():
        logger.debug(f"Caching bootstraps for {dataset} {column} at {bootstrap_csv}")
        df = get_test_preds(sample_run)
        patients = df.index.unique()
        bootstraps = np.random.choice(patients, size=(n_bootstraps, len(patients)), replace=True)
        # Ensure that there is at least one instance from each class in each bootstrap
        for i, bootstrap in enumerate(bootstraps):
            while not all(df.loc[bootstrap, column].astype(str).value_counts().reindex(classes).fillna(0) > 0):
                bootstraps[i] = np.random.choice(patients, size=len(patients), replace=True)
        bootstraps_df = pd.DataFrame(bootstraps)
        BOOTSTRAPS_DIR.mkdir(parents=True, exist_ok=True)
        bootstraps_df.to_csv(bootstrap_csv, index=False, header=False)

    return pd.read_csv(bootstrap_csv, header=None).values


def compute_auroc(df, column, classes):
    auroc = MulticlassAUROC(num_classes=len(classes))

    logits = df[[f"{column}_{c}" for c in classes]].values
    y_true = df[column].map(lambda x: classes.index(str(x))).values
    auroc.update(torch.from_numpy(logits), torch.from_numpy(y_true))
    return auroc.compute().numpy()


def compare_auroc_diffs(runs_a, runs_b, bootstraps, column, classes):
    auroc_diffs = []
    for seed, (run_a, run_b) in enumerate(zip(runs_a, runs_b)):
        df_a = get_test_preds(run_a)
        df_b = get_test_preds(run_b)
        for bootstrap in bootstraps:
            df_bootstrap_a = df_a.loc[bootstrap]
            df_bootstrap_b = df_b.loc[bootstrap]
            diff = compute_auroc(df_bootstrap_a, column, classes) - compute_auroc(df_bootstrap_b, column, classes)
            auroc_diffs.append(diff)
    auroc_diffs = np.array(auroc_diffs)
    mean = auroc_diffs.mean()
    std = auroc_diffs.std()
    ci_lo, ci_hi = np.percentile(auroc_diffs, [2.5, 97.5])
    return {"mean": mean, "std": std, "ci_lo": ci_lo, "ci_hi": ci_hi}


@cached_df(lambda augmentation_a, augmentation_b, *args, **kwargs: f"bootstrapped_{augmentation_a}_vs_{augmentation_b}")
def compare_bootstraps(augmentation_a, augmentation_b):
    logger.info("Loading runs...")
    api = wandb.Api()
    runs = list(api.runs("histaug"))
    runs = filter_runs(runs, {"state": "finished"})
    datasets = [*[f"tcga_brca_{target}" for target in ["subtype", "CDH1", "PIK3CA", "TP53"]]]  # TODO: add other targets
    feature_extractors = ["ctranspath", "swin", "owkin", "vit", "resnet50", "retccl", "bt", "swav", "dino_p16"]
    models = ["Transformer", "MeanAveragePooling", "AttentionMIL"]

    augmentation_a = "none"
    augmentation_b = "Macenko_patchwise"

    configs = []

    for dataset in datasets:
        for feature_extractor in feature_extractors:
            for model in models:
                configs.append((dataset, feature_extractor, model))

    results = []
    for dataset, feature_extractor, model in (pbar := tqdm(configs)):
        pbar.set_description(f"{dataset} {feature_extractor} {model}")
        filtered_runs = [
            run
            for run in runs
            if run.config["dataset"]["name"] == dataset
            and run.config["settings"]["feature_extractor"] == feature_extractor
            and run.config["model"]["_target_"] == f"histaug.train.models.{model}"
        ]
        runs_a = [run for run in filtered_runs if run.config["dataset"]["augmentations"]["name"] == augmentation_a]
        runs_b = [run for run in filtered_runs if run.config["dataset"]["augmentations"]["name"] == augmentation_b]

        seeds = sorted({run.config["seed"] for run in [*runs_a, *runs_b]})
        assert len(seeds) == len(runs_a) == len(runs_b)

        runs_a = {run.config["seed"]: run for run in runs_a}
        runs_b = {run.config["seed"]: run for run in runs_b}
        runs_a = [runs_a[seed] for seed in seeds]
        runs_b = [runs_b[seed] for seed in seeds]

        # Get info from a sample run
        sample_run = runs_a[0]
        column = sample_run.config["dataset"]["targets"][0]["column"]
        classes = sample_run.config["dataset"]["targets"][0]["classes"]
        train_dataset = sample_run.config["dataset"]["name"]
        test_dataset = sample_run.config["test"]["dataset"]["name"]

        bootstraps = load_bootstraps(sample_run, n_bootstraps=N_BOOTSTRAPS // len(seeds))
        result = compare_auroc_diffs(runs_a, runs_b, bootstraps=bootstraps, column=column, classes=classes)
        results.append(
            {
                "train_dataset": train_dataset,
                "test_dataset": test_dataset,
                "feature_extractor": feature_extractor,
                "model": model,
                "target": column,
                **result,
            }
        )

    return pd.DataFrame(results).set_index(["train_dataset", "test_dataset", "target", "feature_extractor", "model"])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--a", type=str, default="none", help="Augmentation A (default: none)")
    parser.add_argument(
        "--b", type=str, default="Macenko_patchwise", help="Augmentation B (default: Macenko_patchwise)"
    )
    args = parser.parse_args()
    compare_bootstraps(args.a, args.b)
