import hydra
from pathlib import Path
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm
import pytorch_lightning as pl
from typing import Sequence
import zarr
import shutil

import histaug
from .utils import make_dataset_df, pathlist
from ..data import FeatureDataset


@hydra.main(config_path=str(Path(histaug.__file__).parent.with_name("conf")), config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    pl.seed_everything(42)
    dataset_cfg = cfg.dataset
    augmentations = sorted(cfg.dataset.augmentations.train)
    dataset_df = make_dataset_df(
        slide_tables=pathlist(dataset_cfg.slide_tables),
        feature_dirs=pathlist(dataset_cfg.feature_dirs),
        patient_col=dataset_cfg.patient_col,
        filename_col=dataset_cfg.filename_col,
        target_labels=None,
    )
    ds = FeatureDataset(
        patient_ids=dataset_df.index,
        bags=dataset_df.path.values,
        targets=None,
        instances_per_bag=cfg.dataset.instances_per_bag,
        augmentations=augmentations,
    )
    loader = DataLoader(
        ds,
        batch_size=None,
        num_workers=cfg.dataset.num_workers,
        pin_memory=False,
        shuffle=False,
    )
    cache_dir = Path(cfg.dataset.cache_dir)
    shutil.rmtree(cache_dir, ignore_errors=True)

    for epoch in tqdm(range(cfg.max_epochs), desc="Epochs", position=0):
        epoch_cache_dir = cache_dir / f"epoch_{epoch:03d}"
        epoch_cache_dir.mkdir(parents=True, exist_ok=True)
        for feats, coords, _, patient_id in tqdm(loader, desc="Batches", position=1):
            batch_cache_file = epoch_cache_dir / f"{patient_id}.zarr"
            batch_cache_file.parent.mkdir(parents=True, exist_ok=True)
            f = zarr.open(str(batch_cache_file), mode="w")
            f.attrs["patient_id"] = patient_id
            f.attrs["epoch"] = epoch
            f.attrs["augmentations"] = sorted(augmentations)
            f.create_dataset("feats", data=feats.numpy(), chunks=-1)
            f.create_dataset("coords", data=coords.numpy(), chunks=-1)


if __name__ == "__main__":
    main()
