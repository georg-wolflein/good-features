from torch.utils.data import DataLoader
import torch
import itertools
from tqdm import tqdm
from pathlib import Path
from loguru import logger
from typing import List, Dict, NamedTuple, Sequence
import numpy as np
import zarr

from histaug.data import Kather100k
from histaug.augmentations import load_augmentations
from histaug.feature_extractors import load_feature_extractor, FEATURE_EXTRACTORS


def process_dataset(loader, model, augmentations, device="cuda", n_batches: int = None):
    model.to(device)
    augmentations.to(device)
    n_batches = n_batches or len(loader)

    with torch.no_grad():
        all_labels = []
        all_feats = []
        all_feats_augs = {aug_name: [] for aug_name in augmentations}
        all_files = []

        for imgs, labels, files in tqdm(
            itertools.islice(loader, n_batches), desc="Processing dataset", total=n_batches
        ):
            imgs = imgs.to(device)
            feats = model(imgs)

            feats_augs = {
                aug_name: model(aug(imgs)) for aug_name, aug in augmentations.items()
            }  # dict of {aug_name: feats_aug} where feats_aug is a tensor of shape (n_imgs, n_feats)

            all_labels.append(labels.detach().cpu())
            all_feats.append(feats.detach().cpu())
            all_files.extend(files)
            for aug_name, feats_aug in feats_augs.items():
                all_feats_augs[aug_name].append(feats_aug.detach().cpu())

        labels = torch.cat(all_labels)
        feats = torch.cat(all_feats)
        feats_augs = {aug_name: torch.cat(feats_augs) for aug_name, feats_augs in all_feats_augs.items()}
        files = all_files
        return feats, feats_augs, labels, files


def save_features(
    file: Path,
    feats: torch.Tensor,
    feats_augs: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    files: list,
    classes: List[str],
    chunk_size: int = 2048,
):
    f = zarr.open_group(str(file), mode="w")
    f.attrs["classes"] = classes
    f.create_dataset("labels", data=labels.numpy(), chunks=False)
    f.create_dataset("files", data=np.array(files, dtype=str), chunks=False)
    f.create_dataset("feats", data=feats.numpy(), chunks=(chunk_size, *feats.shape[1:]))
    aug_group = f.create_group("feats_augs")
    for aug_name, feats_aug in feats_augs.items():
        aug_group.create_dataset(aug_name, data=feats_aug.numpy(), chunks=(chunk_size, *feats_aug.shape[1:]))


class LoadedFeatures(NamedTuple):
    feats: np.ndarray
    feats_augs: Dict[str, np.ndarray]
    labels: np.ndarray
    files: np.ndarray


def load_features(path: Path, remove_classes: Sequence[str] = ()) -> LoadedFeatures:
    f = zarr.open_group(str(path), mode="r")
    classes = np.array(f.attrs["classes"])

    feats = f["feats"][:]
    labels = classes[f["labels"][:]]
    files = f["files"][:]

    feats_augs = {k: f["feats_augs"][k][:] for k in f["feats_augs"].keys()}

    # Remove classes
    remove_mask = np.isin(labels, remove_classes)
    feats = feats[~remove_mask]
    labels = labels[~remove_mask]
    files = files[~remove_mask]
    feats_augs = {k: v[~remove_mask] for k, v in feats_augs.items()}
    return LoadedFeatures(feats=feats, feats_augs=feats_augs, labels=labels, files=files)


if __name__ == "__main__":
    torch.manual_seed(42)
    import argparse

    parser = argparse.ArgumentParser(description="Extract features and augmented features from a dataset")
    parser.add_argument("--dataset", type=Path, default="/data/NCT-CRC-HE-100K", help="Path to the Kather100k dataset")
    parser.add_argument("--output", type=Path, default="kather100k.zarr", help="Path to the output file")
    parser.add_argument(
        "--model",
        type=str,
        choices=list(FEATURE_EXTRACTORS.keys()),
        default="ctranspath",
        help="Feature extractor model",
    )
    parser.add_argument("--n-batches", type=int, default=None, help="Number of batches to process. Defaults to all.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for feature extraction")
    args = parser.parse_args()

    ds = Kather100k(args.dataset)
    logger.info(f"Loaded dataset with {len(ds)} samples and {len(ds.classes)} classes")

    loader = DataLoader(
        ds, batch_size=128, shuffle=True, num_workers=8, pin_memory=True
    )  # shuffle is true because image augmentations are done across the whole batch (i.e. same rotation angle for all images per batch)
    model = load_feature_extractor(args.model)
    augmentations = load_augmentations()

    logger.info("Processing dataset")
    feats, feats_augs, labels, files = process_dataset(
        loader, model, augmentations, device=args.device, n_batches=args.n_batches
    )

    logger.info(f"Saving features to {args.output}")
    save_features(file=args.output, feats=feats, feats_augs=feats_augs, labels=labels, files=files, classes=ds.classes)
