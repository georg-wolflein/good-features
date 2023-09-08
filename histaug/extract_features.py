from torch.utils.data import DataLoader
import torch
import itertools
from tqdm import tqdm
from pathlib import Path
from loguru import logger
from typing import List, Dict
import h5py

from histaug.data import Kather100k
from histaug.augmentations import load_augmentations
from histaug.feature_extractors import load_feature_extractor


def process_dataset(loader, model, augmentations, device="cuda", batches: int = None):
    model.to(device)
    augmentations.to(device)
    batches = batches or len(loader)

    with torch.no_grad():
        all_labels = []
        all_feats = []
        all_feats_augs = {aug_name: [] for aug_name in augmentations}
        all_files = []

        for imgs, labels, files in tqdm(itertools.islice(loader, batches), desc="Processing dataset", total=batches):
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
):
    with h5py.File(file, "w") as f:
        f.attrs["classes"] = classes
        f.create_dataset("labels", data=labels.numpy())
        f.create_dataset("files", data=files)
        f.create_dataset("feats", data=feats.numpy())
        aug_group = f.create_group("feats_augs")
        for aug_name, feats_aug in feats_augs.items():
            aug_group.create_dataset(aug_name, data=feats_aug.numpy())


def load_features(path: Path):
    with h5py.File(path, "r") as f:
        classes = f.attrs["classes"]

        feats = f["feats"][:]
        labels = classes[f["labels"]]
        files = f["files"][:]

        feats_augs = {k: f["feats_augs"][k][:] for k in f["feats_augs"].keys()}
    return feats, feats_augs, labels, files


if __name__ == "__main__":
    torch.manual_seed(42)
    import argparse

    parser = argparse.ArgumentParser(description="Extract features and augmented features from a dataset")
    parser.add_argument("--dataset", type=Path, default="/data/NCT-CRC-HE-100K", help="Path to the Kather100k dataset")
    parser.add_argument("--output", type=Path, default="kather100k.h5", help="Path to the output file")
    parser.add_argument(
        "--model",
        type=str,
        choices=["ctranspath", "retccl", "resnet50"],
        default="ctranspath",
        help="Feature extractor model",
    )
    args = parser.parse_args()

    ds = Kather100k(args.dataset)
    logger.info(f"Loaded dataset with {len(ds)} samples and {len(ds.classes)} classes")

    loader = DataLoader(
        ds, batch_size=128, shuffle=True, num_workers=8, pin_memory=True
    )  # shuffle is true because image augmentations are done across the whole batch (i.e. same rotation angle for all images per batch)
    model = load_feature_extractor(args.model)
    augmentations = load_augmentations()

    logger.info("Processing dataset")
    feats, feats_augs, labels, files = process_dataset(loader, model, augmentations, device="cuda")

    logger.info(f"Saving features to {args.output}")
    save_features(file=args.output, feats=feats, feats_augs=feats_augs, labels=labels, files=files, classes=ds.classes)
