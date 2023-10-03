from torch.utils.data import DataLoader
import torch
from torch import nn
import itertools
from tqdm import tqdm
from pathlib import Path
from loguru import logger

from ..data import Kather100k
from ..augmentations import load_augmentations, Augmentations
from ..feature_extractors import load_feature_extractor, FEATURE_EXTRACTORS
from ..utils import save_features
from .augmented_feature_extractor import AugmentedFeatureExtractor


@torch.no_grad()
def process_dataset(loader, model: nn.Module, augmentations: Augmentations, device="cuda", n_batches: int = None):
    augmented_feature_extractor = AugmentedFeatureExtractor(model, augmentations)
    augmented_feature_extractor.to(device)

    all_labels = []
    all_feats = []
    all_feats_augs = {aug_name: [] for aug_name in augmentations}
    all_files = []

    for imgs, labels, files in tqdm(itertools.islice(loader, n_batches), desc="Processing dataset", total=n_batches):
        imgs = imgs.to(device)
        feats, feats_augs, *_ = augmented_feature_extractor(imgs)

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


if __name__ == "__main__":
    torch.manual_seed(42)
    import argparse

    parser = argparse.ArgumentParser(description="Extract features and augmented features from a dataset")
    parser.add_argument(
        "--cache", type=Path, default="/data/cache/huggingface", help="Path to the huggingface cache folder"
    )
    parser.add_argument(
        "--split", type=str, choices=["train_nonorm", "train", "validate"], default="train_nonorm", help="Dataset split"
    )
    parser.add_argument(
        "--output", type=Path, default="/data/histaug/results/kather100k", help="Path to the output folder"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size with which to load the dataset and apply augmentations (should not be too large because the same augmentation is applied to all images in a batch)",
    )
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

    output_folder = args.output / f"{args.model}.zarr"
    output_folder.mkdir(parents=True, exist_ok=True)

    ds = Kather100k(cache_dir=args.cache, split=args.split)
    logger.info(f"Loaded dataset with {len(ds)} samples and {len(ds.classes)} classes")

    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True
    )  # shuffle is true because image augmentations are done across the whole batch (i.e. same rotation angle for all images per batch)
    model = load_feature_extractor(args.model)
    augmentations = load_augmentations()

    logger.info("Processing dataset")
    feats, feats_augs, labels, files = process_dataset(
        loader, model, augmentations, device=args.device, n_batches=args.n_batches
    )

    logger.info(f"Saving features to {output_folder}")
    save_features(
        file=output_folder,
        feats=feats,
        coords=None,
        feats_augs=feats_augs,
        labels=labels,
        files=files,
        classes=ds.classes,
    )
