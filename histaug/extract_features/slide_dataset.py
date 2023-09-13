from torch.utils.data import DataLoader
import torch
from torch import nn
import itertools
from tqdm import tqdm
from pathlib import Path
from loguru import logger
from typing import Optional

from ..data import SlideDataset
from ..augmentations import load_augmentations, Augmentations
from ..feature_extractors import load_feature_extractor, FEATURE_EXTRACTORS
from ..utils import save_features, GroupedLoader
from .augmented_feature_extractor import AugmentedFeatureExtractor


@torch.no_grad()
def process_dataset(
    loader: DataLoader,
    model: nn.Module,
    augmentations: Augmentations,
    output_folder: Path,
    device="cuda",
    num_slides: Optional[int] = None,
):
    augmented_feature_extractor = AugmentedFeatureExtractor(model, augmentations)
    augmented_feature_extractor.to(device)

    loader = GroupedLoader(iter(loader), extract_group_from_item=lambda x: x.slide)

    for slide, patch_loader in tqdm(loader, desc="Processing slides", total=num_slides, position=0, leave=True):
        output_file = output_folder / f"{slide}.zarr"
        if output_file.exists():
            logger.info(f"Skipping slide {slide}, output file {output_file} already exists")
            continue

        all_feats = []
        all_feats_augs = {aug_name: [] for aug_name in augmentations}

        with tqdm(desc=f"Processing patches in {slide}", position=1, leave=False) as pbar:
            for batch in patch_loader:
                pbar.total = batch.num_patches_in_slide

                imgs = batch.patches
                imgs = imgs.to(device)
                feats, feats_augs = augmented_feature_extractor(imgs)

                all_feats.append(feats.detach().cpu())
                for aug_name, feats_aug in feats_augs.items():
                    all_feats_augs[aug_name].append(feats_aug.detach().cpu())
                pbar.update(imgs.shape[0])

        feats = torch.cat(all_feats)
        feats_augs = {aug_name: torch.cat(feats_augs) for aug_name, feats_augs in all_feats_augs.items()}
        save_features(output_file, feats, feats_augs)


if __name__ == "__main__":
    torch.manual_seed(42)
    import argparse

    parser = argparse.ArgumentParser(description="Extract features and augmented features from a dataset")
    parser.add_argument(
        "--dataset", type=Path, default="/data/shiprec/TCGA-BRCA-DX", help="Path to the dataset created using shiprec"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size with which to load the dataset and apply augmentations (should not be too large because the same augmentation is applied to all images in a batch)",
    )
    parser.add_argument(
        "--output", type=Path, default="/data/histaug/results/TCGA-BRCA-DX", help="Path to the output folder"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=list(FEATURE_EXTRACTORS.keys()),
        default="ctranspath",
        help="Feature extractor model",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for feature extraction")
    args = parser.parse_args()

    output_folder = args.output / args.model
    output_folder.mkdir(parents=True, exist_ok=True)

    ds = SlideDataset(args.dataset, batch_size=args.batch_size)  # dataset already loads patches in batches
    loader = DataLoader(
        ds, batch_size=None, shuffle=False, num_workers=8, pin_memory=True
    )  # shuffle must be False to keep patches in order (so that we save them in order, as they are saved per slide)

    logger.info(f"Loaded dataset with {len(ds)} batches from {ds.num_slides} slides")

    model = load_feature_extractor(args.model)
    augmentations = load_augmentations()

    logger.info(f"Processing dataset, saving features to {output_folder}")
    feats, feats_augs, labels, files = process_dataset(
        loader, model, augmentations, output_folder, device=args.device, num_slides=ds.num_slides
    )
