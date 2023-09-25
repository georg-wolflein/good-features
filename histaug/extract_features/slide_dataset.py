from torch.utils.data import DataLoader
import torch
from torch import nn
from tqdm import tqdm
from pathlib import Path
from loguru import logger
import shutil

from ..data import SlidesDataset
from ..augmentations import load_augmentations, Augmentations
from ..feature_extractors import load_feature_extractor, FEATURE_EXTRACTORS, FEATURE_EXTRACTORS_NORM
from ..utils import save_features, check_version
from .augmented_feature_extractor import AugmentedFeatureExtractor


@torch.no_grad()
def process_dataset(
    ds: SlidesDataset,
    model: nn.Module,
    augmentations: Augmentations,
    output_folder: Path,
    device="cuda",
    num_workers: int = 8,
):
    augmented_feature_extractor = AugmentedFeatureExtractor(model, augmentations)
    augmented_feature_extractor.to(device)

    for slide in tqdm(ds, desc="Processing slides", position=0, leave=True):
        output_file = output_folder / f"{slide.name}.zarr"
        if output_file.exists() and check_version(output_file, throw=False):
            logger.info(f"Skipping slide {slide.name}, output file {output_file} already exists")
            continue

        shutil.rmtree(output_file, ignore_errors=True)

        all_feats = []
        all_feats_augs = {aug_name: [] for aug_name in augmentations}

        loader = DataLoader(
            slide, batch_size=None, shuffle=False, num_workers=num_workers, pin_memory=True
        )  # shuffle must be False to keep patches in order (so that we save them in order, as they are saved per slide)

        for patches, coords in tqdm(loader, desc=f"Processing patches in {slide.name}", position=1, leave=False):
            imgs = patches.to(device)
            feats, feats_augs = augmented_feature_extractor(imgs)

            all_feats.append(feats.detach().cpu())
            for aug_name, feats_aug in feats_augs.items():
                all_feats_augs[aug_name].append(feats_aug.detach().cpu())

        feats = torch.cat(all_feats)
        feats_augs = {aug_name: torch.cat(feats_augs) for aug_name, feats_augs in all_feats_augs.items()}
        save_features(output_file, feats, feats_augs, slide.coords)


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
    parser.add_argument("--start", type=int, default=0, help="Index of the first slide to process")
    parser.add_argument("--end", type=int, default=None, help="Index of the last slide to process")
    args = parser.parse_args()

    output_folder = args.output / args.model
    output_folder.mkdir(parents=True, exist_ok=True)

    norm = FEATURE_EXTRACTORS_NORM[args.model]

    ds = SlidesDataset(
        args.dataset,
        batch_size=args.batch_size,
        start=args.start,
        end=args.end,
        mean=norm.mean,
        std=norm.std,
    )  # dataset already loads patches in batches

    logger.info(f"Loaded dataset with {len(ds)} slides, will process in batches of {args.batch_size} patches")

    model = load_feature_extractor(args.model)
    augmentations = load_augmentations()

    logger.info(f"Processing dataset, saving features to {output_folder}")
    process_dataset(ds, model, augmentations, output_folder, device=args.device, num_workers=8)
