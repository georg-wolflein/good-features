from torch.utils.data import Dataset
from pathlib import Path
from typing import Union, Optional, Sequence
import zarr
import numpy as np
from loguru import logger
import itertools

from ..augmentations import augmentation_names


class FeatureDataset(Dataset):
    def __init__(
        self,
        root: Union[str, Path],
        augmentations: Sequence[str] = (None, *augmentation_names()),
        max_patches: Optional[int] = None,
    ):
        """This dataset yields feature vectors for one slide at a time.

        Args:
            root (Union[str, Path]): Path to the root directory of the dataset.
            augmentations (Sequence[str], optional): Augmentations to apply. Be sure to include None as an augmentation; this is the original feature vector with no augmentation applied.
            max_patches (Optional[int], optional): Maximum number of patches to return per slide. Defaults to None (all patches).
        """

        root = Path(root)

        if None not in augmentations:
            logger.warning(
                "Loading feature dataset without None augmentation means that there is a 0% chance of selecting the original feature vector."
            )

        self.augmentations = augmentations
        self.max_patches = max_patches
        self.slides = sorted(root.glob("*.zarr"))

    def __getitem__(self, index):
        slide = self.slides[index]
        features = zarr.open_group(str(slide), mode="r")
        total_num_patches = features["feats"].shape[0]
        num_patches = min(total_num_patches, self.max_patches or float("inf"))
        augmentations_per_patch = np.random.randint(len(self.augmentations), size=(num_patches,))
        indices = np.random.permutation(total_num_patches)[:num_patches]
        indices_per_augmentation = {
            augmentation: indices[augmentations_per_patch == i] for i, augmentation in enumerate(self.augmentations)
        }
        features_per_augmentation = {
            augmentation: features["feats" if augmentation is None else f"feats_augs/{augmentation}"][indices]
            for augmentation, indices in indices_per_augmentation.items()
        }
        features = np.concatenate(list(features_per_augmentation.values()))  # (num_patches, feature_dim)
        indices = np.concatenate(list(indices_per_augmentation.values()))  # (num_patches,)
        augmentations = itertools.chain.from_iterable(
            [aug] * len(indices) for aug, indices in indices_per_augmentation.items()
        )  # provide a generator for efficiency (only compute the augmentations if needed)
        return features, indices, augmentations, slide.stem

    def transform(self, patches):
        return patches

    def inverse_transform(self, patches):
        return patches

    def __len__(self):
        return len(self.slides)
