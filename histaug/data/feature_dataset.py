from torch.utils.data import Dataset
from pathlib import Path
from typing import Union, Optional, Sequence, Mapping
import torch
import zarr
import numpy as np
from loguru import logger
import itertools

from ..augmentations import augmentation_names


class FeatureDataset(Dataset):
    def __init__(
        self,
        bags: Sequence[Union[str, Path]],
        targets: Optional[Mapping[str, torch.Tensor]],
        instances_per_bag: Optional[int] = None,
        pad: bool = False,
        augmentations: Sequence[str] = (None, *augmentation_names()),
    ):
        """This dataset yields feature vectors for one slide at a time.

        Args:
            bags (Sequence[Union[str, Path]]): Paths to bags of features.
            instances_per_bag (Optional[int], optional): Number of instances to sample from each bag. Defaults to None (all instances).
            pad (bool, optional): If the number of instances in a bag is less than instances_per_bag, pad the remaining instances with zeros. Defaults to False.
            augmentations (Sequence[str], optional): Augmentations to apply. Be sure to include None as an augmentation; this is the original feature vector with no augmentation applied.
        """

        if None not in augmentations:
            logger.warning(
                "Loading feature dataset without None augmentation means that there is a 0% chance of selecting the original feature vector."
            )

        assert (
            not pad or instances_per_bag is not None
        ), "If padding is enabled, you must specify the number of instances per bag."

        self.instances_per_bag = instances_per_bag
        self.pad = pad
        self.augmentations = augmentations
        self.slides = sorted(
            (sorted((Path(b) for b in bag), key=lambda b: b.stem)[0] for bag in bags), key=lambda b: b.stem
        )  # NOTE: this is a dirty hack; we are using the first bag in each list of bags, but we should be using all bags in each list
        self.targets = targets

    def __getitem__(self, index):
        slide = self.slides[index]
        features = zarr.open_group(str(slide), mode="r")
        total_num_patches = features["feats"].shape[0]
        num_patches = min(total_num_patches, self.instances_per_bag or float("inf"))
        augmentations_per_patch = np.random.randint(len(self.augmentations), size=(num_patches,))
        indices = np.random.permutation(total_num_patches)[:num_patches]
        indices_per_augmentation = {
            augmentation: indices[augmentations_per_patch == i] for i, augmentation in enumerate(self.augmentations)
        }
        features_per_augmentation = {
            augmentation: features["feats" if augmentation is None else f"feats_augs/{augmentation}"][indices]
            for augmentation, indices in indices_per_augmentation.items()
        }

        feats = np.concatenate(list(features_per_augmentation.values()))
        coords = features["coords"][indices]

        augmentations = np.array(
            list(
                itertools.chain.from_iterable(
                    [self.augmentations.index(aug)] * len(indices) for aug, indices in indices_per_augmentation.items()
                )
            )
        )
        labels = {label: target[index] for label, target in self.targets.items()} if self.targets else None

        return feats, coords, labels, augmentations, indices, slide.stem

    def transform(self, patches):
        return patches

    def inverse_transform(self, patches):
        return patches

    def __len__(self):
        return len(self.slides)

    def dummy_batch(self, batch_size: int):
        """Create a dummy batch of the largest possible size"""
        sample_feats, sample_coords, sample_labels, *_ = self[0]
        d_model = sample_feats.shape[-1]
        instances_per_bag = self.instances_per_bag or sample_feats.shape[-2]
        tile_tokens = torch.rand((batch_size, instances_per_bag, d_model))
        tile_positions = torch.rand((batch_size, instances_per_bag, 2)) * 100
        labels = {label: value.expand(batch_size, *value.shape) for label, value in sample_labels.items()}
        indices = np.expand_dims(torch.arange(self.instances_per_bag), axis=0).repeat(batch_size, axis=0)
        augmentations = np.zeros((batch_size, instances_per_bag), dtype=int)
        return tile_tokens, tile_positions, labels, augmentations, indices, "sample"
