from torch.utils.data import Dataset
from pathlib import Path
from typing import Union, Optional, Sequence, Mapping, Any
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
        augmentations: Sequence[str] = (None, *augmentation_names()),
    ):
        """This dataset yields feature vectors for one slide at a time.

        Args:
            bags (Sequence[Union[str, Path]]): Paths to bags of features.
            instances_per_bag (Optional[int], optional): Number of instances to sample from each bag. Defaults to None (all instances).
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
        self.slides = list(sorted(Path(b) for b in bag) for bag in bags)
        self.targets = targets

    def __getitem__(self, index):
        slides = self.slides[index]  # we may have multiple slides per patient
        features = [zarr.open_group(str(slide), mode="r") for slide in slides]
        num_patches_per_slide = [f["feats"].shape[0] for f in features]
        total_num_patches = sum(num_patches_per_slide)
        num_patches = min(total_num_patches, self.instances_per_bag or float("inf"))
        augmentations_per_patch = np.random.randint(len(self.augmentations), size=(num_patches,))
        indices = np.random.permutation(total_num_patches)[:num_patches]
        indices = np.concatenate(
            [
                np.stack([np.zeros(n, dtype="long") + i, np.arange(n, dtype="long")], axis=-1)
                for i, n in enumerate(num_patches_per_slide)
            ],
            axis=0,
        )[
            indices
        ]  # indices are now (slide_index, patch_index) pairs
        indices = np.concatenate(
            [indices, np.expand_dims(augmentations_per_patch, axis=-1)], axis=-1
        )  # indices are now (slide_index, patch_index, augmentation_index) triples

        feats = []
        coords = []

        for slide_index, f in enumerate(features):
            slide_indices = indices[indices[:, 0] == slide_index]
            for augmentation_index, augmentation in enumerate(self.augmentations):
                augmentation_indices = slide_indices[slide_indices[:, 2] == augmentation_index]
                feats.append(
                    f["feats" if augmentation is None else f"feats_augs/{augmentation}"][augmentation_indices[:, 1]]
                )
                coords.append(f["coords"][augmentation_indices[:, 1]])

        feats = np.concatenate(feats)
        coords = np.concatenate(coords)
        labels = {label: target[index] for label, target in self.targets.items()} if self.targets else None

        return feats, coords, labels, index

    def transform(self, patches):
        return patches

    def inverse_transform(self, patches):
        return patches

    def __len__(self):
        return len(self.slides)

    def collate_fn(self, batch):
        """Collate a batch of features into a single tensor"""
        feats, coords, labels, indices = zip(*batch)
        n_per_instance = [f.shape[0] for f in feats]
        n_max = max(n_per_instance)
        feats = np.stack([pad(f, n_max, axis=0) for f in feats])
        coords = np.stack([pad(c, n_max, axis=0) for c in coords])
        labels = {label: torch.stack([l[label] for l in labels]) for label in labels[0]} if len(labels) > 0 else None
        mask = torch.arange(n_max) < torch.tensor(n_per_instance).unsqueeze(-1)
        return (
            torch.from_numpy(feats).float(),
            torch.from_numpy(coords).int(),
            mask,
            labels,
            torch.tensor(indices, dtype=torch.long),
        )

    def dummy_batch(self, batch_size: int):
        """Create a dummy batch of the largest possible size"""
        sample_feats, sample_coords, sample_labels, *_ = self[0]
        d_model = sample_feats.shape[-1]
        instances_per_bag = self.instances_per_bag or sample_feats.shape[-2]
        tile_tokens = torch.rand((batch_size, instances_per_bag, d_model))
        tile_positions = torch.rand((batch_size, instances_per_bag, 2)) * 100
        labels = {label: value.expand(batch_size, *value.shape) for label, value in sample_labels.items()}
        indices = torch.arange(batch_size, dtype=torch.long)
        mask = torch.ones((batch_size, instances_per_bag), dtype=torch.bool)
        return tile_tokens, tile_positions, mask, labels, indices


def pad(x: np.ndarray, size: int, axis: int, fill_value: Any = 0):
    """Pad an array with zeros to a given size along a given axis"""
    if x.shape[axis] >= size:
        return x
    pad_width = [(0, 0)] * len(x.shape)
    pad_width[axis] = (0, size - x.shape[axis])
    return np.pad(x, pad_width=pad_width, mode="constant", constant_values=fill_value)
