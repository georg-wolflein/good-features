from torch.utils.data import Dataset
from pathlib import Path
from typing import Union, Optional
import zarr
import torch
import math
from torchvision import transforms as T


class SlideDataset(Dataset):
    def __init__(
        self,
        slide: Union[str, Path],
        transform,
        inverse_transform,
        batch_size: Optional[int] = None,
    ):
        slide = Path(slide)
        self.zarr_group = zarr.open_group(str(slide), mode="r")
        self.batch_size = batch_size
        self.num_patches = self.zarr_group["patches"].shape[0]
        self.num_batches = math.ceil(self.num_patches / self.batch_size) if self.batch_size else 1
        self.name = slide.stem
        self.transform = transform
        self.inverse_transform = inverse_transform

    def __getitem__(self, index):
        start = index * self.batch_size
        end = min(start + self.batch_size, self.num_patches)
        patches = self.zarr_group["patches"][start:end]
        coords = self.zarr_group["coords"][start:end]
        return self.transform(patches), torch.from_numpy(coords)

    def __len__(self):
        return self.num_batches

    @property
    def coords(self):
        return self.zarr_group["coords"][:]


class SlidesDataset:
    def __init__(
        self,
        root: Union[str, Path],
        batch_size: Optional[int] = None,
        start: int = 0,
        end: int = None,
    ):
        """This dataset is a collection of patches from the slides in the root directory.
        Each element of the dataset is a batch of patches from a single slide.

        Args:
            root (Union[str, Path]): Path to the root directory of the dataset.
            batch_size (Optional[int], optional): Number of patches per iteration. Defaults to None (all patches).
            start (int, optional): Index of the first slide to include. Defaults to 0.
            end (int, optional): Index of the last slide to include. Defaults to None (all slides).
        """
        super().__init__()

        self.root = Path(root)
        slides = sorted(self.root.glob("*.zarr"))
        end = end or len(slides)
        self.slides = slides[start:end]
        self.batch_size = batch_size
        self.transform = T.Lambda(
            lambda patches: (torch.from_numpy(patches).float() / 255.0).permute(0, 3, 1, 2)
        )  # we will normalize later (in the feature extractor's forward() method)
        self.inverse_transform = T.Lambda(lambda patches: (patches * 255).byte().permute(0, 2, 3, 1))

    def __getitem__(self, index) -> SlideDataset:
        return SlideDataset(
            self.slides[index],
            batch_size=self.batch_size,
            transform=self.transform,
            inverse_transform=self.inverse_transform,
        )

    def __len__(self):
        return len(self.slides)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
