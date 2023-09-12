from torch.utils.data import Dataset
from pathlib import Path
from typing import Union, Optional
import zarr
import torch


class SlideDataset(Dataset):
    def __init__(self, root: Union[str, Path], batch_size: Optional[int] = None):
        """This dataset is a collection of patches from the slides in the root directory.
        Each element of the dataset is a batch of patches from a single slide.

        Args:
            root (Union[str, Path]): Path to the root directory of the dataset.
            batch_size (Optional[int], optional): Number of patches per iteration. Defaults to None (all patches).
        """
        super().__init__()

        self.root = Path(root)
        slides = list(self.root.glob("*.zarr"))
        self.zarr_groups = {slide.stem: zarr.open_group(str(slide), mode="r") for slide in slides}
        self.batch_size = batch_size
        self.data = [
            (slide.stem, index)
            for slide in slides
            for index in (
                range(0, self.zarr_groups[slide.stem]["patches"].shape[0], batch_size) if batch_size else [None]
            )
        ]

    def __getitem__(self, index):
        slide, index = self.data[index]
        patches = self.zarr_groups[slide]["patches"]
        if index is None:
            patches = patches[:]
        else:
            patches = patches[index : min(index + self.batch_size, patches.shape[0])]
        patches = self.transform(patches[:])
        return patches, slide, index or 0

    def transform(self, patches):
        return (torch.from_numpy(patches).float() / 255).permute(0, 3, 1, 2)

    def inverse_transform(self, patches):
        return (patches * 255).uint8().permute(0, 2, 3, 1).numpy()

    def __len__(self):
        return len(self.data)
