from torch.utils.data import Dataset
from pathlib import Path
from typing import Union, Optional, NamedTuple
import zarr
import torch


class SlideDatasetItem(NamedTuple):
    patches: torch.Tensor
    slide: str
    patch_index_start: int
    num_patches_in_slide: int


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
        num_patches_per_slide = [zarr.open_group(str(slide), mode="r")["patches"].shape[0] for slide in slides]
        self.zarr_groups = {slide.stem: zarr.open_group(str(slide), mode="r") for slide in slides}
        self.batch_size = batch_size
        self.data = [
            (slide.stem, patch_index_start, num_patches_in_slide)
            for slide, num_patches_in_slide in zip(slides, num_patches_per_slide)
            for patch_index_start in (range(0, num_patches_in_slide, batch_size) if batch_size else [None])
        ]
        self.num_slides = len(slides)

    def __getitem__(self, index) -> SlideDatasetItem:
        slide, patch_index_start, num_patches_in_slide = self.data[index]
        patches = self.zarr_groups[slide]["patches"]
        if patch_index_start is None:
            patches = patches[:]
        else:
            patches = patches[patch_index_start : min(patch_index_start + self.batch_size, patches.shape[0])]
        patches = self.transform(patches[:])
        return SlideDatasetItem(patches, slide, patch_index_start, num_patches_in_slide)

    def transform(self, patches):
        return (torch.from_numpy(patches).float() / 255).permute(0, 3, 1, 2)

    def inverse_transform(self, patches):
        return (patches * 255).uint8().permute(0, 2, 3, 1).numpy()

    def __len__(self):
        return len(self.data)
