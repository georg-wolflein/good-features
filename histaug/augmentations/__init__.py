from pathlib import Path
import cv2
import torch
from torch import nn
from loguru import logger
from typing import Union, Sequence, Callable, Any, Mapping
from torchvision import transforms as T
from torchvision.transforms import functional as TF
from kornia import augmentation as K
from functools import partial

import histaug
from .macenko_torchstain import TorchMacenkoNormalizer, FullyTransparentException

PATCH_SIZE = 224

__all__ = ["load_augmentations", "Augmentations"]


class Macenko(nn.Module):
    def __init__(
        self,
        target_image: Path = Path(histaug.__file__).parent.parent / "normalization_template.jpg",
    ):
        super().__init__()
        target = cv2.imread(str(target_image))
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        target = torch.from_numpy(target).permute(2, 0, 1)
        self.normalizer = TorchMacenkoNormalizer()
        logger.info(f"Fitting Macenko normalizer to {target_image}")
        self.normalizer.fit(target)

    def forward_image(self, image: torch.Tensor) -> torch.Tensor:
        try:
            Inorm, H, E = self.normalizer.normalize((image * 255).type(torch.uint8))
            return Inorm.type_as(image) / 255
        except FullyTransparentException as e:
            logger.warning(
                f"Attempting to Macenko normalize fully transparent image ({str(e)}). Returning original image."
            )
            return image

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.forward_image(img) for img in images])


def EnlargeAndCenterCrop(zoom_factor: Union[float, int] = 2, patch_size=PATCH_SIZE):
    return T.Compose([T.Resize(int(zoom_factor * patch_size), antialias=True), T.CenterCrop(patch_size)])


class Augmentations(nn.Module):
    def __init__(self, items=dict()):
        super().__init__()
        self._items = dict()
        self.update(items)

    def items(self):
        return self._items.items()

    def keys(self):
        return self._items.keys()

    def values(self):
        return self._items.values()

    def __contains__(self, key):
        return self._items.__contains__(key)

    def __repr__(self):
        return self._items.__repr__()

    def __iter__(self):
        return self._items.__iter__()

    def __len__(self):
        return self._items.__len__()

    def __getitem__(self, key):
        return self._items.__getitem__(key)

    def __setitem__(self, key, value):
        if isinstance(value, nn.Module):
            self.add_module(key.replace(" ", "_").replace("°", "").replace(".", "_"), value)
        self._items.__setitem__(key, value)

    def update(self, items):
        for key, value in items.items():
            self.__setitem__(key, value)


_unloaded_augmentations: Mapping[str, Callable[[], Any]] = {
    "Macenko": lambda: Macenko(),
    "low brightness": lambda: T.ColorJitter(brightness=(0.7,) * 2),
    "high brightness": lambda: T.ColorJitter(brightness=(1.5,) * 2),
    "low contrast": lambda: T.ColorJitter(contrast=(0.7,) * 2),
    "high contrast": lambda: T.ColorJitter(contrast=(1.5,) * 2),
    "low saturation": lambda: T.ColorJitter(saturation=(0.7,) * 2),
    "high saturation": lambda: T.ColorJitter(saturation=(1.5,) * 2),
    "colour jitter": lambda: T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    "gamma 0.5": lambda: T.Lambda(lambda x: x**0.5),
    "gamma 2.0": lambda: T.Lambda(lambda x: x**2.0),
    "flip horizontal": lambda: T.RandomHorizontalFlip(p=1.0),
    "flip vertical": lambda: T.RandomVerticalFlip(p=1.0),
    "rotate 90°": lambda: partial(TF.rotate, angle=90),
    "rotate 180°": lambda: partial(TF.rotate, angle=180),
    "rotate 270°": lambda: partial(TF.rotate, angle=270),
    "rotate random angle": lambda: T.Lambda(
        lambda img: TF.rotate(img, angle=(torch.randint(0, 4, (1,)) * 90 + torch.randint(10, 80, (1,))).item())
    ),  # rotate by 0, 90, 180, or 270 degrees plus a random angle between 10 and 80 degrees)
    "zoom 1.5x": lambda: EnlargeAndCenterCrop(1.5),
    "zoom 1.75x": lambda: EnlargeAndCenterCrop(1.75),
    "zoom 2x": lambda: EnlargeAndCenterCrop(2),
    "affine": lambda: T.RandomAffine(degrees=10, translate=(0.2, 0.2), scale=(0.8, 1.2), shear=10),
    "warp perspective": lambda: T.RandomPerspective(p=1.0, distortion_scale=0.2, fill=0),
    "jigsaw": lambda: K.RandomJigsaw(p=1.0, grid=(4, 4), same_on_batch=True),
    "Cutout": lambda: T.RandomErasing(p=1.0, scale=(0.02, 0.25), ratio=(0.3, 3.3), value=0, inplace=False),
    "AugMix": lambda: T.Compose(
        [
            T.Lambda(lambda x: (x * 255).type(torch.uint8)),
            T.AugMix(),
            T.Lambda(lambda x: x.type(torch.float32) / 255),
        ],
    ),
    "sharpen": lambda: T.RandomAdjustSharpness(p=1.0, sharpness_factor=5.0),
    "gaussian blur": lambda: T.GaussianBlur(kernel_size=5, sigma=2.0),
    "median blur": lambda: K.RandomMedianBlur(p=1.0, kernel_size=5, same_on_batch=True),
    "gaussian noise": lambda: T.Lambda(lambda x: x + torch.randn_like(x) * 0.1),
}


def load_augmentations() -> Augmentations:
    return Augmentations({k: v() for k, v in _unloaded_augmentations.items()})


def augmentation_names() -> Sequence[str]:
    return list(_unloaded_augmentations.keys())
