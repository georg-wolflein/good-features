from abc import ABC, abstractmethod
from pathlib import Path
import cv2
from torchstain import normalizers as torchstain_normalizers
import histaug
import torch
from torch import nn
from loguru import logger

from .macenko import TorchMacenkoNormalizer, FullyTransparentException


class Augmentation(ABC):
    """Abstract class for augmentation."""

    @abstractmethod
    def __call__(self, image):
        """Apply augmentation to image."""
        pass

    @abstractmethod
    def __repr__(self):
        """Return a string representation of augmentation."""
        pass


class Macenko(Augmentation, nn.Module):
    def __init__(
        self,
        target_image: Path = Path(histaug.__file__).parent.parent / "normalization_template.jpg",
    ):
        super().__init__()
        target = cv2.imread(str(target_image))
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        target = torch.from_numpy(target).permute(2, 0, 1)
        # self.normalizer = torchstain_normalizers.MacenkoNormalizer(backend="torch")
        self.normalizer = TorchMacenkoNormalizer()
        self.normalizer.fit(target)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        try:
            Inorm, H, E = self.normalizer.normalize((image * 255).type(torch.uint8))
            return Inorm.type_as(image) / 255
        except FullyTransparentException:
            logger.warning("Attempting to Macenko normalize fully transparent image. Returning original image.")
            return image

    def __repr__(self):
        return "Macenko"
