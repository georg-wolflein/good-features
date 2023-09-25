from torchvision.datasets import ImageFolder
from pathlib import Path

from ..utils.images import transform_with_norm, inverse_transform_with_norm


class Kather100k(ImageFolder):
    def __init__(self, root, mean: tuple, std: tuple):
        self.transform = transform_with_norm(mean, std)
        self.inverse_transform = inverse_transform_with_norm(mean, std)
        super().__init__(root=root, transform=self.transform)

    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        filename, _ = self.imgs[index]
        return image, label, Path(filename).stem
