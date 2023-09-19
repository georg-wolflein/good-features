from torchvision.datasets import ImageFolder
from torchvision import transforms as T
from pathlib import Path

from ..feature_extractors.utils import IMAGENET_MEAN, IMAGENET_STD, UnNormalize


class Kather100k(ImageFolder):
    transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ]
    )
    inverse_transform = T.Compose([UnNormalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), T.ToPILImage()])

    def __init__(self, root):
        super().__init__(root=root, transform=self.transform)

    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        filename, _ = self.imgs[index]
        return image, label, Path(filename).stem
