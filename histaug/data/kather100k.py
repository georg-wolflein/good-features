from torchvision.datasets import ImageFolder
from torchvision import transforms as T
from pathlib import Path


class Kather100k(ImageFolder):
    transform = T.ToTensor()
    inverse_transform = T.ToPILImage()

    def __init__(self, root):
        super().__init__(root=root, transform=self.transform)

    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        filename, _ = self.imgs[index]
        return image, label, Path(filename).stem
