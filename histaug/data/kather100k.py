from torchvision.datasets import ImageFolder
from pathlib import Path
from torchvision import transforms as T


class Kather100k(ImageFolder):
    def __init__(self, root):
        self.transform = T.ToTensor()  # we will normalize later (in the feature extractor's forward() method)
        self.inverse_transform = T.ToPILImage()
        super().__init__(root=root, transform=self.transform)

    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        filename, _ = self.imgs[index]
        return image, label, Path(filename).stem
