from torchvision.datasets import ImageFolder
from torchvision import transforms as T


class Kather100k(ImageFolder):
    def __init__(self, root):
        self.transform = T.ToTensor()
        self.inverse_transform = T.ToPILImage()
        super().__init__(root=root, transform=self.transform)