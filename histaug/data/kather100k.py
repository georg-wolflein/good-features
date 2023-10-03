from torchvision import transforms as T
from datasets import load_dataset
from torch.utils.data import Dataset


class Kather100k(Dataset):
    def __init__(self, cache_dir: str = None, split: str = "train_nonorm"):
        super().__init__()
        self.transform = T.ToTensor()  # we will normalize later (in the feature extractor's forward() method)
        self.inverse_transform = T.ToPILImage()
        self.ds = load_dataset("DykeF/NCTCRCHE100K", split=split, cache_dir=cache_dir).sort("file")

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        item = self.ds[index]
        img = self.transform(item["image"])
        return img, item["label"], item["file"]
