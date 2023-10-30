from torchvision import transforms as T
from datasets import load_dataset
from torch.utils.data import Dataset
import numpy as np


class Kather100k(Dataset):
    def __init__(self, cache_dir: str = None, split: str = "train_nonorm"):
        super().__init__()
        self.transform = T.ToTensor()  # we will normalize later (in the feature extractor's forward() method)
        self.inverse_transform = T.ToPILImage()
        self.ds = load_dataset("DykeF/NCTCRCHE100K", split=split, cache_dir=cache_dir).sort("file")
        self.classes = sorted(["LYM", "MUS", "TUM", "STR", "DEB", "BACK", "ADI", "NORM", "MUC"])

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        item = self.ds[index]
        img = self.transform(item["image"])
        label = np.array(self.classes.index(item["label"]))
        filename = item["file"]
        return img, label, filename
