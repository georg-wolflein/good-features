from pathlib import Path
from typing import Union, Optional
from loguru import logger
import hashlib
from torchvision import transforms as T

# Mean and standard deviation used for ImageNet normalization.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class UnNormalize(T.Normalize):
    def __init__(self, mean, std, *args, **kwargs):
        new_mean = [-m / s for m, s in zip(mean, std)]
        new_std = [1 / s for s in std]
        super().__init__(new_mean, new_std, *args, **kwargs)


def download_file(weights_path: Union[Path, str], url: str, checksum: Optional[str] = None):
    weights_path = Path(weights_path)
    if not weights_path.exists():
        logger.info(f"Downloading {url} to {weights_path}")
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        import gdown

        gdown.download(
            url,
            str(weights_path),
            quiet=False,
        )
    else:
        logger.info(f"Skipping download of {url} to {weights_path} as file already exists")

    sha256 = hashlib.sha256()
    with weights_path.open("rb") as f:
        while True:
            data = f.read(1 << 16)
            if not data:
                break
            sha256.update(data)

    assert sha256.hexdigest() == checksum
