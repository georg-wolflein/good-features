from torch import nn
import torch
from pathlib import Path
from typing import Union
import timm
from huggingface_hub import login, hf_hub_download


import histaug


class UNI(nn.Module):
    def __init__(
        self,
        weights_path: Union[Path, str] = Path(histaug.__file__).parent.parent / "weights" / "uni",
    ):
        super().__init__()

        if not weights_path.exists() or not (weights_path / "pytorch_model.bin").exists():
            login()  # login with your User Access Token, found at https://huggingface.co/settings/tokens
            weights_path.mkdir(parents=True)
            hf_hub_download("MahmoodLab/UNI", filename="pytorch_model.bin", local_dir=weights_path, force_download=True)
        self.model = timm.create_model(
            "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
        )
        self.model.load_state_dict(torch.load(weights_path / "pytorch_model.bin", map_location="cpu"), strict=True)
        self.model.eval()

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
