from torchvision.models import resnet50, ResNet50_Weights, swin_t, Swin_T_Weights, vit_b_16, ViT_B_16_Weights
from torch import nn
import torch

from .ctranspath import CTransPath
from .retccl import RetCCL
from .owkin import Owkin

__all__ = [
    "CTransPath",
    "RetCCL",
    "SwinTransformer",
    "ResNet50",
    "Owkin",
    "load_feature_extractor",
    "FEATURE_EXTRACTORS",
]


class ResNet50(nn.Module):
    """ResNet50 feature extractor."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        self.model.fc = nn.Identity()

    def forward(self, x):
        return self.model(x)


class SwinTransformer(nn.Module):
    """SwinTransformer feature extractor."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.model = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1 if pretrained else None)
        self.model.head = nn.Identity()

    def forward(self, x):
        return self.model(x)


class ViT(nn.Module):
    """ViT-B feature extractor."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None)

    def forward(self, x):
        # https://stackoverflow.com/a/75875049
        feats = self.model._process_input(x)

        # Expand the class token to the full batch
        batch_class_token = self.model.class_token.expand(x.shape[0], -1, -1)
        feats = torch.cat([batch_class_token, feats], dim=1)

        feats = self.model.encoder(feats)

        # We're only interested in the representation of the classifier token that we appended at position 0
        feats = feats[:, 0]

        return feats


FEATURE_EXTRACTORS = {
    "ctranspath": CTransPath,
    "retccl": RetCCL,
    "resnet50": ResNet50,
    "swin": SwinTransformer,
    "owkin": Owkin,
    "vit": ViT,
}


def load_feature_extractor(model_name: str, **kwargs) -> nn.Module:
    try:
        return FEATURE_EXTRACTORS[model_name](**kwargs)
    except KeyError:
        raise ValueError(f"Unknown feature extractor model {model_name}.")
