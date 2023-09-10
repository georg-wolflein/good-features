from torchvision.models import resnet50, ResNet50_Weights, swin_t, Swin_T_Weights
from torch import nn

from .ctranspath import CTransPath
from .retccl import RetCCL

__all__ = [
    "CTransPath",
    "RetCCL",
    "SwinTransformer",
    "ResNet50",
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


FEATURE_EXTRACTORS = {
    "ctranspath": CTransPath,
    "retccl": RetCCL,
    "resnet50": ResNet50,
    "swin": SwinTransformer,
}


def load_feature_extractor(model_name: str, **kwargs) -> nn.Module:
    try:
        return FEATURE_EXTRACTORS[model_name](**kwargs)
    except KeyError:
        raise ValueError(f"Unknown feature extractor model {model_name}.")
