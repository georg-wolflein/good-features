from torchvision.models import resnet50, ResNet50_Weights
from torch import nn

from .ctranspath import CTransPath
from .retccl import RetCCL

__all__ = ["CTransPath", "RetCCL", "ResNet50", "load_feature_extractor"]


class ResNet50(nn.Module):
    """ResNet50 feature extractor."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        self.model.fc = nn.Identity()

    def forward(self, x):
        return self.model(x)


def load_feature_extractor(model_name: str, **kwargs) -> nn.Module:
    try:
        return {
            "resnet50": ResNet50,
            "ctranspath": CTransPath,
            "retccl": RetCCL,
        }[
            model_name
        ](**kwargs)
    except KeyError:
        raise ValueError(f"Unknown feature extractor model {model_name}.")
