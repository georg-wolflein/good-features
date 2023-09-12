from torch import nn
import torch
from typing import Dict, Tuple

from ..augmentations import Augmentations


class AugmentedFeatureExtractor(nn.Module):
    def __init__(self, feature_extractor: nn.Module, augmentations: Augmentations):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.augmentations = augmentations

    def forward(self, images) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        feats = self.feature_extractor(images)
        feats_augs = {aug_name: self.feature_extractor(aug(images)) for aug_name, aug in self.augmentations.items()}
        return feats, feats_augs
