from torch import nn
import torch
from typing import Dict, Tuple


from ..augmentations import Augmentations


class AugmentedFeatureExtractor(nn.Module):
    def __init__(self, feature_extractor: nn.Module, augmentations: Augmentations):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.augmentations = augmentations

    def forward(self, patches, norm_patches=None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        feats = self.feature_extractor(patches)

        augs = {aug_name: aug(patches) for aug_name, aug in self.augmentations.items()}
        feats_augs = {aug_name: self.feature_extractor(aug) for aug_name, aug in augs.items()}

        feats_norm = self.feature_extractor(norm_patches) if norm_patches is not None else None

        return feats, feats_augs, feats_norm
