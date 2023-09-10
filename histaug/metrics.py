from torch import nn
from typing import Dict, Union, Callable
import pandas as pd
import torch

from .extract_features import LoadedFeatures

__all__ = ["SIMILARITY_METRICS", "Metric", "compute_dists"]

Metric = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

SIMILARITY_METRICS: Dict[str, Metric] = {
    "cosine": nn.CosineSimilarity(dim=-1),
    "manhattan": nn.PairwiseDistance(p=1),
    "euclidean": nn.PairwiseDistance(p=2),
}


def compute_dists(features: LoadedFeatures, metric: Union[str, Metric]) -> pd.DataFrame:
    if isinstance(metric, str):
        metric = SIMILARITY_METRICS[metric]
    dists = {
        aug_name: metric(torch.from_numpy(features.feats), torch.from_numpy(feats_aug))
        for aug_name, feats_aug in features.feats_augs.items()
    }
    dists = pd.DataFrame(dists)

    return dists
