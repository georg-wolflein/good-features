from torch import nn
from typing import Dict, Union, Callable
import pandas as pd
import torch
import numpy as np

__all__ = ["SIMILARITY_METRICS", "Metric", "compute_dists"]

Metric = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
TensorLike = Union[torch.Tensor, np.ndarray]


def mahalanobis_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Compute the Mahalanobis distance between two sets of features.

    NOTE: this only works if you pass all the features at once, not one by one, because it computes the covariance matrix.
    """
    # Calculate the covariance matrix and its inverse
    combined = torch.cat([X, Y], dim=0)
    cov_matrix = torch.cov(combined)
    inv_cov_matrix = torch.pinverse(cov_matrix)

    # Compute difference between features and augmented features
    diff = X - Y

    # Compute Mahalanobis distance
    temp = diff @ inv_cov_matrix
    distances = torch.sqrt(torch.einsum("ij,ij->i", temp, diff))

    return distances


SIMILARITY_METRICS: Dict[str, Metric] = {
    "cosine": nn.CosineSimilarity(dim=-1),
    "manhattan": nn.PairwiseDistance(p=1),
    "euclidean": nn.PairwiseDistance(p=2),
    "mahalanobis": mahalanobis_distance,
}


def ensure_tensor(x: TensorLike) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    return torch.from_numpy(x)


def compute_dists(feats: TensorLike, feats_augs: Dict[str, TensorLike], metric: Union[str, Metric]) -> pd.DataFrame:
    feats = ensure_tensor(feats)
    feats_augs = {k: ensure_tensor(v) for k, v in feats_augs.items()}
    if isinstance(metric, str):
        metric = SIMILARITY_METRICS[metric]
    dists = {aug_name: metric(feats, feats_aug) for aug_name, feats_aug in feats_augs.items()}
    dists = pd.DataFrame(dists)

    return dists
