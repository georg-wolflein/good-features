from torch import nn

__all__ = ["SIMILARITY_METRICS"]

SIMILARITY_METRICS = {
    "cosine": nn.CosineSimilarity(dim=-1),
    "euclidean": nn.PairwiseDistance(p=2),
}
