from torch import nn
from omegaconf import ListConfig
from typing import Optional


class AttentionMIL(nn.Module):
    def __init__(self, targets: ListConfig, d_features: int, hidden_dim: Optional[int] = 256):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(d_features, hidden_dim), nn.ReLU())
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.Tanh(), nn.Linear(hidden_dim // 2, 1), nn.Softmax(dim=-2)
        )
        # self.pre_head = nn.Sequential(nn.BatchNorm1d(hidden_dim), nn.Dropout())
        self.pre_head = nn.Dropout()
        self.heads = nn.ModuleDict(
            {
                target.column: nn.Linear(
                    in_features=hidden_dim,
                    out_features=len(target.classes) if target.type == "categorical" else 1,
                )
                for target in targets
            }
        )
        self.targets = targets

    def forward(self, feats, *args, **kwargs):
        B, N, D = feats.shape
        assert B == 1, "currently only supports batch size 1"
        embeddings = self.encoder(feats)  # B, N, D
        attention = self.attention(embeddings)  # B, N, 1
        embeddings = embeddings * attention  # B, N, D
        slide_tokens = embeddings.sum(dim=-2)  # B, D
        slide_tokens = self.pre_head(slide_tokens)  # B, D

        # Apply the corresponding head to each slide-level token
        logits = {target.column: self.heads[target.column](slide_tokens).squeeze(-1) for target in self.targets}
        return logits
