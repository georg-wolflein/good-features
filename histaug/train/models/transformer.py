from torch import nn
from omegaconf import ListConfig
from typing import Optional
import torch.nn.functional as F


class Transformer(nn.Module):
    def __init__(
        self,
        targets: ListConfig,
        d_features: int,
        hidden_dim: int,
        dropout: float = 0.1,
        num_heads: int = 8,
        num_layers: int = 2,
        feedforward_dim: Optional[int] = None,
    ):
        super().__init__()
        feedforward_dim = feedforward_dim or hidden_dim
        self.encoder = nn.Sequential(nn.Linear(d_features, hidden_dim), nn.ReLU())
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=feedforward_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=num_layers,
            mask_check=False,
        )
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
        self.dropout = nn.Dropout(dropout)

    def forward(self, feats, coords, mask, *args, **kwargs):
        embeddings = self.encoder(feats)  # B, N, D
        embeddings = self.dropout(embeddings)
        embeddings = self.transformer(embeddings)  # B, N, D
        slide_tokens = embeddings.mean(dim=-2)  # B, D

        # Apply the corresponding head to each slide-level token
        logits = {target.column: self.heads[target.column](slide_tokens).squeeze(-1) for target in self.targets}
        return logits
