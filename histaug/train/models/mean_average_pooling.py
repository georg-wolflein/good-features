from torch import nn
from omegaconf import ListConfig
from typing import Optional


class MeanAveragePooling(nn.Module):
    def __init__(self, targets: ListConfig, d_features: int, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden_dim = hidden_dim or d_features
        self.mlp = nn.Sequential(nn.Linear(d_features, hidden_dim), nn.ReLU())
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
        slide_tokens = feats.mean(dim=-2)
        slide_tokens = self.mlp(slide_tokens)

        # Apply the corresponding head to each slide-level token
        logits = {target.column: self.heads[target.column](slide_tokens).squeeze(-1) for target in self.targets}
        return logits
