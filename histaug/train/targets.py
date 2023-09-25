import abc
import pandas as pd
import torch
from omegaconf import DictConfig
from torch.nn import functional as F

__all__ = ["TargetEncoder", "CategoricalTargetEncoder", "ContinuousTargetEncoder"]


class TargetEncoder(abc.ABC):
    def __init__(self, target_cfg: DictConfig) -> None:
        self.target_cfg = target_cfg

    @abc.abstractmethod
    def fit(self, clini_df: pd.DataFrame) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def __call__(self, clini_df: pd.DataFrame) -> torch.Tensor:
        pass

    @staticmethod
    def for_target(target_cfg: DictConfig):
        if target_cfg.type == "categorical":
            return CategoricalTargetEncoder(target_cfg)
        elif target_cfg.type == "continuous":
            return ContinuousTargetEncoder(target_cfg)
        else:
            raise NotImplementedError(f"target type {target_cfg.type} not implemented")


class CategoricalTargetEncoder(TargetEncoder):
    def __call__(self, clini_df: pd.DataFrame) -> torch.Tensor:
        values = clini_df[self.target_cfg.column].map({c: i for i, c in enumerate(self.target_cfg.classes)})
        return torch.tensor(values.values, dtype=torch.long)

    fit = __call__


class ContinuousTargetEncoder(TargetEncoder):
    def fit(self, clini_df: pd.DataFrame) -> torch.Tensor:
        # Normalize
        values = clini_df[self.target_cfg.column].astype(float).values
        self.mean = values.mean()
        self.std = values.std()
        return self(clini_df)

    def __call__(self, clini_df: pd.DataFrame) -> torch.Tensor:
        values = clini_df[self.target_cfg.column].astype(float).values
        return torch.tensor((values - self.mean) / self.std).float()
