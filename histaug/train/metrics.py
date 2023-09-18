from typing import Type
from torchmetrics import Metric, MetricCollection
from torch import Tensor
from torchmetrics.classification import (
    BinaryAUROC,
    BinaryAveragePrecision,
    MulticlassAUROC,
    MulticlassAccuracy,
)
from torchmetrics.regression import (
    MeanAbsoluteError,
    MeanSquaredError,
    R2Score,
    ExplainedVariance,
)
from functools import partial
from collections import ChainMap

__all__ = [
    "ClasswiseMulticlassAUROC",
    "ClasswiseMulticlassAveragePrecision",
    "create_metrics_for_target",
    "MULTICLASS_METRICS",
    "CLASSWISE_METRICS",
    "REGRESSION_METRICS",
    "METRIC_GOALS",
]


def _make_classwise_metric(metric_class: Type[Metric]) -> Type[Metric]:
    class ClasswiseMetric(metric_class):
        def __init__(self, *args, class_id, **kwargs):
            super().__init__(*args, **kwargs)
            self.class_id = class_id

        def update(self, preds: Tensor, target: Tensor):
            super().update(preds[..., self.class_id], target == self.class_id)

    return ClasswiseMetric


ClasswiseMulticlassAUROC = _make_classwise_metric(BinaryAUROC)
ClasswiseMulticlassAveragePrecision = _make_classwise_metric(BinaryAveragePrecision)

MAX, MIN = "max", "min"

MULTICLASS_METRICS = {
    "auroc": (MulticlassAUROC, MAX),
    "acc": (partial(MulticlassAccuracy, average="macro"), MAX),
    "bal_acc": (partial(MulticlassAccuracy, average="weighted"), MAX),
    # "ap": MulticlassAveragePrecision,
}
CLASSWISE_METRICS = {
    "auroc": (ClasswiseMulticlassAUROC, MAX),
    # "ap": ClasswiseMulticlassAveragePrecision,
}
REGRESSION_METRICS = {
    "mae": (MeanAbsoluteError, MIN),
    "mse": (MeanSquaredError, MIN),
    "r2": (R2Score, MAX),
    "ev": (ExplainedVariance, MAX),
}
METRIC_GOALS = ChainMap(
    *(
        {name: goal for name, (metric, goal) in metrics.items()}
        for metrics in (MULTICLASS_METRICS, CLASSWISE_METRICS, REGRESSION_METRICS)
    )
)


def create_metrics_for_target(target) -> MetricCollection:
    if target.type == "categorical":
        metrics = {
            **{
                name: (metric(num_classes=len(target.classes)), goal)
                for name, (metric, goal) in MULTICLASS_METRICS.items()
            },
            **{
                f"{name}_{c}": (metric(class_id=i), goal)
                for i, c in enumerate(target.classes)
                for name, (metric, goal) in CLASSWISE_METRICS.items()
            },
        }

    elif target.type == "continuous":
        metrics = {name: (metric(), goal) for name, (metric, goal) in REGRESSION_METRICS.items()}
    else:
        raise NotImplementedError(f"Unknown target type {target.type}")
    goals = {name: goal for name, (metric, goal) in metrics.items()}
    return (
        MetricCollection({name: metric for name, (metric, goal) in metrics.items()}),
        goals,
    )
