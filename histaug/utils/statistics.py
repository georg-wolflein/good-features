from typing import NamedTuple
import math


class Statistic(NamedTuple):
    mean: float
    std: float


class RunningStats:
    """Welford's algorithm for computing variance incrementally.

    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm
    """

    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, new_value: float):
        self.count += 1
        delta = new_value - self.mean
        self.mean += delta / self.count
        delta2 = new_value - self.mean
        self.M2 += delta * delta2

    def compute(self):
        # Retrieve the mean, variance and sample variance from an aggregate
        if self.count == 0:
            return Statistic(float("nan"), float("nan"))
        mean = self.mean
        variance = self.M2 / self.count
        # sample_variance = self.M2 / (self.count - 1)
        return Statistic(mean, math.sqrt(variance))
