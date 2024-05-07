from typing import NamedTuple
import math
from abc import ABC, abstractmethod
import numpy as np


class Statistic(NamedTuple):
    mean: float
    std: float


class BaseStats(ABC):
    @abstractmethod
    def update(self, v: float):
        pass

    @abstractmethod
    def update_batch(self, v: np.ndarray):
        pass

    @abstractmethod
    def compute(self) -> Statistic:
        pass


class NaiveStats(BaseStats):
    def __init__(self):
        self.values = []

    def update(self, v: float):
        self.values.append(v)

    def update_batch(self, v: np.ndarray):
        self.values.extend(v)

    def compute(self) -> Statistic:
        n = len(self.values)
        mean = float("nan") if n == 0 else sum(self.values) / n
        std = float("nan") if n < 2 else math.sqrt(sum((x - mean) ** 2 for x in self.values) / n)
        return Statistic(mean=mean, std=std)


class RunningStats:
    """Welford's algorithm for computing variance incrementally.

    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm
    """

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, v: float):
        self.n += 1
        delta = v - self.mean
        self.mean += delta / self.n
        delta2 = v - self.mean
        self.M2 += delta * delta2

    def update_batch(self, v: np.ndarray):
        # Check if the array is empty
        if v.size == 0:
            return

        batch_size = v.size
        batch_mean = np.mean(v)

        # Update the global mean and M2
        old_mean = self.mean
        self.mean = (self.mean * self.n + batch_mean * batch_size) / (self.n + batch_size)

        self.n += batch_size
        self.M2 += (
            np.sum((v - batch_mean) ** 2) + batch_size * (batch_mean - old_mean) ** 2 * (self.n - batch_size) / self.n
        )

    def compute(self) -> Statistic:
        mean = self.mean if self.n > 0 else float("nan")
        std = math.sqrt(self.M2 / self.n) if self.n >= 2 else float("nan")
        return Statistic(mean=mean, std=std)
