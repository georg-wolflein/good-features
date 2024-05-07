import pytest
from typing import Type
import numpy as np

from histaug.utils.statistics import RunningStats, NaiveStats, BaseStats


def generate_random_test_cases(num_cases, max_size, min=-10.0, max=1e4) -> np.ndarray:
    np.random.seed(0)
    return [np.random.uniform(-min, max, size=np.random.randint(0, max_size)) for _ in range(num_cases)]


def generate_random_batched_test_cases(num_cases, max_num_batches, max_batch_size, min=-10.0, max=1e4) -> np.ndarray:
    np.random.seed(0)
    return [
        [
            np.random.uniform(min, max, size=np.random.randint(0, max_batch_size))
            for _ in range(np.random.randint(0, max_num_batches))
        ]
        for _ in range(num_cases)
    ]


test_cases = generate_random_test_cases(10, 100)  # 10 test cases, up to 100 elements each
batched_test_cases = generate_random_batched_test_cases(
    10, 10, 100
)  # 10 test cases, up to 10 batches, up to 100 elements each


@pytest.mark.parametrize("values", test_cases)
@pytest.mark.parametrize("implementation", [NaiveStats, RunningStats])
def test_stats_update(values, implementation: Type[BaseStats]):
    impl = implementation()

    for value in values:
        impl.update(value)

    stats = impl.compute()
    real_mean = np.mean(values) if len(values) else float("nan")
    real_std = np.std(values) if len(values) >= 2 else float("nan")

    assert stats.mean == pytest.approx(real_mean, abs=1e-6, nan_ok=True)
    assert stats.std == pytest.approx(real_std, abs=1e-6, nan_ok=True)


@pytest.mark.parametrize("values", batched_test_cases)
@pytest.mark.parametrize("implementation", [NaiveStats, RunningStats])
def test_stats_update_batch(values, implementation: Type[BaseStats]):
    impl = implementation()

    for batch in values:
        impl.update_batch(batch)

    stats = impl.compute()
    num_values = sum(batch.size for batch in values)
    real_mean = np.mean(np.concatenate(values)) if num_values else float("nan")
    real_std = np.std(np.concatenate(values)) if num_values >= 2 else float("nan")

    assert stats.mean == pytest.approx(real_mean, abs=1e-6, nan_ok=True)
    assert stats.std == pytest.approx(real_std, abs=1e-6, nan_ok=True)
