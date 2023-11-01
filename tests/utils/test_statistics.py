from histaug.utils.statistics import RunningStats
import math


def test_initialization():
    stats = RunningStats()
    assert stats.count == 0
    assert stats.mean == 0.0
    assert stats.M2 == 0.0


def test_update_method():
    stats = RunningStats()

    stats.update(5)
    assert stats.count == 1
    assert stats.mean == 5.0
    assert stats.M2 == 0.0

    stats.update(7)
    assert stats.count == 2
    assert stats.mean == 6.0
    # Hand calculated M2 value after two updates
    assert stats.M2 == 2.0


def test_compute_method():
    stats = RunningStats()

    # Since count is 0, mean and variance should be NaN
    mean, std = stats.compute()
    assert math.isnan(mean)
    assert math.isnan(std)

    stats.update(5)
    stats.update(7)

    mean, std = stats.compute()
    assert mean == 6.0
    assert std == 1.0  # (5-6)^2 + (7-6)^2 = 1 + 1 = 2, divided by 2, sqrt


def test_update_with_multiple_data_points():
    stats = RunningStats()

    data = [2, 4, 4, 4, 5, 5, 7, 9]
    expected_mean = sum(data) / len(data)  # which is 5.0

    # Calculating the variance manually for this dataset:
    expected_std = math.sqrt(sum((x - expected_mean) ** 2 for x in data) / len(data))  # which is 4.0

    for number in data:
        stats.update(number)

    mean, std = stats.compute()
    assert mean == expected_mean
    assert std == expected_std
