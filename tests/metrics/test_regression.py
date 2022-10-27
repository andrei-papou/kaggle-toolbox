import math

import torch

from kaggle_toolbox.metrics.regression import MSEMetric


def test_mse():
    x = torch.tensor([
        [0.1, 0.9, 0.8, 0.2],
        [0.7, 0.9, 0.1, 0.2],
    ])
    y = torch.tensor([
        [0.0, 1.0, 1.0, 0.0],
        [1.0, 1.0, 0.0, 0.0],
    ])

    metric = MSEMetric()
    metric(x, y)

    assert math.isclose(
        metric.compute(),
        (0.1 ** 2 + 0.1 ** 2 + 0.2 ** 2 + 0.2 ** 2 + 0.3 ** 2 + 0.1 ** 2 + 0.1 ** 2 + 0.2 ** 2) / 8,
        abs_tol=1e-8)
