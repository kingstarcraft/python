import numpy as np
import scipy.optimize

from . import metric, gaussian


def solve(start, end, alpha, probabilty=gaussian.ggd, method=metric.hellinger):
    def core(start, end, alpha):
        d = metric.distance(start, end, probabilty, method)

        def loss(param):
            s = metric.distance(start, param, probabilty=probabilty, method=method)
            e = metric.distance(end, param, probabilty=probabilty, method=method)
            return (s - alpha * d) ** 2 + (s + e - d) ** 2

        return scipy.optimize.minimize(
            loss, start * (1 - alpha) + alpha * end,
            bounds=[[
                np.minimum(start[i], end[i]) - 0.01, np.maximum(start[i], end[i]) + 0.01
            ] for i in range(len(start))]
        )

    assert start.shape == end.shape

    shape = start.shape
    start = start.reshape([-1, shape[-1]])
    end = end.reshape([-1, shape[-1]])
    if isinstance(alpha, (int, float)):
        alpha = [alpha for _ in range(len(start))]
    elif len(shape) >= 3:
        alpha = np.repeat(np.expand_dims(alpha, axis=-1), shape[-2], axis=-1).reshape([-1])
    root = []
    for i in range(len(start)):
        root.append(core(start[i], end[i], alpha[i]).x)
    root = np.stack(root, 0)
    return root.reshape(shape)
