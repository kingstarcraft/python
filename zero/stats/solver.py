import numpy as np
from scipy.optimize import fsolve

from . import metric, gaussian


def solve(start, end, alpha, probabilty=gaussian.ggd, method=metric.hellinger):
    def core(start, end, alpha):
        d = metric.distance(start, end, probabilty, method)

        def loss(param):
            s = metric.distance(start, param, probabilty=probabilty, method=method)
            e = metric.distance(end, param, probabilty=probabilty, method=method)
            return s - alpha * d, s + e - d

        return fsolve(loss, start * (1 - alpha) + alpha * end)

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
        root.append(core(start[i], end[i], alpha[i]))
    root = np.stack(root, 0)
    return root.reshape(shape)
