import numpy as np
import scipy.special


class Uniform:
    def __init__(self, param):
        super(Uniform, self).__init__()
        if isinstance(param, (tuple, list)):
            assert len(param) == 2
            self._min = np.minimum(*param)
            self._max = np.maximum(*param)
        else:
            self._min = np.minimum(-param, param)
            self._max = np.maximum(-param, param)

    def __call__(self):
        return np.random.uniform(self._min, self._max)


def ses(x, a, b, c):
    if len(x.shape) <= 1:
        t = x ** 2
    else:
        t = (x ** 2).sum(axis=0)
    return np.exp(c) * np.exp((-a) * t) / np.maximum(b + t, 1e-10)


def constraint(a, b):
    b = 1 / np.maximum(b, 1e-10)
    return np.maximum(a, 0) ** b / scipy.special.gamma(1 + b) / 2


def ggd(x, a, b):
    # c = a ** (1 / b) / scipy.special.gamma(1 + 1 / b) / 2
    c = constraint(a, b)
    if len(x.shape) <= 1:
        return c * np.exp(-a * np.abs(x) ** b)
    else:
        return c * np.exp(-a * (np.abs(x) ** b).sum(axis=0))


def normal(x, a):
    return ggd(x, a, 2)


def hyper_laplacion(x, a):
    return ggd(x, a, 2 / 3)


def laplacion(x, a):
    return ggd(x, a, 1)
