import numpy as np
import scipy.special


def ses(x, a, b, c):
    if len(x.shape) <= 1:
        t = x ** 2
    else:
        t = (x ** 2).sum(axis=0)
    return np.exp(c) * np.exp((-a) * t) / np.maximum(b + t, 1e-10)


def constraint(a, b):
    b = 1 / np.maximum(b, 1e-5)
    return np.maximum(a, 0) ** b / scipy.special.gamma(1 + b) / 2


def ggd(x, a, b):
    # c = a ** (1 / b) / scipy.special.gamma(1 + 1 / b) / 2
    b = np.maximum(b, 0)
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
