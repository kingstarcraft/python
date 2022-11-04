import numpy as np


def hellinger(p, q):
    return np.linalg.norm(np.sqrt(p) - np.sqrt(q)) / np.sqrt(2)


def distance(s, t, probabilty, method):
    x = np.linspace(-255, 255, 511, endpoint=True)
    p = probabilty(x, *s)
    q = probabilty(x, *t)
    return method(p, q)
