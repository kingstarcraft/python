import numpy as np
import scipy.stats


def l1(p, q):
    return np.abs(p - q).mean()


def l2(p, q):
    return np.sqrt(((p - q) ** 2).mean())


def cos(p, q):
    return 1 - (np.sqrt(p) * np.sqrt(q)).sum().clip(0, 1)


def hellinger(p, q):
    return np.linalg.norm(np.sqrt(p) - np.sqrt(q)) / np.sqrt(2)


def wasserstein(p, q):
    return scipy.stats.wasserstein_distance(p, q)


def chisquare(p, q):
    p = np.array(p)
    q = np.array(q)
    q = q + (q == 0) * 1e-10
    p = p + (p == 0) * 1e-10
    return scipy.stats.chisquare(p, q)


def distance(s, t, probabilty, method):
    x = np.linspace(-255, 255, 511, endpoint=True)
    p = probabilty(x, *s)
    q = probabilty(x, *t)
    return method(p, q)
