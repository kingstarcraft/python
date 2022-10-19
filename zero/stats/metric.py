import numpy as np


def hellinger(p, q):
    return np.linalg.norm(np.sqrt(p) - np.sqrt(q)) / np.sqrt(2)
