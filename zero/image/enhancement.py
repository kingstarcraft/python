import numpy as np


def optical_density(image, I0=256):
    return 255 + np.log((image + 1.0) / I0) * 255.0 / np.log(I0)
