import numpy as np
import scipy


def possion(laplacian, mean=None):
    # H, W, C
    f = scipy.fft.dct(scipy.fft.dct(laplacian, axis=0, norm='ortho'), axis=1, norm='ortho')
    shape = laplacian.shape[0:2]
    kx, ky = np.meshgrid(
        np.arange(shape[1]),
        np.arange(shape[0]),
    )
    c = 2 * (np.cos(np.pi * ky / (shape[0] + 2)) + np.cos(np.pi * kx / (shape[-1] + 2)) - 2)
    u = f / np.expand_dims(c + (c == 0), axis=-1)
    reconst = scipy.fft.idct(scipy.fft.idct(u, axis=1, norm='ortho'), axis=0, norm='ortho')
    if mean is None:
        return reconst
    else:
        normal = reconst - reconst.mean(axis=(0, 1))
        return normal + mean
