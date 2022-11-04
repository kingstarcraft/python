import numpy as np
from zero import stats
from . import solver


def map(inputs, source, target):
    def core(tensor, source, target, const):
        sign = np.sign(tensor)
        sa, sb = source[..., 0], source[..., 1]
        ta, tb = target[..., 0], target[..., 1]
        outputs = sa * (np.abs(tensor) ** sb) + const
        outputs = (outputs.clip(0) / ta) ** (1 / tb)
        return sign * outputs

    const = np.log(
        stats.gaussian.constraint(target[..., 0], target[..., 1]) /
        stats.gaussian.constraint(source[..., 0], source[..., 1])
    )
    if isinstance(inputs, (list, tuple)):
        return type(inputs)([core(_, source, target, const) for _ in inputs])
    return core(inputs, source, target, const)


def _pad_x(image):
    h, w = image.shape[0:2]
    pad = np.zeros([h, w + 2] + list(image.shape[2:]), dtype=image.dtype)
    pad[:, 1:-1, ...] = image
    pad[:, 0, ...] = pad[:, 1, ...]
    pad[:, w + 1, ...] = pad[:, w, ...]
    return pad


def _pad_y(image):
    h, w = image.shape[0:2]
    pad = np.zeros([h + 2, w] + list(image.shape[2:]), dtype=image.dtype)
    pad[1:-1, :, ...] = image
    pad[0, :, ...] = pad[1, :, ...]
    pad[h + 1, :, ...] = pad[h, :, ...]
    return pad


def transform(image, source, target, const=None):
    if const is None:
        const = image.mean(axis=(0, 1))

    ty = np.diff(_pad_y(image).astype('int32'), axis=0)
    tx = np.diff(_pad_x(image).astype('int32'), axis=1)

    if source.shape[-2] == 2:
        source_y, source_x = source[..., 0, :], source[..., 1, :]
        target_y, target_x = target[..., 0, :], target[..., 1, :]
    elif source.shape[-2] == 1:
        source_y, source_x = source[..., 0, :], source[..., 0, :]
        target_y, target_x = target[..., 0, :], target[..., 0, :]
    else:
        raise RuntimeError

    my = map(ty, source_y, target_y)
    mx = map(tx, source_x, target_x)

    fx = np.diff(mx, axis=1)
    fy = np.diff(my, axis=0)
    return solver.possion(fx + fy, const)
