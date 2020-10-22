import numpy as np


def grid(shape, dtype='float32', normal: bool = True):
    outputs = []
    axises = list(range(len(shape)))
    for i, s in enumerate(shape):
        axis = tuple(filter(lambda axis: axis != i, axises))
        if normal:
            data = np.expand_dims(2 * np.array(range(s)) / (s - 1) - 1, axis=axis).astype(dtype)
        else:
            data = np.expand_dims(np.array(range(s)), axis=axis).astype(dtype)
        for j in axis:
            data = data.repeat(shape[j], axis=j)
        outputs.append(data)
    return np.stack(outputs, axis=-1)


if __name__ == '__main__':
    shape = grid((512, 512))
