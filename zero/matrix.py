import numpy as np


def stack_dim(inputs):
    outputs = []
    axises = list(range(len(inputs)))
    for i, dim in enumerate(inputs):
        axis = tuple(filter(lambda axis: axis != i, axises))
        data = np.expand_dims(np.array(dim), axis=axis)
        for j in axis:
            data = data.repeat(len(inputs[j]), axis=j)
        outputs.append(data)
    return np.stack(outputs, axis=-1)


def split(size, crop, overlap=0):
    if crop is None:
        return [(0, 0, *size)]
    crop = np.minimum(size, crop)
    overlap = 0 if overlap is None else (np.zeros_like(crop) + overlap)
    assert len(crop) == len(size) == len(overlap)
    interval = crop - overlap
    assert not (interval <= 0).any()
    numbel = ((size - crop + interval - 1) / interval).astype('int32') + 1
    starts, ends = [], []
    for s, c, i, n in zip(size, crop, interval, numbel):
        starts.append([min(_ * i, s - c) for _ in range(n)])
        ends.append([min(_ * i, s - c) + c for _ in range(n)])
    starts = stack_dim(starts)
    ends = stack_dim(ends)
    return np.concatenate((starts, ends), axis=-1)


def crop(image, crop=1024, overlap=0, start=-3, end=-1, keep=True):
    dim = abs(start) if end is None else end - start
    rois = np.reshape(split(image.shape[start:end], crop, overlap), (-1, 2 * dim))
    pathes = []
    for roi in rois:
        slices = [slice(None) for _ in image.shape]
        for n in range(dim):
            slices[start + n] = slice(roi[n], roi[n + dim])
        if keep:
            pathes.append((tuple(roi[0:dim]), image[tuple(slices)]))
        else:
            pathes.append(image[tuple(slices)])
    return pathes
