import numpy as np
import torch
import zero


class MMDect(torch.nn.Module):
    def __init__(self, net, threshold=1.0, size=None, overlap=None):
        super(MMDect, self).__init__()
        self._net = net
        self._size = size
        self._overlap = overlap
        self._threshold = threshold

        for key in dir(net):
            if not hasattr(self, key):
                if hasattr(net, key) and callable(getattr(net, key)):
                    setattr(self, key, getattr(net, key))

    def __call__(self, img, *args, **kwargs):
        results = []
        for i in img:
            patches = zero.matrix.crop(i, self._size, self._overlap, start=-2, end=None)
            result = None
            format_tuple = False
            for (y, x), patch in patches:
                offset = np.array([x, y, x, y, 0])
                src = self._net(img=[patch], *args, **kwargs)[0]
                dst = [s + offset for s in src]
                if result is None:
                    result = dst
                elif isinstance(src, np.ndarray):
                    result = np.concatenate([result, src])
                else:
                    assert len(result) == len(dst)
                    result = [np.concatenate((result[i], dst[i])) for i in range(len(src))]
                    format_tuple = isinstance(src, tuple)
            if self._size is not None:
                if isinstance(result, np.ndarray) and result.shape[-1] == 5:
                    result = zero.box.filter_box(result, self._threshold)
                else:
                    for i in range(len(result)):
                        if result[i].shape[-1] == 5:
                            result[i] = zero.box.filter_box(result[i])
            result = tuple(result) if format_tuple else result
            results.append(result)
        return results
