import numpy as np
import torch
import zero


class SlidingWindow(torch.nn.Module):
    def __init__(self, model, threshold=1.0, window=None, overlap=None):
        super(SlidingWindow, self).__init__()
        self._model = model
        self._window = window
        self._overlap = overlap
        self._filter = zero.boxes.Filter(threshold=self._threshold)

        for key in dir(model):
            if not hasattr(self, key):
                if hasattr(model, key) and callable(getattr(model, key)):
                    setattr(self, key, getattr(model, key))

    def __call__(self, img, *args, **kwargs):
        results = []
        for i in img:
            patches = zero.matrix.crop(i, self._window, self._overlap, start=-2, end=None)
            result = None
            format_tuple = False
            for (y, x), patch in patches:
                offset = np.array([x, y, x, y, 0])
                src = self._model(img=[patch], *args, **kwargs)[0]
                dst = [s + offset for s in src]
                if result is None:
                    result = dst
                elif isinstance(src, np.ndarray):
                    result = np.concatenate([result, src])
                else:
                    assert len(result) == len(dst)
                    result = [np.concatenate((result[i], dst[i])) for i in range(len(src))]
                    format_tuple = isinstance(src, tuple)
            if self._window is not None:
                if isinstance(result, np.ndarray) and result.shape[-1] == 5:
                    result = self._filter(result)
                else:
                    for i in range(len(result)):
                        if result[i].shape[-1] == 5:
                            result[i] = self._filter(result[i])
            result = tuple(result) if format_tuple else result
            results.append(result)
        return results
