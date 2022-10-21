import numpy as np


class Uniform:
    def __init__(self, param):
        super(Uniform, self).__init__()
        if isinstance(param, (tuple, list)):
            assert len(param) == 2
            self._min = np.minimum(*param)
            self._max = np.maximum(*param)
        else:
            self._min = np.minimum(-param, param)
            self._max = np.maximum(-param, param)

    def __call__(self):
        return np.random.uniform(self._min, self._max)
