import numpy as np
from . import conversion


class ReinhardNormal:
    def __init__(self, from_color, to_color, target=None):
        self._from_color = from_color
        self._to_color = to_color
        self._target = target

    def __call__(self, inputs, dst=None, src=None, offset=None):
        if src is None:
            shape = inputs.shape
            temps = np.reshape([-1, shape[-1]])
            src = np.mean(temps, axis=0), np.std(temps, axis=0)
        else:
            src = np.array(src[0]), np.array(src[1])
        if offset is not None:
            src = src[0] + offset[0], src[1] + offset[1]
        if dst is None:
            dst = self._target
        outputs = self._from_color(inputs)
        outputs = (outputs - src[0]) / src[1]
        outputs = outputs * dst[1] + dst[0]
        return self._to_color(outputs)


class ReinhardNormalBGR(ReinhardNormal):
    def __init__(self):
        super(ReinhardNormalBGR, self).__init__(conversion.BGR2LAB(), conversion.LAB2BGR())


class ReinhardNormalRGB(ReinhardNormal):
    def __init__(self):
        super(ReinhardNormalRGB, self).__init__(conversion.RGB2LAB(), conversion.LAB2RGB())
