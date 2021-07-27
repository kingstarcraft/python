import numpy as np


class _Core:
    def __init__(self, core):
        self._core = core

    def __call__(self, inputs):
        outputs = inputs
        if isinstance(self._core, list) or isinstance(self._core, tuple):
            if len(self._core) == 0:
                return np.copy(outputs)
            for core in self._core:
                outputs = core(outputs)
            return outputs
        else:
            return self._core(outputs)


class _Dense:
    def __init__(self, weight):
        super(_Dense, self).__init__()
        self.weight = weight

    def __call__(self, inputs):
        return inputs @ self.weight


class RGB2LMS(_Dense):
    def __init__(self):
        super(RGB2LMS, self).__init__([
            [0.3811, 0.1967, 0.0241],
            [0.5783, 0.7244, 0.1288],
            [0.0402, 0.0782, 0.8444]
        ])


class LMS2RGB(_Dense):
    def __init__(self):
        super(LMS2RGB, self).__init__(np.linalg.inv([
            [0.3811, 0.1967, 0.0241],
            [0.5783, 0.7244, 0.1288],
            [0.0402, 0.0782, 0.8444]
        ]))


class BGR2LMS(_Dense):
    def __init__(self):
        super(BGR2LMS, self).__init__([
            [0.0241, 0.1967, 0.3811],
            [0.1288, 0.7244, 0.5783],
            [0.8444, 0.0782, 0.0402]
        ])


class LMS2BGR(_Dense):
    def __init__(self):
        super(LMS2BGR, self).__init__(np.linalg.inv([
            [0.0241, 0.1967, 0.3811],
            [0.1288, 0.7244, 0.5783],
            [0.8444, 0.0782, 0.0402]
        ]))


class LMS2LAB(_Dense):
    def __init__(self):
        super(LMS2LAB, self).__init__(np.dot(
            np.array([[1 / (3 ** 0.5), 0, 0],
                      [0, 1 / (6 ** 0.5), 0],
                      [0, 0, 1 / (2 ** 0.5)]]),
            np.array([[1, 1, 1],
                      [1, 1, -2],
                      [1, -1, 0]])
        ).T)

    def __call__(self, inputs):
        output = np.copy(inputs)
        output[inputs <= 0] = np.spacing(1)
        return super(LMS2LAB, self).__call__(np.log(output))


class LAB2LMS(_Dense):
    def __init__(self):
        super(LAB2LMS, self).__init__(np.linalg.inv(np.dot(
            np.array([[1 / (3 ** 0.5), 0, 0],
                      [0, 1 / (6 ** 0.5), 0],
                      [0, 0, 1 / (2 ** 0.5)]]),
            np.array([[1, 1, 1],
                      [1, 1, -2],
                      [1, -1, 0]])
        ).T))

    def __call__(self, inputs):
        output = super(LAB2LMS, self).__call__(inputs)
        output = np.exp(output.clip(max=255))
        output[output == np.spacing(1)] = 0
        return output


class RGB2LAB(_Core):
    def __init__(self):
        super(RGB2LAB, self).__init__((RGB2LMS(), LMS2LAB()))


class BGR2LAB(_Core):
    def __init__(self):
        super(BGR2LAB, self).__init__((BGR2LMS(), LMS2LAB()))


class LAB2RGB(_Core):
    def __init__(self):
        super(LAB2RGB, self).__init__((LAB2LMS(), LMS2RGB()))


class LAB2BGR(_Core):
    def __init__(self):
        super(LAB2BGR, self).__init__((LAB2LMS(), LMS2BGR()))


class COLOR2OD(object):
    def __init__(self):
        pass

    def __call__(self, inputs):
        outputs = inputs.clip(1)
        return -1 * np.log(outputs / 255)


class OD2COLOR(object):
    def __init__(self):
        pass

    def __call__(self, inputs):
        return (255 * np.exp(-1 * inputs)).astype(np.uint8)
