import numpy
import torch
from zero.torch.base import Net


class _Dense(torch.nn.Linear):
    def __init__(self, weight):
        super(_Dense, self).__init__(3, 3, False)
        self.weight.data[:] = torch.Tensor(weight)


class RGB2LMS(_Dense):
    def __init__(self):
        super(RGB2LMS, self).__init__([
            [0.3811, 0.5783, 0.0402],
            [0.1967, 0.7244, 0.0782],
            [0.0241, 0.1288, 0.8444]
        ])


class LMS2RGB(_Dense):
    def __init__(self):
        super(LMS2RGB, self).__init__(numpy.linalg.inv([
            [0.3811, 0.5783, 0.0402],
            [0.1967, 0.7244, 0.0782],
            [0.0241, 0.1288, 0.8444]
        ]))


class BGR2LMS(_Dense):
    def __init__(self):
        super(BGR2LMS, self).__init__([
            [0.0402, 0.5783, 0.3811],
            [0.0782, 0.7244, 0.1967],
            [0.8444, 0.1288, 0.0241]
        ])


class LMS2BGR(_Dense):
    def __init__(self):
        super(LMS2BGR, self).__init__(numpy.linalg.inv([
            [0.0402, 0.5783, 0.3811],
            [0.0782, 0.7244, 0.1967],
            [0.8444, 0.1288, 0.0241]
        ]))


class LMS2LAB(_Dense):
    def __init__(self):
        super(LMS2LAB, self).__init__(numpy.dot(
            numpy.array([[1 / (3 ** 0.5), 0, 0],
                         [0, 1 / (6 ** 0.5), 0],
                         [0, 0, 1 / (2 ** 0.5)]]),
            numpy.array([[1, 1, 1],
                         [1, 1, -2],
                         [1, -1, 0]])
        ))

    def __call__(self, input):
        output = torch.clone(input)
        output[input <= 0] = numpy.spacing(1)
        return super(LMS2LAB, self).__call__(torch.log(output))


class LAB2LMS(_Dense):
    def __init__(self):
        super(LAB2LMS, self).__init__(numpy.linalg.inv(numpy.dot(
            numpy.array([[1 / (3 ** 0.5), 0, 0],
                         [0, 1 / (6 ** 0.5), 0],
                         [0, 0, 1 / (2 ** 0.5)]]),
            numpy.array([[1, 1, 1],
                         [1, 1, -2],
                         [1, -1, 0]])
        )))

    def __call__(self, input):
        output = super(LAB2LMS, self).__call__(input)
        output = torch.exp(output)
        output[output == numpy.spacing(1)] = 0
        return output


class RGB2LAB(Net):
    def __init__(self):
        super(RGB2LAB, self).__init__()
        self._net = torch.nn.Sequential(
            RGB2LMS(),
            LMS2LAB()
        )


class BGR2LAB(Net):
    def __init__(self):
        super(BGR2LAB, self).__init__()
        self._net = torch.nn.Sequential(
            BGR2LMS(),
            LMS2LAB()
        )


class LAB2RGB(Net):
    def __init__(self):
        super(LAB2RGB, self).__init__()
        self._net = torch.nn.Sequential(
            LAB2LMS(),
            LMS2RGB()
        )


class LAB2BGR(Net):
    def __init__(self):
        super(LAB2BGR, self).__init__()
        self._net = torch.nn.Sequential(
            LAB2LMS(),
            LMS2BGR()
        )
