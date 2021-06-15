import torch
from . import conversion
from zero.torch.base import Net


class _Distribution(torch.nn.Module):
    def __init__(self, mean, std):
        super(_Distribution, self).__init__()
        self._mean = None if mean is None else torch.Tensor(mean)
        self._std = None if std is None else torch.Tensor(std)


class Normalize(_Distribution):
    def __init__(self, mean, std):
        super(Normalize, self).__init__(mean, std)

    def __call__(self, inputs):
        data = torch.reshape(inputs, [-1, inputs.shape[-1]])
        mean = torch.mean(data, dim=0) if self._mean is None else self._mean
        std = torch.std(data, dim=0) if self._std is None else self._std
        return (inputs - mean) / std


class Denormalize(_Distribution):
    def __init__(self, mean, std):
        super(Denormalize, self).__init__(mean, std)

    def __call__(self, inputs):
        return inputs * self._std + self._mean


class ReinhardNormalRGB(Net):
    def __init__(self, src, dst):
        super(ReinhardNormalRGB, self).__init__()
        src = (None, None) if src is None else src
        self._net = torch.nn.Sequential(
            conversion.RGB2LAB(),
            Normalize(src[0], src[1]),
            Denormalize(dst[0], dst[1]),
            conversion.LAB2RGB()
        )


class ReinhardNormalBGR(Net):
    def __init__(self, src, dst):
        super(ReinhardNormalBGR, self).__init__()
        src = (None, None) if src is None else src
        self._net = torch.nn.Sequential(
            conversion.BGR2LAB(),
            Normalize(src[0], src[1]),
            Denormalize(dst[0], dst[1]),
            conversion.LAB2BGR()
        )
