import torch
from . import conversion
from zero.torch.base import Net


class _Net(Net):
    @torch.no_grad()
    def __call__(self, *args, **kwargs):
        return self._net(*args, **kwargs)


class _Distribution(torch.nn.Module):
    def __init__(self, mean, std):
        super(_Distribution, self).__init__()
        self._mean = None if mean is None else torch.Tensor(mean)
        self._std = None if std is None else torch.Tensor(std)


class Normalize(_Distribution):
    def __init__(self, mean, std):
        super(Normalize, self).__init__(mean, std)

    @torch.no_grad()
    def __call__(self, inputs):
        data = torch.reshape(inputs, [-1, inputs.shape[-1]])
        mean = torch.mean(data, dim=0) if self._mean is None else self._mean
        std = torch.std(data, dim=0) if self._std is None else self._std
        return (inputs - mean) / std


class Denormalize(_Distribution):
    def __init__(self, mean, std):
        super(Denormalize, self).__init__(mean, std)

    @torch.no_grad()
    def __call__(self, inputs):
        return inputs * self._std + self._mean


class ReinhardNormalRGB(_Net):
    def __init__(self, src, dst):
        src = (None, None) if src is None else src
        super(ReinhardNormalRGB, self).__init__(
            conversion.RGB2LAB(),
            Normalize(src[0], src[1]),
            Denormalize(dst[0], dst[1]),
            conversion.LAB2RGB()
        )


class ReinhardNormalBGR(_Net):
    def __init__(self, src, dst):
        src = (None, None) if src is None else src
        super(ReinhardNormalBGR, self).__init__(
            conversion.BGR2LAB(),
            Normalize(src[0], src[1]),
            Denormalize(dst[0], dst[1]),
            conversion.LAB2BGR()
        )
