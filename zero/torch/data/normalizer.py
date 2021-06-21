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


# class ReinhardNormalRGB(_Net):
#     def __init__(self, src, dst):
#         src = (None, None) if src is None else src
#         super(ReinhardNormalRGB, self).__init__(
#             conversion.RGB2LAB(),
#             Normalize(src[0], src[1]),
#             Denormalize(dst[0], dst[1]),
#             conversion.LAB2RGB()
#         )
#
#
# class ReinhardNormalBGR(_Net):
#     def __init__(self, src, dst):
#         src = (None, None) if src is None else src
#         super(ReinhardNormalBGR, self).__init__(
#             conversion.BGR2LAB(),
#             Normalize(src[0], src[1]),
#             Denormalize(dst[0], dst[1]),
#             conversion.LAB2BGR()
#         )

class ReinhardNormal(torch.nn.Module):
    def __init__(self, from_color, to_color, target=None):
        super(ReinhardNormal, self).__init__()
        self._from_color = from_color
        self._to_color = to_color
        self._target = target

    def __call__(self, inputs, dst=None, src=None, offset=None):
        outputs = self._from_color(inputs)
        if src is None:
            shape = outputs.shape
            temps = torch.reshape(outputs, [-1, shape[-1]])
            src = torch.mean(temps, dim=0), torch.std(temps, dim=0)
        if offset is not None:
            src = src[0] + offset[0], src[1] + offset[1]
        if dst is None:
            dst = self._target
        outputs = (outputs - src[0]) / src[1]
        outputs = outputs * dst[1] + dst[0]
        return self._to_color(outputs)


class ReinhardNormalBGR(ReinhardNormal):
    def __init__(self, target=None):
        super(ReinhardNormalBGR, self).__init__(conversion.BGR2LAB(), conversion.LAB2BGR(), target)


class ReinhardNormalRGB(ReinhardNormal):
    def __init__(self, target=None):
        super(ReinhardNormalRGB, self).__init__(conversion.RGB2LAB(), conversion.LAB2RGB(), target)
