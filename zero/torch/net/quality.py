import numpy as np
from scipy.optimize import fsolve

import torch

from zero import stats
from .. import layer, solver


def metric(s, t, probabilty, method):
    x = np.linspace(-255, 255, 511, endpoint=True)
    p = probabilty(x, *s)
    q = probabilty(x, *t)
    return method(p, q)


def solve(start, end, alpha, probabilty=stats.gaussian.ggd, method=stats.metric.hellinger):
    def core(start, end, alpha):
        d = metric(start, end, probabilty, method)

        def loss(param):
            s = metric(start, param, probabilty=probabilty, method=method)
            e = metric(end, param, probabilty=probabilty, method=method)
            return s - alpha * d, s + e - d

        return fsolve(loss, start * (1 - alpha) + alpha * end)

    assert start.shape == end.shape

    shape = start.shape
    start = start.reshape([-1, shape[-1]])
    end = end.reshape([-1, shape[-1]])
    if isinstance(alpha, (int, float)):
        alpha = [alpha for _ in range(len(start))]
    elif len(shape) >= 3:
        alpha = np.repeat(np.expand_dims(alpha, axis=-1), shape[-2], axis=-1).reshape([-1])
    root = []
    for i in range(len(start)):
        root.append(core(start[i], end[i], alpha[i]))
    root = np.stack(root, 0)
    return root.reshape(shape)


class Mapper(torch.nn.Module):
    def __init__(self):
        super(Mapper, self).__init__()

    def forward(self, inputs, source, target):
        const = np.log(
            stats.gaussian.constraint(target[..., 0], target[..., 1]) /
            stats.gaussian.constraint(source[..., 0], source[..., 1])
        )

        if isinstance(inputs, (list, tuple)):
            device, dtype = inputs[0].device, inputs[0].dtype
        else:
            device, dtype = inputs.device, inputs.dtype
        source = torch.tensor(source, dtype=dtype, device=device)[..., None, None, :, :]
        target = torch.tensor(target, dtype=dtype, device=device)[..., None, None, :, :]
        const = torch.tensor(const, dtype=dtype, device=device)[..., None, None, :]

        if isinstance(inputs, (list, tuple)):
            return type(inputs)([self._core(_, source, target, const) for _ in inputs])
        return self._core(inputs, source, target, const)

    def _core(self, tensor, source, target, const):
        sign = torch.sign(tensor)
        if len(tensor.shape) == 4:
            tensor = tensor.permute((0, 2, 3, 1))
        sa, sb = source[..., 0], source[..., 1]
        ta, tb = target[..., 0], target[..., 1]

        outputs = sa * torch.pow(torch.abs(tensor), sb) + const
        outputs = torch.pow(outputs.clip(0) / ta, 1 / tb)

        if len(tensor.shape) == 4:
            outputs = outputs.permute(0, 3, 1, 2)
        return sign * outputs


class Transform(torch.nn.Module):
    def __init__(self):
        super(Transform, self).__init__()
        self._mapper = Mapper()
        self._gradient_tx = layer.GradientX(pad=True)
        self._gradient_ty = layer.GradientY(pad=True)
        self._gradient_fx = layer.GradientX(pad=False)
        self._gradient_fy = layer.GradientY(pad=False)

    def forward(self, tensor, source, target, const=None):
        if const is None:
            const = tensor.mean(dim=(-2, -1))

        ty = self._gradient_ty(tensor)
        tx = self._gradient_tx(tensor)

        if source.shape[-2] == 2:
            source_y, source_x = source[..., 0, :], source[..., 1, :]
            target_y, target_x = target[..., 0, :], target[..., 1, :]
        elif source.shape[-2] == 1:
            source_y, source_x = source[..., 0, :], source[..., 0, :]
            target_y, target_x = target[..., 0, :], target[..., 0, :]
        else:
            raise RuntimeError

        my = self._mapper(ty, source_y, target_y)
        mx = self._mapper(tx, source_x, target_x)

        fy = self._gradient_fy(my)
        fx = self._gradient_fx(mx)
        return solver.possion(fx + fy, const)
