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

        return fsolve(loss, (start + end) / 2)

    assert start.shape == end.shape

    shape = start.shape
    start = start.reshape([-1, shape[-1]])
    end = end.reshape([-1, shape[-1]])
    if isinstance(alpha, (int, float)):
        alpha = [alpha for _ in range(len(start))]
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
            stats.gaussian.constraint(target[:, 0], target[:, 1]) /
            stats.gaussian.constraint(source[:, 0], source[:, 1])
        )

        if isinstance(inputs, (list, tuple)):
            device, dtype = inputs[0].device, inputs[0].dtype
        else:
            device, dtype = inputs.device, inputs.dtype
        source = torch.tensor(source, dtype=dtype, device=device)
        target = torch.tensor(target, dtype=dtype, device=device)
        const = torch.tensor(const, dtype=dtype, device=device)

        if isinstance(inputs, (list, tuple)):
            return type(inputs)([self.core(_, source, target, const) for _ in inputs])
        return self._core(inputs, source, target, const)

    def _core(self, tensor, source, target, const):
        sign = torch.sign(tensor)
        if len(tensor.shape) == 4:
            tensor = torch.permute(tensor, (0, 2, 3, 1))
        sa, sb = source[:, 0], source[:, 1]
        ta, tb = target[:, 0], target[:, 1]

        outputs = sa * torch.pow(torch.abs(tensor), sb) + const
        outputs = torch.pow(outputs.clip(0) / ta, 1 / tb)

        if len(tensor.shape) == 4:
            outputs = torch.permute(outputs, (0, 3, 1, 2))
        return sign * outputs


class Transform(torch.nn.Module):
    def __init__(self):
        super(Transform, self).__init__()
        self._mapper = Mapper()
        self._gradient_tx = layer.GradientX(pad=True)
        self._gradient_ty = layer.GradientY(pad=True)
        self._gradient_fx = layer.GradientX(pad=False)
        self._gradient_fy = layer.GradientY(pad=False)

    def forward(self, tensor, source, tagret, const=None):
        if const is None:
            const = tensor.mean(dim=(-2, -1))
        tx = self._gradient_tx(tensor)
        ty = self._gradient_ty(tensor)
        mx = self._mapper(tx, source, tagret)
        my = self._mapper(ty, source, tagret)
        fx = self._gradient_fx(mx)
        fy = self._gradient_fy(my)
        return solver.possion(fx + fy, const)
