import numpy as np
import torch
import zero


class Residual(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, active=torch.nn.ReLU(inplace=True), normalizer=None):
        super(Residual, self).__init__()
        padding = (np.array(kernel_size) - 1) // 2

        net = [
            zero.torch.nn.Conv2d(channels, channels, kernel_size, 1, padding, normalizer=normalizer, active=active),
            zero.torch.nn.Conv2d(channels, channels, kernel_size, 1, padding, normalizer=normalizer)
        ]
        if type(active).__name__ in ('type', 'function'):
            active = active()
        self._active = active
        self._net = torch.nn.Sequential(*net)

    def forward(self, x):
        y = x + self._net(x)
        if self._active is None:
            return y
        else:
            return self._active(y)
