import torch
from .__base__ import Layer


class Sine(torch.nn.Module):
    __first = True

    def __init__(self, w=30):
        super(Sine, self).__init__()
        self._w = w
        self._first = Sine.__first
        Sine.__first = False

    def __float__(self):
        return 0.0 if self._first else float(self._w)

    def forward(self, inputs):
        return torch.sin(self._w * inputs)


class Linear(Layer):
    @Layer.init
    def __init__(self, in_features: int, out_features: int,
                 bias=True, normalizer=None, active=None, initilalizer='uniform'):
        pass


class Conv1d(Layer):
    @Layer.init
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros',
                 normalizer=None, active=None, initilalizer='uniform'):
        pass


class Conv2d(Layer):
    @Layer.init
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros',
                 normalizer=None, active=None, initilalizer='uniform'):
        pass


class ConvTranspose2d(Layer):
    @Layer.init
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros',
                 normalizer=None, active=None, initilalizer='uniform'):
        pass
