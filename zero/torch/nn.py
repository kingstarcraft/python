import math
import torch
from . import init, util
from zero import function


class Layer(torch.nn.Module):
    __init = None

    @staticmethod
    def init(func):

        def call(cls, *args, **kwargs):
            cls = eval('torch.nn.' + cls.__name__)
            func = cls.__init__
            arguments = {}
            if func.__code__.co_argcount > 1:
                argument = function.Argument(func, *args, **kwargs)
                for name in func.__code__.co_varnames[1:func.__code__.co_argcount]:
                    arguments[name] = argument[name]
            return cls(**arguments)

        def inner(*args, **kwargs):
            argument = function.Argument(func, *args, **kwargs)
            self = argument['self']
            active = argument['active']
            normalizer = argument['normalizer']
            out = (argument['out_features'] if 'out_features' in argument else argument['out_channels'])
            initilalizer = argument['initilalizer']

            Layer.__init__(self, initilalizer)
            func(*args, **kwargs)
            active = util.instance(active)
            normalizer = util.instance(normalizer, out)

            for cls in Layer.__subclasses__():
                if isinstance(self, cls):
                    self._net = torch.nn.Sequential(*([call(cls, *args, **kwargs)] +
                                                      ([] if normalizer is None else [normalizer]) +
                                                      ([] if active is None else [active])))
            self._init(active)

        return inner

    def __init__(self, initilalizer):
        super(Layer, self).__init__()
        self._initilalizer = initilalizer

    def forward(self, inputs):
        if callable(self._net):
            return self._net(inputs)
        return inputs

    def _init(self, active):
        with torch.no_grad():
            for layer in self._net:
                if 'BatchNorm' in str(layer):
                    if hasattr(layer, 'bias'):
                        torch.nn.init.zeros_(layer.bias)
                    if hasattr(layer, 'weight'):
                        torch.nn.init.ones_(layer.weight)

                elif hasattr(layer, 'weight'):
                    init.Initilalizer.run(layer.weight, active, self._initilalizer)
                    if hasattr(layer, 'bias'):
                        if layer.bias is None:
                            continue
                        size = init.Initilalizer._size(layer.weight)
                        if self._initilalizer == 'normal':
                            torch.nn.init.normal_(layer.bias, mean=0, std=1 / math.sqrt(size))
                        elif self._initilalizer == 'uniform':
                            torch.nn.init.uniform_(layer.bias, -1 / math.sqrt(size), 1 / math.sqrt(size))


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


class Module(torch.nn.Module):
    def __init__(self):
        super(Module, self).__init__()

    def forward(self, inputs):
        assert hasattr(self, '_net')
        return self._net(inputs)


class Axis(torch.nn.Module):
    def __init__(self, net, axis=1):
        super(Axis, self).__init__()
        if isinstance(net, list) or isinstance(net, tuple):
            self._net = torch.nn.Sequential(*net)
        else:
            self._net = net
        self._axis = axis

    def forward(self, inputs):
        net = self._net(torch.movedim(inputs, self._axis, -1))
        return torch.movedim(net, -1, self._axis)


class Dense(Module):
    def __init__(self, in_features: int, out_features: int,
                 bias=True, normalizer=None, active=None, initilalizer='uniform'):
        super(Dense, self).__init__()
        self._net = Axis(Linear(in_features, out_features, bias, normalizer, active, initilalizer), 1)


class LayerNorm(Module):
    def __init__(self, channels, axis=1):
        super(LayerNorm, self).__init__()
        self._net = Axis(torch.nn.LayerNorm(channels), axis=axis)
