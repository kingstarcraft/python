import torch
import numpy as np
import zero.function as function


class Sine(torch.nn.Module):
    __first = True

    def __init__(self, w=30):
        super(Sine, self).__init__()
        self._w = w
        Sine.__first = False

    def get(self, size):
        if self.__first:
            return 1 / size
        else:
            return np.sqrt(6 / size) / self._w

    def forward(self, inputs):
        return torch.sin(self._w * inputs)


class Layer(torch.nn.Module):
    @staticmethod
    def init(func):

        def call(cls, *args, **kwargs):
            cls = eval('torch.nn.' + cls.__name__)
            func = cls.__init__
            arguments = {}
            if func.__code__.co_argcount > 1:
                argument = function.Argument(func, *args, **kwargs)
                for name in func.__code__.co_varnames[1:]:
                    arguments[name] = argument[name]
            return cls(**arguments)

        def inner(*args, **kwargs):
            argument = function.Argument(func, *args, **kwargs)
            self = argument['self']
            active = argument['active']
            initilalizer = argument['initilalizer']
            Layer.__init__(self, initilalizer)
            func(*args, **kwargs)
            if type(active).__name__ in ('type', 'function'):
                active = active()
            for cls in Layer.__subclasses__():
                if isinstance(self, cls):
                    self._core = torch.nn.Sequential(
                        *([call(cls, *args, **kwargs)] + ([] if active is None else [active])))
            self._init(active)

        return inner

    def __init__(self, initilalizer):
        super(Layer, self).__init__()
        self._initilalizer = initilalizer

    def forward(self, inputs):
        if callable(self._core):
            return self._core(inputs)
        return inputs

    def _init(self, active):
        with torch.no_grad():
            for layer in self._core:
                if hasattr(layer, 'weight'):
                    if layer.weight.dim() < 2:
                        continue
                    print('init', str(layer))
                    size = layer.weight.size(1)
                    if layer.weight.dim() > 2:
                        size = size * layer.weight[0][0].numel()

                    if self._initilalizer == 'normal':
                        if active is None or isinstance(active, torch.nn.SELU):
                            torch.nn.init.normal_(layer.weight, std=1 / np.sqrt(size))
                        elif isinstance(active, Sine):
                            torch.nn.init.normal_(layer.weight, 0, active.get(size))
                        elif isinstance(active, torch.nn.ReLU) or \
                                isinstance(active, torch.nn.Softplus) or \
                                isinstance(active, torch.nn.PReLU):
                            torch.nn.init.kaiming_normal_(layer.weight, a=0.0, nonlinearity='relu', mode='fan_in')
                        elif isinstance(active, torch.nn.Sigmoid) or isinstance(active, torch.nn.Tanh):
                            torch.nn.init.xavier_normal_(layer.weight)
                        elif isinstance(active, torch.nn.ELU):
                            torch.nn.init.normal_(layer.weight, std=np.sqrt(1.5505188080679277 / size))
                        else:
                            raise NotImplementedError('normal init of %s was not implemented.' % type(active))
                    elif self._initilalizer == 'uniform':
                        if active is None or isinstance(active, torch.nn.SELU):
                            std = 1 / np.sqrt(size)
                            torch.nn.init.uniform_(layer.weight, -std, std)
                        elif isinstance(active, Sine):
                            torch.nn.init.uniform_(layer.weight, -active.get(size), active.get(size))
                        elif isinstance(active, torch.nn.ReLU) or \
                                isinstance(active, torch.nn.Softplus) or \
                                isinstance(active, torch.nn.PReLU):
                            torch.nn.init.kaiming_uniform_(layer.weight, a=0.0, nonlinearity='relu', mode='fan_in')
                        elif isinstance(active, torch.nn.Sigmoid) or isinstance(active, torch.nn.Tanh):
                            torch.nn.init.xavier_uniform_(layer.weight)
                        elif isinstance(active, torch.nn.ELU):
                            std = np.sqrt(1.5505188080679277 / size)
                            torch.nn.init.uniform_(layer.weight, -std, std)
                        else:
                            raise NotImplementedError('uniform init of %s was not implemented.' % type(active))
                    elif self._initilalizer is not None:
                        raise NotImplementedError('%s init was not implemented.' % self._initilalizer)
                if hasattr(layer, 'bias'):
                    if layer.bias is None:
                        continue

                    if self._initilalizer == 'normal':
                        torch.nn.init.normal_(layer.bias, mean=0, std=0.01)
                    elif self._initilalizer == 'uniform':
                        torch.nn.init.uniform_(layer.bias, -0.01, 0.01)


class Linear(Layer):
    @Layer.init
    def __init__(self, in_features: int, out_features: int, bias=True, active=None, initilalizer='uniform'):
        pass


class Conv1d(Layer):
    @Layer.init
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros',
                 active=None, initilalizer='normal'):
        pass
