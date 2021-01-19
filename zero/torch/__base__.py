import math
import torch
import zero.torch.init as init
import zero.function as function


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
                for name in func.__code__.co_varnames[1:]:
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
            if type(active).__name__ in ('type', 'function'):
                active = active()
            if type(normalizer).__name__ in ('type', 'function'):
                normalizer = normalizer(out)
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
                    continue

                if hasattr(layer, 'weight'):
                    init.Initilalizer.run(layer.weight, active, self._initilalizer)
                    if hasattr(layer, 'bias'):
                        if layer.bias is None:
                            continue
                        size = init.Initilalizer._size(layer.weight)
                        if self._initilalizer == 'normal':
                            torch.nn.init.normal_(layer.bias, mean=0, std=1 / math.sqrt(size))
                        elif self._initilalizer == 'uniform':
                            torch.nn.init.uniform_(layer.bias, -1 / math.sqrt(size), 1 / math.sqrt(size))
