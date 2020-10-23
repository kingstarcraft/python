import torch
import math
import zero.torch.layers as layers


class Initilalizer(object):
    __func = None

    @staticmethod
    def __size(weight):
        assert weight.dim() >= 2
        size = weight.size(1)
        if weight.dim() > 2:
            size = size * weight[0][0].numel()
        return size

    @staticmethod
    def __get(weight, active):
        size = Initilalizer.__size(weight)
        if active is None and isinstance(active, torch.nn.SELU):
            return 1 / math.sqrt(size)
        elif isinstance(active, layers.Sine):
            param = float(active)
            return 1 / size if param is None else math.sqrt(6 / size) / param
        elif isinstance(active, torch.nn.ReLU) or \
                isinstance(active, torch.nn.Softplus) or \
                isinstance(active, torch.nn.PReLU):
            return 'kaiming'
        elif isinstance(active, torch.nn.Sigmoid) or isinstance(active, torch.nn.Tanh):
            return 'xavier'
        elif isinstance(active, torch.nn.ELU):
            return math.sqrt(1.5505188080679277 / size)
        else:
            raise NotImplementedError('Get param of %s init was not implemented.' % type(active))

    @staticmethod
    def run(weight, active, type='normal'):
        if active is not None or Initilalizer.__func is None:
            param = Initilalizer.__get(weight, active)
            if isinstance(param, str):
                func = lambda tensor: eval('torch.nn.init.%s_%s_' % (param, type))(
                    tensor, a=0.0, nonlinearity='relu', mode='fan_in')
            else:
                if type == 'normal':
                    func = lambda tensor: torch.nn.init.normal_(
                        tensor, std=param)
                elif type == 'uniform':
                    func = lambda tensor: torch.nn.init.uniform_(
                        tensor, -param, param)
                else:
                    raise NotImplementedError('%s init was not implemented.' % type)
            Initilalizer.__func = func

        if Initilalizer.__func is not None:
            Initilalizer.__func(weight)


def normal_(tensor, active):
    Initilalizer.run(tensor, active, type='normal')


def uniform_(tensor, active):
    Initilalizer.run(tensor, active, type='uniform')
