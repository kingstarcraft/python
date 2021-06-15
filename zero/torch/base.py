import torch


class Net(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(Net, self).__init__()
        self._net = torch.nn.Sequential(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self._net(*args, **kwargs)
