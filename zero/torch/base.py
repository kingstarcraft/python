import torch


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

    def __call__(self, *args, **kwargs):
        return self._net(*args, **kwargs)
