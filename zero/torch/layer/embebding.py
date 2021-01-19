import torch


class Position(torch.nn.Module):
    def __init__(self, shape, channels, mean=0, std=0.02):
        '''
        :param shape: the shape of position
        :param channels: the channel of position
        '''
        super(Position, self).__init__()
        if isinstance(shape, int):
            self.weight = torch.nn.Parameter(torch.Tensor(shape, channels))
        else:
            self.weight = torch.nn.Parameter(torch.Tensor(*channels, shape))
        torch.nn.init.trunc_normal_(self.weight, mean=mean, std=std, a=mean - 2 * std, b=mean + 2 * std)

    def forward(self, inputs, start=None):
        '''
        :param inputs: batch_size * length * channels or batch_size * channels * height * weight
        :return:
        '''
        if len(inputs.shape) == 3:
            if start is None:
                start = 0
            weight = self.weight[start:start + inputs.shape[1]]
        else:
            input_shape = inputs.shape
            weight_shape = self.weight.shape
            offset = len(input_shape) - len(weight_shape)
            assert offset >= 0
            assert input_shape[offset] == weight_shape[0]
            offset = offset + 1
            weight_shape = weight_shape[1:]
            if start is None:
                start = [0 for _ in weight_shape]
            assert len(start) == len(weight_shape)
            slices = [slice(None)]
            for id in range(0, len(weight_shape)):
                end = start[id] + input_shape[offset + id]
                assert end <= weight_shape[id]
                slices.append(slice(start[id], end))
            weight = self.weight[slices]
        return inputs + torch.unsqueeze(weight, dim=0)
