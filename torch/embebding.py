import torch


class Position(torch.nn.Module):
    def __init__(self, shape, channels):
        '''
        :param shape: the shape of position
        :param channels: the channel of position
        '''
        super(Position, self).__init__()
        if isinstance(shape, int):
            shape = (shape,)
        self.weight = torch.nn.Parameter(torch.Tensor(channels, *shape))
        torch.nn.init.trunc_normal_(self.weight, mean=0, std=0.02, )

    def forward(self, inputs, start=None):
        '''
        :param inputs: bath_size*channels*length
        :return:
        '''
        input_shape = inputs.shape
        weight_shape = self.weight.shape
        offset = len(input_shape) - len(weight_shape)
        assert offset >= 0
        assert input_shape[offset] == weight_shape[0]
        offset = offset+1
        weight_shape = weight_shape[1:]
        if shape is None:
            shape = [0 for _ in weight_shape]
        assert len(shape) == len(weight_shape)
        slices = []
        for id in range(0,  len(weight_shape)):
            end = start[id]+weight_shape[id]
            assert end <= input_shape[offset + id]
            slices.append(slice(start[id], end)
        return inputs + self.weight[..., *slices]
