import math
import numpy as np
import torch
import zero
from . import util
from torch.nn import functional


class ResidualBlock(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, active=torch.nn.ReLU(inplace=True), normalizer=None):
        super(ResidualBlock, self).__init__()
        padding = (np.array(kernel_size) - 1) // 2

        net = [
            zero.torch.nn.Conv2d(channels, channels, kernel_size, 1, padding, normalizer=normalizer, active=active),
            zero.torch.nn.Conv2d(channels, channels, kernel_size, 1, padding, normalizer=normalizer)
        ]
        active = util.instance(active)
        self._active = active
        self._net = torch.nn.Sequential(*net)

    def forward(self, inputs):
        outputs = inputs + self._net(inputs)
        return outputs if self._active is None else self._active(outputs)


class Attention(torch.nn.Module):
    r"""Pytorch implementation of multi-headed attention based on Attention Is All You Need.
    """

    def __init__(self, in_channels, out_channels,
                 query=None, key=None, value=None, dropout=0.0, initilalizer='trunc_normal'):
        '''
        :param in_channels: a tuple with from_tensor channel an to_tensor channel
        :param out_channels:  a tuple of header_size and header_channels
        :param query: active_funcion of query
        :param key: active_funcion of key
        :param value: active_funcion of value
        :param dropout:
        '''

        super(Attention, self).__init__()
        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)
        if isinstance(out_channels, int):
            out_channels = (1, out_channels)
        assert len(in_channels) == 2
        assert len(out_channels) == 2

        self._query = zero.torch.nn.Dense(in_channels[0], out_channels[0] * out_channels[1],
                                          active=query, initilalizer=initilalizer)
        self._key = zero.torch.nn.Dense(in_channels[1], out_channels[0] * out_channels[1],
                                        active=key, initilalizer=initilalizer)
        self._value = zero.torch.nn.Dense(in_channels[1], out_channels[0] * out_channels[1],
                                          active=value, initilalizer=initilalizer)
        self._shape = out_channels
        self._alpha = 1 / math.sqrt(out_channels[1])
        self._probs = torch.nn.Sequential(
            torch.nn.Softmax(dim=-1),
            torch.nn.Dropout(dropout)
        )

    def forward(self, inputs, mask=None):
        '''
        :param inputs: a tuple of from_tensor and to_tensor with the shape with [B, C, F...] and  [B, C, T...])
        :param attention_mask: a tensor with shape [B, F, T],  the values should be 1 or 0.
        :return a tensor with shape [B, N*H, F...]:
        '''
        if isinstance(inputs, torch.Tensor):
            inputs = inputs, inputs
        shape = (inputs[0].shape[0], -1, *inputs[0].shape[2:])
        inputs = [torch.reshape(_, (*_.shape[0:2], -1)) for _ in inputs]

        from_tensor_shape = inputs[0].shape  # [B, Cf, F]
        to_tensor_shape = inputs[1].shape  # [B, Ct, T]

        query = self._query(inputs[0])  # [B, N*H,  F]
        key = self._key(inputs[1])  # [B, N*H, T]
        value = self._value(inputs[1])  # [B, N*H, T]

        query = torch.reshape(query, (from_tensor_shape[0], *self._shape, *from_tensor_shape[2:]))  # [B, N, H, F]
        key = torch.reshape(key, (to_tensor_shape[0], *self._shape, *to_tensor_shape[2:]))  # [B, N, H, T]
        value = torch.reshape(value, (to_tensor_shape[0], *self._shape, *to_tensor_shape[2:]))  # [B, N, H, T]

        # [B, N, F, H] * [B, N, H, T]
        score = torch.transpose(query, 2, 3) @ key * self._alpha  # [B, N, F, T]

        if mask is not None:
            mask = torch.unsqueeze(mask, 1)  # [B, 1, F, T]
            adder = (1 - mask) * (-1e5)
            score += adder

        probs = self._probs(score)  # [B, N, F, T]
        context = probs @ torch.transpose(value, 2, 3)  # [B, N, F, T] * [B, N, T, H]
        context = torch.transpose(context, 2, 3)  # [B, N, H, F]
        return torch.reshape(context, shape)  # [B, N * H, ...]


class Transformer(torch.nn.Module):
    def __init__(self, channels, attention_headers=12, attention_channels=256, feed_channels=3072,
                 query=None, key=None, value=None, intermediate=torch.nn.GELU,
                 attention_dropout=0.1, block_dropout=0.1, initilalizer='trunc_normal'):
        '''
        :param channels: a tuple of channels of from_tensor and to_tensor
        :param attention_channels:
        :param feed_channels:
        :param query:
        :param key:
        :param value:
        :param intermediate:
        :param attention_dropout:
        :param block_dropout:
        :param initilalizer:
        '''

        super(Transformer, self).__init__()
        if isinstance(channels, int):
            from_channels, to_channels = channels, channels
        else:
            from_channels, to_channels = channels
        self._attention = Attention(channels, (attention_headers, attention_channels), query, key, value,
                                    attention_dropout,
                                    initilalizer)
        self._dense = torch.nn.Sequential(
            zero.torch.nn.Dense(attention_headers * attention_channels, from_channels, initilalizer=initilalizer),
            torch.nn.Dropout(block_dropout)
        )
        self._attention_norm = zero.torch.nn.LayerNorm(from_channels)
        self._intermediate = zero.torch.nn.Axis((
            zero.torch.nn.Linear(from_channels, feed_channels, active=intermediate, initilalizer=initilalizer),
            zero.torch.nn.Linear(feed_channels, from_channels, initilalizer=initilalizer),
            torch.nn.Dropout(block_dropout)
        ))
        self._block_norm = zero.torch.nn.LayerNorm(from_channels)

    def forward(self, inputs, mask=None):
        '''
        :param inputs:  a tuple of channels of from_tensor and to_tensor
        :param mask:
        :return: a tensor with from_tensor_shape
        '''
        if isinstance(inputs, torch.Tensor):
            inputs = inputs, inputs
        net = self._dense(self._attention(inputs, mask=mask))
        attention = self._attention_norm(net + inputs[0])
        return self._block_norm(attention + self._intermediate(net))


class PositionEmbeding(torch.nn.Module):
    def __init__(self, shape, channels, mean=0, std=0.02):
        '''
        :param shape: the shape of position
        :param channels: the channel of position
        '''
        super(PositionEmbeding, self).__init__()
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


class ASPP(torch.nn.Module):
    def __init__(self, in_channels=512, out_channels=512, blocks=(6, 12, 18), normalizer=None, active=None):
        super(ASPP, self).__init__()
        self._blocks = [
            torch.nn.Sequential(
                torch.nn.AdaptiveAvgPool2d((1, 1)),
                zero.torch.nn.Conv2d(in_channels, out_channels, 1, 1, bias=False, normalizer=normalizer, active=active)
            ),
            zero.torch.nn.Conv2d(in_channels, out_channels, 1, 1, bias=False, normalizer=normalizer, active=active)
        ]
        for block in blocks:
            self._blocks.append(
                zero.torch.nn.Conv2d(
                    in_channels, out_channels, 3, 1,
                    bias=False, padding=block, dilation=block, normalizer=normalizer, active=active
                )
            )
        self._merge = zero.torch.nn.Conv2d(
            out_channels * len(self._blocks), out_channels, 1, bias=False, normalizer=normalizer, active=active
        )

    def forward(self, inputs):
        aspp = []
        for block in self._blocks:
            aspp.append(block(inputs))
        aspp[0] = functional.interpolate(aspp[0], size=aspp[-1].size()[2:], mode='bilinear', align_corners=True)
        return self._merge(torch.cat(aspp, dim=1))


class GradientReversal(torch.nn.Module):
    def __init__(self, gamma=1):
        super(GradientReversal, self).__init__()
        self.gamma = gamma

    def forward(self, x):
        return util.GradientReversalFunction.apply(x, self.gamma)


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(torch.nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height, -1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")

            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = torch.nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = torch.nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = torch.nn.Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args, **kwargs):
        self._update_u_v()
        return self.module.forward(*args, **kwargs)


class GradientX(torch.nn.Module):
    def __init__(self, reverse=False, pad=True):
        super(GradientX, self).__init__()
        self._core = torch.nn.Conv2d(kernel_size=(1, 2), in_channels=1, out_channels=1, bias=False)
        self._core.weight.data[:] = torch.Tensor([[[[1, -1]]]]) if reverse else torch.Tensor([[[[-1, 1]]]])
        self._pad = pad

    @torch.no_grad()
    def forward(self, tensor: torch.Tensor):
        n, c, h, w = tensor.shape
        if self._pad:
            pad = torch.zeros(n, c, h, w + 2, dtype=tensor.dtype, device=tensor.device)
            pad[..., :, 1:-1] = tensor
            pad[..., :, 0] = pad[..., :, 1]
            pad[..., :, w + 1] = pad[..., :, w]
        else:
            pad = tensor
        n, c, h, w = pad.shape
        gradient = self._core(pad.reshape(n * c, 1, h, w))
        return gradient.reshape([n, c, h, w - 1])


class GradientY(torch.nn.Module):
    def __init__(self, reverse=False, pad=True):
        super(GradientY, self).__init__()
        self._core = torch.nn.Conv2d(kernel_size=(2, 1), in_channels=1, out_channels=1, bias=False)
        self._core.weight.data[:] = torch.Tensor([[[1], [-1]]]) if reverse else torch.Tensor([[[-1], [1]]])
        self._pad = pad

    @torch.no_grad()
    def forward(self, tensor: torch.Tensor):
        n, c, h, w = tensor.shape
        if self._pad:
            pad = torch.zeros(n, c, h + 2, w, dtype=tensor.dtype, device=tensor.device)
            pad[..., 1:-1, :] = tensor
            pad[..., 0, :] = pad[..., 1, :]
            pad[..., h + 1, :] = pad[..., h, :]
        else:
            pad = tensor
        n, c, h, w = pad.shape
        gradient = self._core(pad.reshape(n * c, 1, h, w))
        return gradient.reshape([n, c, h - 1, w])


class GradientXOY(torch.nn.Module):
    def __init__(self, pad=False):
        super(GradientXOY, self).__init__()
        self._grad_y = GradientY(pad=pad)
        self._grad_x = GradientX(pad=pad)

    @torch.no_grad()
    def forward(self, inputs):
        if isinstance(inputs, (list, tuple)):
            assert len(inputs) == 2
            return self._grad_y(inputs[0]), self._grad_x(inputs[1])
        else:
            grad_y = self._grad_y(inputs)
            grad_x = self._grad_x(inputs)
            return [grad_y, grad_x]


class Laplacian(torch.nn.Module):
    def __init__(self, alpha=4, pad=True):
        super(Laplacian, self).__init__()
        self._core = torch.nn.Conv2d(kernel_size=(3, 3), in_channels=1, out_channels=1, bias=False)
        if alpha == 4:
            self._core.weight.data[:] = torch.Tensor([0, 1, 0, 1, -4, 1, 0, 1, 0]).reshape([1, 1, 3, 3])
        elif alpha == 8:
            self._core.weight.data[:] = torch.Tensor([1, 1, 1, 1, -8, 1, 1, 1, 1]).reshape([1, 1, 3, 3])
        self._pad = pad

    @torch.no_grad()
    def forward(self, tensor):
        n, c, h, w = tensor.shape
        if self._pad:
            pad = torch.zeros(n, c, h + 2, w + 2, dtype=tensor.dtype, device=tensor.device)
            pad[..., 1:-1, 1:-1] = tensor

            pad[..., 0, :] = pad[..., 1, :]
            pad[..., h + 1, :] = pad[..., h, :]

            pad[..., :, 0] = pad[..., :, 1]
            pad[..., :, w + 1] = pad[..., :, w]
        else:
            pad = tensor
        n, c, h, w = pad.shape
        return self._core(pad.reshape([n * c, 1, h, w])).reshape((n, c, h - 2, w - 2))
