import math
import torch
from zero.torch import nn


class MultiHeaded(torch.nn.Module):
    r"""Pytorch implementation of multi-headed attention based on Attention Is All You Need.
    """

    def __init__(self, from_channel, to_channel,
                 num_header=1, header_size=512,
                 query=None, key=None, value=None,
                 dropout=0.0):
        super(MultiHeaded, self).__init__()
        self._query = nn.Linear(from_channel, num_header * header_size,
                                active=query, initilalizer='trunc_normal')
        self._key = nn.Linear(to_channel, num_header * header_size,
                              active=key, initilalizer='trunc_normal')
        self._value = nn.Linear(to_channel, num_header * header_size,
                                active=value, initilalizer='trunc_normal')
        self._shape = [num_header, header_size]
        self._alpha = 1 / math.sqrt(header_size)
        self._probs = torch.nn.Sequential(
            torch.nn.Softmax(dim=-1),
            torch.nn.Dropout(dropout)
        )

    def forward(self, from_tensor, to_tensor, mask=None):
        '''
        :param from_tensor: float tensor of shape [B, Cf, F] or [B, F]
        :param to_tensor: float tensor with shape [B, Ct, T] or [B, T]
        :param attention_mask: a tensor with shape [B, F, T],  the values should be 1 or 0.
        :return:
        '''

        from_tensor_shape = from_tensor.shape
        to_tensor_shape = to_tensor.shape
        assert from_tensor_shape[0] == to_tensor_shape[0]
        if len(from_tensor_shape) == 2:
            from_tensor = torch.unsqueeze(from_tensor, 1)
        if len(to_tensor) == 2:
            to_tensor = torch.unsqueeze(to_tensor, 1)

        query = self._query(from_tensor)  # [B, N*H, F]
        key = self._key(to_tensor)  # [B, N*H, T]
        value = self._value(to_tensor)  # [B, N*H, T]

        shape = (from_tensor_shape[0], *self._shape, -1)
        query = torch.reshape(query, shape)  # [B, N, H, F]
        key = torch.reshape(key, shape)  # [B, N, H, T]
        value = torch.reshape(value, shape)  # [B, N, H, T]

        score = torch.transpose(query, 2, 3) @ key * self._alpha  # [B, N, F, T]

        if mask is not None:
            mask = torch.unsqueeze(mask, 1)
            adder = (1 - mask) * (-100000.0)
            score += adder

        probs = self._probs(score)  # [B, N, F, T]
        context = probs @ torch.transpose(value, 2, 3)  # [B, N, F, H]
        context = torch.transpose(context, 2, 3)
        return torch.reshape(context, [shape[0], shape[1] * shape[2], shape[3]])
