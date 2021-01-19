import math
import torch
from zero.torch import nn


class Attention(torch.nn.Module):
    r"""Pytorch implementation of multi-headed attention based on Attention Is All You Need.
    """

    def __init__(self, from_channel, to_channel,
                 num_header=1, header_size=512,
                 query=None, key=None, value=None,
                 dropout=0.0):
        super(Attention, self).__init__()
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
        :param from_tensor: float tensor of shape [..., F, Cf]
        :param to_tensor: float tensor with shape [..., T, Ct]
        :param attention_mask: a tensor with shape [..., F, T],  the values should be 1 or 0.
        :return:
        '''

        from_tensor_shape = from_tensor.shape
        to_tensor_shape = to_tensor.shape

        query = self._query(from_tensor)  # [B, F, N*H]
        key = self._key(to_tensor)  # [B, T, N*H]
        value = self._value(to_tensor)  # [B, T, N*H]

        query = torch.reshape(query, (*from_tensor_shape[0:-1], *self._shape))  # [..., F, N, H]
        key = torch.reshape(key, (*to_tensor_shape[0:-1], *self._shape[::-1]))  # [..., T, H, N]
        value = torch.reshape(value, (*to_tensor_shape[0:-1], *self._shape))  # [..., T, N, H]

        # [B, N, F, H] * [B, N, H, T]
        score = torch.transpose(query, -3, -2) @ torch.transpose(key, -3, -1) * self._alpha  # [..., N, F, T]

        if mask is not None:
            mask = torch.unsqueeze(mask, 1)
            adder = (1 - mask) * (-100000.0)
            score += adder

        probs = self._probs(score)  # [B, N, F, T]
        context = probs @ torch.transpose(value, -3, -2)  # [..., N, F, T] * [..., N, T, H]
        context = torch.transpose(context, -3, -2)  # [..., F, N, H]

        return torch.reshape(context, (*from_tensor_shape[:-1], -1))  # [..., F, N * H]


class Encoder(torch.nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
