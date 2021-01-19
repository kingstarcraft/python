import torch
from zero.torch import layer


class Bert(torch.nn.Module):
    def __init__(self, vocab_channel, type_channel=None, max_length=None, hidden_size=256):
        super(Bert, self).__init__()
        self._vocab = torch.nn.Embedding(vocab_channel, hidden_size)
        if type_channel is not None:
            self._type = torch.nn.Embedding(type_channel, hidden_size)
        if max_length is not None:
            self._poistion = layer.embebding.Position(max_length, hidden_size)

    def forward(self, vocab, type=None, mask=None):
        embebding = self._vocab(vocab)
        if type is not None:
            assert hasattr(self, '_type')
            type = self._type(type)
            embebding += type
        if hasattr(self, '_poistion'):
            embebding = self._poistion(embebding)

        return
