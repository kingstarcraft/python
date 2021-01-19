import numpy as np
import torch


class _State(torch.nn.Module):
    def __init__(self):
        super(_State, self).__init__()

    def state(self):
        return self._state

    def clear(self):
        if isinstance(self._state, list) or isinstance(self._state, tuple):
            for s in self._state:
                torch.nn.init.zeros_(s)
        else:
            torch.nn.init.zeros_(self._state)


class _Memory(torch.nn.Module):
    def __init__(self, length, channel):
        '''
        :param length: the number of memory locations.
        :param channel: the vector size at each location.
        '''
        super(_Memory, self).__init__()
        self._length = length
        self._channel = channel
        self.register_buffer('_seed', torch.zeros(self._length, self._channel))

        # ==================================
        stdev = 1 / (np.sqrt(self._length + self._channel))
        torch.nn.init.uniform_(self._seed, -stdev, stdev)

    def reset(self, batch_size):
        self._batch_size = batch_size
        # self._memory = self.memory.clone().repeat(batch_size, 1, 1)
        self._memory = self._seed.clone().repeat(batch_size, 1, 1)

    def size(self):
        return self._length, self._channel

    def read(self, weight):
        return torch.matmul(weight.unsqueeze(1), self._memory).squeeze(1)

    def write(self, weight, memory):
        '''
        :param weight: the weight of memory.
        :param memory: the memory of erase and add.
        '''

        '''
        why the grad is differnet between section1 and section2 
        '''
        #  section1
        #  weight = weight.unsqueeze(-1)
        #  erase = torch.matmul(weight, memory[0].unsqueeze(1))
        #  add = torch.matmul(weight, memory[1].unsqueeze(1))

        # section2
        erase = torch.matmul(weight.unsqueeze(-1), memory[0].unsqueeze(1))
        add = torch.matmul(weight.unsqueeze(-1), memory[1].unsqueeze(1))
        self._memory = self._memory * (1 - erase) + add

    def address(self, key, β, gate, shift, γ, weight):
        '''
        :param key: the vector of key.
        :param β: the key strength (focus).
        :param gate: the scalar interpolation gate
        :param shift: the weighting of shift
        :param γ: sharpen weighting scalar.
        :param weight: previous weight
        :return: the weight of key
        '''

        weight = self._interpolate(weight, self._similarity(key, β), gate)
        weight = self._shift(weight, shift)
        weight = self._sharpen(weight, γ)
        return weight

    def _similarity(self, key, β):
        key = key.view(self._batch_size, 1, -1)
        weight = torch.softmax(β * torch.cosine_similarity(self._memory + 1e-16, key + 1e-16, dim=-1), dim=1)
        return weight

    def _shift(self, weight, sharpen):
        result = torch.zeros(weight.size())
        for i in range(self._batch_size):
            result[i] = self._convolve(weight[i], sharpen[i])
        return result

    @staticmethod
    def _interpolate(previous, current, gate):
        return gate * current + (1 - gate) * previous

    @staticmethod
    def _convolve(weight, sharpen):
        """Circular convolution implementation."""
        assert sharpen.size(0) == 3
        weight = torch.cat([weight[-1:], weight, weight[:1]])
        return torch.conv1d(weight.view(1, 1, -1), sharpen.view(1, 1, -1)).view(-1)

    @staticmethod
    def _sharpen(weight, γ):
        weight = weight ** γ
        weight = torch.div(weight, torch.sum(weight, dim=1).view(-1, 1) + 1e-16)
        return weight


class _Head(_State):
    def __init__(self, channel, memory):
        super(_Head, self).__init__()
        self._channel = channel
        self._memory = memory
        self.register_buffer('_seed', torch.zeros(self._memory.size()[0]))

    def reset(self, batch_size):
        self._state = self._seed.clone().repeat(batch_size, 1)

    @staticmethod
    def _split_cols(mat, lengths):
        """Split a 2D matrix to variable length columns."""
        assert mat.size()[1] == sum(lengths), "Lengths must be summed to num columns"
        lengths = np.cumsum([0] + list(lengths))
        results = []
        for start, end in zip(lengths[:-1], lengths[1:]):
            results += [mat[:, start:end]]
        return results

    def _address(self, key, β, gate, shift, γ):
        #   key = key.clone()
        β = torch.nn.functional.softplus(β)
        gate = torch.sigmoid(gate)
        shift = torch.softmax(shift, dim=1)
        γ = 1 + torch.nn.functional.softplus(γ)
        weight = self._memory.address(key, β, gate, shift, γ, self._state)
        self._state = weight
        return weight


class _Reader(_Head):
    def __init__(self, channel, memory):
        super(_Reader, self).__init__(channel, memory)
        self._lengths = self._memory.size()[1], 1, 1, 3, 1
        self._net = torch.nn.Linear(channel, sum(self._lengths))

        # ==================================
        torch.nn.init.xavier_uniform_(self._net.weight, gain=1.4)
        torch.nn.init.normal_(self._net.bias, std=0.01)

    def forward(self, features):
        params = self._net(features)
        params = self._split_cols(params, self._lengths)
        weight = self._address(*params)
        data = self._memory.read(weight)
        return data


class _Writer(_Head):
    def __init__(self, channel, memory):
        super(_Writer, self).__init__(channel, memory)
        self._lengths = self._memory.size()[1], 1, 1, 3, 1, self._memory.size()[1], self._memory.size()[1]
        self._net = torch.nn.Linear(channel, sum(self._lengths))

        # ==================================
        torch.nn.init.xavier_uniform_(self._net.weight, gain=1.4)
        torch.nn.init.normal_(self._net.bias, std=0.01)

    def forward(self, features):
        params = self._net(features)
        params = self._split_cols(params, self._lengths)
        weight = self._address(*params[:-2])
        erase = torch.sigmoid(params[-2])
        self._memory.write(weight, (erase, params[-1]))
        return


class _Controller(_State):
    def __init__(self, input_size, output_size, num_layers):
        super(_Controller, self).__init__()
        self._size = (input_size, output_size)
        self._net = torch.nn.LSTM(input_size=self._size[0], hidden_size=self._size[1], num_layers=num_layers)
        # self._seed = torch.nn.Parameter(torch.zeros(2, num_layers, 1, self._size[1]))

        # ==================================
        self._seed1 = torch.nn.Parameter(torch.randn(num_layers, 1, self._size[1]) * 0.05)
        self._seed2 = torch.nn.Parameter(torch.randn(num_layers, 1, self._size[1]) * 0.05)

        for p in self._net.parameters():
            if p.dim() == 1:
                torch.nn.init.constant_(p, 0)
            else:
                stdev = 5 / (np.sqrt(input_size + output_size))
                torch.nn.init.uniform_(p, -stdev, stdev)

    def reset(self, batch_size):
        self._state = self._seed1.clone().repeat(1, batch_size, 1), self._seed2.clone().repeat(1, batch_size, 1)

    def forward(self, inputs):
        if inputs.dim() == 2:
            inputs = inputs.unsqueeze(0)
        else:
            assert inputs.size(0) == 1
        outputs, state = self._net(inputs, self._state)
        self._state = state
        return outputs.squeeze(0)

    def size(self):
        return self._size

    def state(self):
        return self._state


class NTM(_State):
    def __init__(self, input_size, output_size, num_headers, hidden_size, num_layers, memory_size,
                 active=torch.nn.Sigmoid()):
        super(NTM, self).__init__()
        assert len(memory_size) == 2
        self._memory = _Memory(*memory_size)

        self._state = []
        self.register_buffer("_seed", torch.zeros(num_headers, 1, memory_size[1]))
        self._controller = _Controller(input_size + num_headers * memory_size[1], hidden_size, num_layers)
        self._heads = torch.nn.ModuleList([])
        for i in range(num_headers):
            self._heads += [_Reader(hidden_size, self._memory), _Writer(hidden_size, self._memory)]

            # ==================================
            self._seed[i, :] = torch.randn(1, memory_size[-1]) * 0.01
        net = [torch.nn.Linear(hidden_size + num_headers * memory_size[1], output_size)]
        if active is not None:
            net.append(active)

        self._net = torch.nn.Sequential(*net)
        # ==================================
        torch.nn.init.xavier_uniform_(net[0].weight, gain=1)
        torch.nn.init.normal_(net[0].bias, std=0.01)

    def reset(self, batch_size):
        self._state = [s.clone().repeat(batch_size, 1) for s in self._seed]
        self._controller.reset(batch_size)
        for head in self._heads:
            head.reset(batch_size)
        self._memory.reset(batch_size)

    def clear(self):
        super(NTM, self).clear()
        self._controller.clear()
        for head in self._heads:
            head.clear()

    def _forward(self, inputs):
        assert inputs.dim() == 2
        inputs = torch.cat([inputs] + self._state, dim=1)
        control = self._controller(inputs)
        state = []
        for head in self._heads:
            data = head(control)
            if data is not None:
                state.append(data)
        outputs = torch.cat([control] + state, dim=1)
        self._state = state
        return self._net(outputs)

    def forward(self, inputs):
        outputs = []
        for input in inputs:
            outputs.append(self._forward(input))
        return torch.stack(outputs, 0)


def build(input_size, output_size, num_headers=1, hidden_size=256, num_layers=1, memory_size=(1024, 256)):
    return NTM(input_size, output_size, num_headers, hidden_size, num_layers, memory_size)