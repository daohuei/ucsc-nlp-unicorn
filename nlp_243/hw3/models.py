import torch
from torch import nn
from torch.nn import functional as F

from gpu import device

# The baseline model using RNN and one-hot encoding
class BaselineModel(nn.Module):
    def __init__(
        self, input_size, output_size, seq_len, emb_layer,
    ):
        super(BaselineModel, self).__init__()
        # Defining some parameters
        self.input_size = input_size
        self.output_size = output_size
        self.seq_len = seq_len

        # Defining the layers
        # Embedding Layer
        self.emb_layer = emb_layer
        # RNN Layer
        self.rnn = nn.RNNCell(input_size, output_size)

    def forward(self, x):
        batch_size = x.size(0)
        out = self.emb_layer(x)

        hiddens = []

        # Initializing hidden state for first input
        hidden = torch.zeros(batch_size, self.output_size).to(device)

        for t in range(self.seq_len):
            hidden = self.rnn(out[:, t, :], hidden)
            # make an additional dimension
            hiddens.append(torch.unsqueeze(hidden, dim=2))

        # concat all the hidden layer
        hiddens = torch.cat(hiddens, dim=2)
        out = F.softmax(hiddens, dim=1)
        return out


class RNNTwoLayerModel(nn.Module):
    def __init__(
        self, input_size, hidden_size, output_size, seq_len, emb_layer,
    ):
        super(RNNTwoLayerModel, self).__init__()
        # Defining some parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.seq_len = seq_len

        # Defining the layers
        # Embedding Layer
        self.emb_layer = emb_layer
        # RNN Layer
        self.rnn1 = nn.RNNCell(input_size, hidden_size)
        self.rnn2 = nn.RNNCell(hidden_size, output_size)

    def forward(self, x):
        batch_size = x.size(0)
        out = self.emb_layer(x)

        # First Layer
        hiddens = []
        hidden = torch.zeros(batch_size, self.hidden_size).to(device)
        for t in range(self.seq_len):
            hidden = self.rnn1(out[:, t, :], hidden)
            # make an additional dimension
            hiddens.append(torch.unsqueeze(hidden, dim=1))
        # concat all the hidden layer
        out = torch.cat(hiddens, dim=1)

        # Second Layer
        hiddens = []
        hidden = torch.zeros(batch_size, self.output_size).to(device)
        for t in range(self.seq_len):
            hidden = self.rnn2(out[:, t, :], hidden)
            # make an additional dimension
            hiddens.append(torch.unsqueeze(hidden, dim=2))
        # concat all the hidden layer
        out = torch.cat(hiddens, dim=2)

        out = F.softmax(out, dim=1)
        return out
