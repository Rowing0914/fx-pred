import torch
import torch.nn as nn


class RNNFamily(nn.Module):
    def __init__(self, dim_in=1, dim_hidden=128, num_layers=1, dropout_rate=0.25, model_type="gru", device="cpu"):
        super(RNNFamily, self).__init__()
        self._device = device
        self._model_type = model_type
        self._num_layers = num_layers
        self._dim_hidden = dim_hidden

        if self._model_type == "lstm":
            self.seq_model = nn.LSTM(dim_in, dim_hidden, num_layers, dropout=dropout_rate, batch_first=True)
        elif self._model_type == "gru":
            self.seq_model = nn.GRU(dim_in, dim_hidden, num_layers, dropout=dropout_rate, batch_first=True)
        else:
            raise ValueError

        self.mlp = nn.Sequential(
            nn.Linear(dim_hidden, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.l_out = nn.Sigmoid()

    def forward(self, inputs):
        self.seq_model.flatten_parameters()
        batch_size, seq_len, dim = inputs.shape

        # Set initial hidden and cell states
        h0 = torch.zeros(self._num_layers, batch_size, self._dim_hidden).to(self._device)
        c0 = torch.zeros(self._num_layers, batch_size, self._dim_hidden).to(self._device)

        if self._model_type == "lstm":
            hidden, _ = self.seq_model(inputs, (h0, c0))
        elif self._model_type == "gru":
            hidden, _ = self.seq_model(inputs, h0)
        else:
            raise ValueError

        # Decode the hidden state of the last time step
        _input = hidden[:, -1, :]
        out = self.mlp(_input)
        out = self.l_out(out)
        return out


class DeepSet(torch.nn.Module):
    def __init__(self, dim_in=1, dim_hidden: int = 64):
        super().__init__()
        self.pre_mean = nn.Sequential(nn.Linear(dim_in, dim_hidden), nn.ReLU(), nn.Linear(dim_hidden, dim_hidden))
        self.post_mean = nn.Sequential(nn.Linear(dim_hidden, dim_hidden), nn.ReLU(), nn.Linear(dim_hidden, 1))
        self.l_out = nn.Sigmoid()

    def forward(self, _in):
        x = self.pre_mean(_in)
        x = torch.mean(x, dim=1)
        x = self.post_mean(x)
        x = self.l_out(x)
        return x


class BiLSTM(torch.nn.Module):
    def __init__(self, dim_in=1, dim_hidden: int = 64, num_layers=2):
        super().__init__()
        self.dim_hidden = dim_hidden
        self.pre_lstm = nn.Sequential(nn.Linear(dim_in, dim_hidden), nn.ReLU(), nn.Linear(dim_hidden, dim_hidden))
        self.lstm = nn.LSTM(dim_hidden, dim_hidden, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.post_lstm = nn.Sequential(nn.Linear(dim_hidden, dim_hidden), nn.ReLU(), nn.Linear(dim_hidden, 1))
        self.l_out = nn.Sigmoid()

    def forward(self, _in):
        self.lstm.flatten_parameters()
        x = self.pre_lstm(_in)
        x, _ = self.lstm(x)
        x = x.view(x.shape[0], -1, 2, self.dim_hidden)
        x = torch.mean(x, dim=[1, 2])  # mean over num-layers and sequence-length
        x = self.post_lstm(x)
        x = self.l_out(x)
        return x


class Transformer(torch.nn.Module):
    def __init__(self, dim_in=1, dim_hidden: int = 64, num_layers=2, num_heads=1):
        super().__init__()
        _layer = torch.nn.TransformerEncoderLayer(d_model=dim_in, nhead=num_heads, dim_feedforward=dim_hidden)
        self.model = torch.nn.TransformerEncoder(encoder_layer=_layer, num_layers=num_layers)
        self.post = nn.Sequential(nn.Linear(dim_in, dim_hidden), nn.ReLU(), nn.Linear(dim_hidden, 1))
        self.l_out = nn.Sigmoid()

    def forward(self, _in):
        """ _in: batch x seq x dim """
        _in = _in.permute((1, 0, 2))  # seq x batch x dim
        x = self.model(_in)  # seq x batch x dim
        x = torch.mean(x, dim=0)  # batch x dim
        x = self.post(x)  # batch x dim_out
        x = self.l_out(x)
        return x
