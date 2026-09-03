"""
The simple baselines of Table 1, so that a comparison can be run with the same data
pipeline, the same multi-period prediction head and the same training loop as FactorGCL.

Covered are the baselines the supplement describes in a sentence and that need nothing
beyond the price-volume panel: MLP, GRU, TCN, Transformer, ALSTM and a plain HyperGCN over
the prior factor hypergraph. The remaining baselines of Table 1 (SFM, GAT, HIST, STHAN-SR,
FactorVAE, CI-STHPAN) are papers of their own and are not reimplemented here.

Every baseline exposes the interface `train.train_model` expects of `model.FactorGCL`: it
takes ``(x, beta)`` and returns a dict with a ``y_hat`` of shape ``(N, L)``. None of them
has an alpha embedding, so ``use_alpha`` is False and the contrastive term must be switched
off (``gamma = 0``) when training them.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import (DEFAULT_HIDDEN_SIZE, DEFAULT_LEAKY_SLOPE, DEFAULT_NUM_RNN_LAYERS,
                   FeatureExtractor, HyperGCNLayer)


class _Baseline(nn.Module):
    """Shared plumbing: a sequence encoder plus the linear multi-period prediction head."""

    use_prior = False
    use_hidden = False
    use_alpha = False
    projection = None

    def __init__(self, hidden_size, num_periods):
        super().__init__()
        self.head = nn.Linear(hidden_size, num_periods)

    def encode(self, x, beta):
        raise NotImplementedError

    def forward(self, x, beta=None):
        embeddings = self.encode(x, beta)
        return {"y_hat": self.head(embeddings), "stock_embeddings": embeddings,
                "prior_beta": None, "hidden_beta": None, "alpha": None, "beta_hidden": None,
                "residual": embeddings}


class MLPModel(_Baseline):
    """"A multi-layer perceptron model with a linear prediction layer", over the flattened
    sequence."""

    def __init__(self, input_size, seq_len, hidden_size=DEFAULT_HIDDEN_SIZE, num_periods=4,
                 num_layers=2, dropout=0.0):
        super().__init__(hidden_size, num_periods)
        layers, in_features = [], input_size * seq_len
        for _ in range(num_layers):
            layers += [nn.Linear(in_features, hidden_size), nn.LeakyReLU(DEFAULT_LEAKY_SLOPE),
                       nn.Dropout(dropout)]
            in_features = hidden_size
        self.net = nn.Sequential(*layers)

    def encode(self, x, beta=None):
        return self.net(x.reshape(x.shape[0], -1))


class GRUModel(_Baseline):
    """"A gated recurrent unit layer followed by a linear prediction layer"."""

    def __init__(self, input_size, hidden_size=DEFAULT_HIDDEN_SIZE, num_periods=4,
                 num_layers=DEFAULT_NUM_RNN_LAYERS, dropout=0.0, bn_position="input"):
        super().__init__(hidden_size, num_periods)
        self.encoder = FeatureExtractor(input_size, hidden_size, num_layers, dropout, bn_position)

    def encode(self, x, beta=None):
        return self.encoder(x)


class _Chomp(nn.Module):
    """Drops the padding a causal convolution adds on the right."""

    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, x):
        return x[:, :, :-self.size] if self.size > 0 else x


class TCNModel(_Baseline):
    """A temporal convolutional network: stacked causal dilated convolutions with residual
    connections (Bai, Kolter and Koltun 2018)."""

    def __init__(self, input_size, hidden_size=DEFAULT_HIDDEN_SIZE, num_periods=4,
                 num_layers=3, kernel_size=3, dropout=0.0):
        super().__init__(hidden_size, num_periods)
        blocks, in_channels = [], input_size
        for level in range(num_layers):
            dilation = 2 ** level
            padding = (kernel_size - 1) * dilation
            blocks.append(nn.Sequential(
                nn.Conv1d(in_channels, hidden_size, kernel_size, padding=padding,
                          dilation=dilation),
                _Chomp(padding), nn.LeakyReLU(DEFAULT_LEAKY_SLOPE), nn.Dropout(dropout)))
            in_channels = hidden_size
        self.blocks = nn.ModuleList(blocks)
        self.residuals = nn.ModuleList([
            nn.Conv1d(input_size if i == 0 else hidden_size, hidden_size, 1)
            for i in range(num_layers)])

    def encode(self, x, beta=None):
        h = x.transpose(1, 2)                       # (N, D, T)
        for block, residual in zip(self.blocks, self.residuals):
            h = block(h) + residual(h)
        return h[:, :, -1]                          # the last time step


class TransformerModel(_Baseline):
    """A time-series Transformer encoder with a sinusoidal positional encoding, reading out
    the last time step."""

    def __init__(self, input_size, hidden_size=DEFAULT_HIDDEN_SIZE, num_periods=4,
                 num_layers=2, num_heads=4, dropout=0.0, max_len=512):
        super().__init__(hidden_size, num_periods)
        self.input_projection = nn.Linear(input_size, hidden_size)
        position = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, hidden_size, 2).float()
                        * (-torch.log(torch.tensor(10000.0)) / hidden_size))
        encoding = torch.zeros(max_len, hidden_size)
        encoding[:, 0::2] = torch.sin(position * div)
        encoding[:, 1::2] = torch.cos(position * div)[:, :encoding[:, 1::2].shape[1]]
        self.register_buffer("positional_encoding", encoding)
        layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads,
                                           dim_feedforward=4 * hidden_size, dropout=dropout,
                                           batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def encode(self, x, beta=None):
        h = self.input_projection(x) + self.positional_encoding[:x.shape[1]].unsqueeze(0)
        return self.encoder(h)[:, -1, :]


class ALSTMModel(_Baseline):
    """An attention-based LSTM: "a temporal attention aggregation layer built upon a standard
    LSTM", whose context vector is concatenated with the last hidden state."""

    def __init__(self, input_size, hidden_size=DEFAULT_HIDDEN_SIZE, num_periods=4,
                 num_layers=DEFAULT_NUM_RNN_LAYERS, dropout=0.0):
        super().__init__(hidden_size, num_periods)
        self.input_projection = nn.Sequential(nn.Linear(input_size, hidden_size),
                                              nn.LeakyReLU(DEFAULT_LEAKY_SLOPE))
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.attention = nn.Sequential(nn.Linear(hidden_size, hidden_size // 2),
                                       nn.Tanh(), nn.Linear(hidden_size // 2, 1))
        self.merge = nn.Linear(2 * hidden_size, hidden_size)

    def encode(self, x, beta=None):
        h, _ = self.lstm(self.input_projection(x))
        weights = torch.softmax(self.attention(h).squeeze(-1), dim=1).unsqueeze(-1)
        context = (h * weights).sum(dim=1)
        return F.leaky_relu(self.merge(torch.cat([h[:, -1, :], context], dim=1)),
                            DEFAULT_LEAKY_SLOPE)


class HyperGCNModel(_Baseline):
    """A plain hypergraph convolutional network over the prior factor hypergraph: the same
    sequence encoder as FactorGCL followed by one HyperGCN layer, without the residual
    cascade, the hidden factors or the contrastive loss."""

    use_prior = True

    def __init__(self, input_size, hidden_size=DEFAULT_HIDDEN_SIZE, num_periods=4,
                 num_layers=DEFAULT_NUM_RNN_LAYERS, dropout=0.0, bn_position="input"):
        super().__init__(hidden_size, num_periods)
        self.encoder = FeatureExtractor(input_size, hidden_size, num_layers, dropout, bn_position)
        self.conv = HyperGCNLayer(hidden_size, hidden_size)

    def encode(self, x, beta):
        if beta is None:
            raise ValueError("the HyperGCN baseline needs the prior factor exposures")
        return self.conv(self.encoder(x), beta)


BASELINES = ("mlp", "gru", "tcn", "transformer", "alstm", "hypergcn")


@dataclass
class BaselineConfig:
    """Selects and builds one baseline; drop-in replacement for `train.ModelConfig`."""

    name: str = "gru"
    hidden_size: int = DEFAULT_HIDDEN_SIZE
    num_layers: int = DEFAULT_NUM_RNN_LAYERS
    dropout: float = 0.0
    bn_position: str = "input"
    seq_len: int = 60           # only the MLP needs it, to size its flattened input

    use_alpha = False           # no baseline has an alpha embedding to contrast

    def build(self, input_size, num_prior_factors, num_periods):
        name = self.name.lower()
        if name not in BASELINES:
            raise ValueError(f"unknown baseline {self.name!r}, expected one of {BASELINES}")
        if name == "mlp":
            return MLPModel(input_size, self.seq_len, self.hidden_size, num_periods,
                            dropout=self.dropout)
        if name == "gru":
            return GRUModel(input_size, self.hidden_size, num_periods, self.num_layers,
                            self.dropout, self.bn_position)
        if name == "tcn":
            return TCNModel(input_size, self.hidden_size, num_periods, dropout=self.dropout)
        if name == "transformer":
            return TransformerModel(input_size, self.hidden_size, num_periods,
                                    num_layers=self.num_layers, dropout=self.dropout)
        if name == "alstm":
            return ALSTMModel(input_size, self.hidden_size, num_periods, self.num_layers,
                              self.dropout)
        return HyperGCNModel(input_size, self.hidden_size, num_periods, self.num_layers,
                             self.dropout, self.bn_position)
