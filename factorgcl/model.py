"""
The FactorGCL model of Duan, Wang and Li (2025), "FactorGCL: A Hypergraph-Based Factor
Model with Temporal Residual Contrastive Learning for Stock Returns Prediction" (AAAI-25).

The model is a *cascading residual hypergraph* architecture (Figure 3 of the article).
For one trading day it takes the cross-section of ``N`` stocks -- their raw sequential
market data ``x`` of shape ``(N, T, D)`` and their prior factor exposures ``beta`` of shape
``(N, K)`` -- and decomposes the predicted returns into three components:

- **prior beta** ``e_p``: the influence of the ``K`` human-designed prior factors, obtained
  by running a hypergraph convolution (`HyperGCNLayer`) over the hypergraph whose nodes are
  stocks and whose hyperedges are the prior factors (Eq. 5);
- **hidden beta** ``e_h``: the influence of ``M`` data-driven hidden factors, mined from the
  residual ``e_r = e_s - e_p`` as a hyperedge generation task (Eq. 6);
- **individual alpha** ``e_alpha``: the stock-specific information left in the residual
  ``e_s - e_p - e_h`` (Eq. 7).

The prediction is a linear map of the three embeddings, summed, over ``L`` forward periods
(Eq. 8) -- the "multi-label prediction" of the article, with ``L = 4`` for the horizons
``dt = 1, 5, 10, 20`` used in the experiments.

Training adds a second branch over *future* data ``x'`` of shape ``(N, T', D)`` that reuses
the prior and hidden exposures extracted from the historical branch to produce the future
alpha embedding of Eq. 10; `loss.info_nce_loss` contrasts the two alpha embeddings.

Every module works on one cross-section at a time, which is what the hypergraph structure
asks for: the incidence matrix is rebuilt for every trading day, and the number of stocks
``N`` may differ from day to day.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# defaults of the supplementary material ("Implementation Details")
DEFAULT_HIDDEN_SIZE = 32       # H, the dimension of the feature embeddings
DEFAULT_NUM_HIDDEN_FACTORS = 32  # M, the number of hidden factors
DEFAULT_NUM_RNN_LAYERS = 2     # the number of GRU layers of the feature extractor
DEFAULT_LEAKY_SLOPE = 0.01     # torch's default negative slope for LeakyReLU
# the four forward prediction periods (Delta t) the article predicts jointly
DEFAULT_PREDICT_PERIODS = (1, 5, 10, 20)
# guards the degree matrices of the hypergraph convolution against isolated nodes and
# hyperedges, which have degree zero and would otherwise divide by zero
DEGREE_EPS = 1e-8

BN_POSITIONS = ("input", "output", "both", "none")


class HyperGCNLayer(nn.Module):
    """One hypergraph convolution layer (Eq. 1 of the article, after Feng et al. 2019).

        e' = sigma( Dn^{-1/2} H W De^{-1} H^T Dn^{-1/2} e w )

    ``H`` is the incidence matrix of shape ``(N, E)``, ``W`` the diagonal matrix of
    hyperedge weights (the identity in the article), ``Dn`` and ``De`` the diagonal degree
    matrices of the nodes and the hyperedges, and ``w`` a learnable weight matrix. The
    article reads the product as the three steps of Figure 4, and the implementation below
    follows them one by one instead of forming the ``N x N`` propagation matrix:

    - *message extraction*: ``e w``, a linear map of the node features;
    - *message aggregation*: ``H^T`` collects the (degree-normalised) node messages onto the
      hyperedge they belong to, and ``De^{-1}`` averages them;
    - *message sharing*: ``H`` sends the hyperedge message back to its nodes.

    The incidence matrix may be "soft": the hidden beta module feeds in exposures in
    ``[0, 1]`` rather than a 0/1 membership, and the degrees are then the row and column
    sums of those weights.
    """

    def __init__(self, in_features, out_features, bias=True, negative_slope=DEFAULT_LEAKY_SLOPE):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.negative_slope = negative_slope

    def forward(self, node_features, incidence, edge_weight=None):
        """``node_features`` is ``(N, d_in)``, ``incidence`` is ``(N, E)``, the result ``(N, d_out)``."""
        if node_features.dim() != 2 or incidence.dim() != 2:
            raise ValueError("HyperGCNLayer works on a single cross-section: pass 2-D tensors")
        if node_features.shape[0] != incidence.shape[0]:
            raise ValueError("node_features and incidence must agree on the number of stocks")
        n_edges = incidence.shape[1]
        if edge_weight is None:
            # W = I in the article
            edge_weight = incidence.new_ones(n_edges)
        elif edge_weight.shape != (n_edges,):
            raise ValueError("edge_weight must hold one weight per hyperedge")

        # degree matrices: Dn(i) = sum_j H(i,j) W(j), De(j) = sum_i H(i,j)
        node_degree = (incidence * edge_weight.unsqueeze(0)).sum(dim=1)
        edge_degree = incidence.sum(dim=0)
        dn_inv_sqrt = node_degree.clamp(min=DEGREE_EPS).pow(-0.5).unsqueeze(1)   # (N, 1)
        de_inv = edge_degree.clamp(min=DEGREE_EPS).reciprocal().unsqueeze(1)     # (E, 1)

        message = self.linear(node_features)                     # message extraction: e w
        message = dn_inv_sqrt * message                          # Dn^{-1/2}
        edge_message = incidence.transpose(0, 1) @ message       # H^T: aggregation
        edge_message = de_inv * edge_message                     # De^{-1}
        edge_message = edge_weight.unsqueeze(1) * edge_message   # W
        out = incidence @ edge_message                           # H: sharing
        out = dn_inv_sqrt * out                                  # Dn^{-1/2}
        return F.leaky_relu(out, self.negative_slope)


class FeatureExtractor(nn.Module):
    """The stock feature extractor ``phi_feat`` -- "a gated recurrent unit with a batch
    normalization ..., using the hidden state at the last time step as the stock feature
    embeddings".

    The article does not say where the batch normalisation sits; ``bn_position`` covers the
    readings. The default normalises the ``D`` input channels over the flattened
    stock-timestep axis, which is the usual arrangement for price-volume sequences, and
    keeps the GRU output untouched.
    """

    def __init__(self, input_size, hidden_size=DEFAULT_HIDDEN_SIZE,
                 num_layers=DEFAULT_NUM_RNN_LAYERS, dropout=0.0, bn_position="input"):
        super().__init__()
        if bn_position not in BN_POSITIONS:
            raise ValueError(f"bn_position must be one of {BN_POSITIONS}, got {bn_position!r}")
        self.bn_position = bn_position
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.input_bn = nn.BatchNorm1d(input_size) if bn_position in ("input", "both") else None
        self.output_bn = nn.BatchNorm1d(hidden_size) if bn_position in ("output", "both") else None
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0)

    def forward(self, x):
        """``x`` is ``(N, T, D)``; the result is the last hidden state, ``(N, H)``."""
        if x.dim() != 3:
            raise ValueError("the feature extractor expects (N, T, D) sequential data")
        if self.input_bn is not None:
            n, t, d = x.shape
            x = self.input_bn(x.reshape(n * t, d)).reshape(n, t, d)
        _, h_n = self.gru(x)
        out = h_n[-1]                       # the hidden state at the last time step
        if self.output_bn is not None:
            out = self.output_bn(out)
        return out


class PriorBetaModule(nn.Module):
    """The prior beta module ``phi_prior`` (Eq. 5): a hypergraph convolution over the
    hypergraph ``G_p`` whose node features are the stock embeddings and whose incidence
    matrix is the given prior factor exposure matrix ``beta``."""

    def __init__(self, hidden_size=DEFAULT_HIDDEN_SIZE, negative_slope=DEFAULT_LEAKY_SLOPE):
        super().__init__()
        self.conv = HyperGCNLayer(hidden_size, hidden_size, negative_slope=negative_slope)

    def forward(self, stock_embeddings, beta):
        return self.conv(stock_embeddings, beta)


class HiddenBetaModule(nn.Module):
    """The hidden beta module ``phi_hidden`` (Eq. 6).

    Mining hidden factors is framed as hyperedge generation: ``M`` learnable *hidden factor
    prototypes* ``c`` are compared with the residual embeddings, and the soft exposures

        beta_h(i, j) = Sigmoid( e_r(i) . c(j)^T )

    become the incidence matrix of a second hypergraph ``G_h`` whose node features are the
    residual embeddings. Exposures live in ``(0, 1)`` rather than ``{0, 1}``, which is what
    the article calls a "soft" hyperedge.
    """

    def __init__(self, hidden_size=DEFAULT_HIDDEN_SIZE,
                 num_hidden_factors=DEFAULT_NUM_HIDDEN_FACTORS,
                 negative_slope=DEFAULT_LEAKY_SLOPE):
        super().__init__()
        self.num_hidden_factors = num_hidden_factors
        self.prototypes = nn.Parameter(torch.empty(num_hidden_factors, hidden_size))
        nn.init.xavier_uniform_(self.prototypes)
        self.conv = HyperGCNLayer(hidden_size, hidden_size, negative_slope=negative_slope)

    def exposures(self, residual_embeddings):
        """The hidden factor exposure matrix ``beta_h``, of shape ``(N, M)``."""
        return torch.sigmoid(residual_embeddings @ self.prototypes.transpose(0, 1))

    def forward(self, residual_embeddings, beta_hidden=None):
        """Returns ``(e_h, beta_h)``; pass ``beta_hidden`` to reuse exposures mined elsewhere,
        which is what the future branch of the contrastive loss does (Eq. 10)."""
        if beta_hidden is None:
            beta_hidden = self.exposures(residual_embeddings)
        return self.conv(residual_embeddings, beta_hidden), beta_hidden


class IndividualAlphaModule(nn.Module):
    """The individual alpha module (Eq. 7): a linear layer with a LeakyReLU activation over
    the residual left after removing the prior and hidden beta embeddings."""

    def __init__(self, hidden_size=DEFAULT_HIDDEN_SIZE, negative_slope=DEFAULT_LEAKY_SLOPE):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.negative_slope = negative_slope

    def forward(self, residual_embeddings):
        return F.leaky_relu(self.linear(residual_embeddings), self.negative_slope)


class ProjectionHead(nn.Module):
    """``p(x)``of Eq. 11: "a 2-layer MLP with LeakyReLU activation functions"."""

    def __init__(self, hidden_size=DEFAULT_HIDDEN_SIZE, out_size=None,
                 negative_slope=DEFAULT_LEAKY_SLOPE):
        super().__init__()
        out_size = hidden_size if out_size is None else out_size
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(negative_slope),
            nn.Linear(hidden_size, out_size),
            nn.LeakyReLU(negative_slope),
        )

    def forward(self, x):
        return self.net(x)


class FactorGCL(nn.Module):
    """The full model.

    ``forward`` runs the historical branch and returns the predictions together with every
    intermediate quantity; ``forward_future`` runs the future branch of Eq. 9-10, which
    shares the prior and hidden beta modules with the historical branch and reuses its
    exposures ``beta`` and ``beta_h``.

    The ablation switches of Table 2 are constructor flags:

    - ``use_prior=False``  -- "-wo Prior";
    - ``use_hidden=False`` -- "-wo Hidden";
    - ``use_alpha=False``  -- "-wo Alpha" (the article pairs it with ``gamma = 0`` since the
      contrastive loss contrasts alpha embeddings: "-wo Alpha&CL").

    ``share_future_feature_extractor`` follows the article's note under Eq. 10, which lists
    only ``phi_prior`` and ``phi_hidden`` as sharing parameters with the historical branch,
    so the future feature extractor is a second GRU by default.
    """

    def __init__(self, input_size, num_prior_factors,
                 hidden_size=DEFAULT_HIDDEN_SIZE,
                 num_hidden_factors=DEFAULT_NUM_HIDDEN_FACTORS,
                 num_rnn_layers=DEFAULT_NUM_RNN_LAYERS,
                 num_periods=len(DEFAULT_PREDICT_PERIODS),
                 dropout=0.0, bn_position="input",
                 negative_slope=DEFAULT_LEAKY_SLOPE,
                 use_prior=True, use_hidden=True, use_alpha=True,
                 share_future_feature_extractor=False,
                 future_alpha_module=False,
                 projection_size=None):
        super().__init__()
        if not (use_prior or use_hidden or use_alpha):
            raise ValueError("at least one of the prior, hidden and alpha modules must be kept")
        self.input_size = input_size
        self.num_prior_factors = num_prior_factors
        self.hidden_size = hidden_size
        self.num_hidden_factors = num_hidden_factors
        self.num_periods = num_periods
        self.use_prior = use_prior
        self.use_hidden = use_hidden
        self.use_alpha = use_alpha
        self.share_future_feature_extractor = share_future_feature_extractor
        self.future_alpha_module = future_alpha_module

        self.feature_extractor = FeatureExtractor(
            input_size, hidden_size, num_layers=num_rnn_layers,
            dropout=dropout, bn_position=bn_position)
        self.future_feature_extractor = self.feature_extractor if share_future_feature_extractor \
            else FeatureExtractor(input_size, hidden_size, num_layers=num_rnn_layers,
                                  dropout=dropout, bn_position=bn_position)

        self.prior_beta = PriorBetaModule(hidden_size, negative_slope) if use_prior else None
        self.hidden_beta = HiddenBetaModule(hidden_size, num_hidden_factors, negative_slope) \
            if use_hidden else None
        self.individual_alpha = IndividualAlphaModule(hidden_size, negative_slope) \
            if use_alpha else None

        # Eq. 8: one linear map per component, a single shared bias per forward period
        self.out_prior = nn.Linear(hidden_size, num_periods, bias=False) if use_prior else None
        self.out_hidden = nn.Linear(hidden_size, num_periods, bias=False) if use_hidden else None
        self.out_alpha = nn.Linear(hidden_size, num_periods, bias=False) if use_alpha else None
        self.out_bias = nn.Parameter(torch.zeros(num_periods))

        self.projection = ProjectionHead(hidden_size, projection_size, negative_slope)

    def _components(self, stock_embeddings, beta, beta_hidden=None, apply_alpha_module=True):
        """The cascade of Eq. 5-7 shared by the historical and future branches."""
        residual = stock_embeddings
        prior = None
        if self.use_prior:
            if beta is None:
                raise ValueError("prior factor exposures are required when use_prior is True")
            prior = self.prior_beta(stock_embeddings, beta)
            residual = residual - prior
        hidden = None
        if self.use_hidden:
            hidden, beta_hidden = self.hidden_beta(residual, beta_hidden)
            residual = residual - hidden
        alpha = None
        if self.use_alpha:
            alpha = self.individual_alpha(residual) if apply_alpha_module else residual
        return prior, hidden, alpha, beta_hidden, residual

    def forward(self, x, beta):
        """The historical branch.

        ``x`` is ``(N, T, D)`` and ``beta`` is ``(N, K)``; returns a dict with the prediction
        ``y_hat`` of shape ``(N, L)`` and the embeddings the contrastive loss needs.
        """
        stock_embeddings = self.feature_extractor(x)
        prior, hidden, alpha, beta_hidden, residual = self._components(stock_embeddings, beta)

        y_hat = self.out_bias.unsqueeze(0).expand(stock_embeddings.shape[0], -1)
        if self.use_prior:
            y_hat = y_hat + self.out_prior(prior)
        if self.use_hidden:
            y_hat = y_hat + self.out_hidden(hidden)
        if self.use_alpha:
            y_hat = y_hat + self.out_alpha(alpha)
        return {"y_hat": y_hat, "stock_embeddings": stock_embeddings, "prior_beta": prior,
                "hidden_beta": hidden, "alpha": alpha, "beta_hidden": beta_hidden,
                "residual": residual}

    def forward_future(self, x_future, beta, beta_hidden):
        """The future branch of Eq. 9-10.

        The prior and hidden beta modules are the historical ones (shared parameters) and the
        exposures ``beta`` and ``beta_hidden`` are those mined from the historical data, so
        only the sequence encoder sees the future window. Eq. 10 writes the future alpha as
        the bare residual ``e'_s - phi'_prior - phi'_hidden``; set ``future_alpha_module=True``
        on the model to push it through the alpha module of Eq. 7 as well.
        """
        stock_embeddings = self.future_feature_extractor(x_future)
        _, _, alpha, _, _ = self._components(stock_embeddings, beta, beta_hidden=beta_hidden,
                                             apply_alpha_module=self.future_alpha_module)
        return {"stock_embeddings": stock_embeddings, "alpha": alpha}
