import random

import torch


class ModuleWithInit(torch.nn.Module):
    """Base class for pytorch module with data-aware initializer on first batch."""

    def __init__(self):
        super().__init__()
        self._is_initialized_tensor = torch.nn.parameter.Parameter(
            torch.tensor(0, dtype=torch.uint8), requires_grad=False)
        self._is_initialized_bool = None
        # Note: this module uses a separate flag self._is_initialized so as to achieve both
        # * persistence: is_initialized is saved alongside model in state_dict
        # * speed: model doesn't need to cache
        # please DO NOT use these flags in child modules

    def initialize(self, *args, **kwargs):
        """Initialize module tensors using first batch of data."""
        raise NotImplementedError("Please implement ")

    def __call__(self, *args, **kwargs):
        if self._is_initialized_bool is None:
            self._is_initialized_bool = bool(self._is_initialized_tensor.item())
        if not self._is_initialized_bool:
            self.initialize(*args, **kwargs)
            self._is_initialized_tensor.data[...] = 1  # type: ignore
            self._is_initialized_bool = True
        return super().__call__(*args, **kwargs)


class PositionWiseFeedForward(torch.nn.Module):
    r"""
    title: Position-wise Feed-Forward Network (FFN)
    summary: Documented reusable implementation of the position wise feedforward network.

    # Position-wise Feed-Forward Network (FFN)
    This is a [PyTorch](https://pytorch.org)  implementation
    of position-wise feedforward network used in transformer.
    FFN consists of two fully connected layers.
    Number of dimensions in the hidden layer $d_{ff}$, is generally set to around
    four times that of the token embedding $d_{model}$.
    So it is sometime also called the expand-and-contract network.
    There is an activation at the hidden layer, which is
    usually set to ReLU (Rectified Linear Unit) activation, $$\\max(0, x)$$
    That is, the FFN function is,
    $$FFN(x, W_1, W_2, b_1, b_2) = \\max(0, x W_1 + b_1) W_2 + b_2$$
    where $W_1$, $W_2$, $b_1$ and $b_2$ are learnable parameters.
    Sometimes the
    GELU (Gaussian Error Linear Unit) activation is also used instead of ReLU.
    $$x \\Phi(x)$$ where $\\Phi(x) = P(X \\le x), X \\sim \\mathcal{N}(0,1)$
    ### Gated Linear Units
    This is a generic implementation that supports different variants including
    [Gated Linear Units](https://arxiv.org/abs/2002.05202) (GLU).
    """

    def __init__(
            self,
            d_model: int,
            d_ff: int,
            dropout: float = 0.1,
            activation: torch.nn.Module = torch.nn.ReLU(),
            is_gated: bool = False,
            bias1: bool = True,
            bias2: bool = True,
            bias_gate: bool = True,):
        """
        * `d_model` is the number of features in a token embedding
        * `d_ff` is the number of features in the hidden layer of the FFN
        * `dropout` is dropout probability for the hidden layer
        * `is_gated` specifies whether the hidden layer is gated
        * `bias1` specified whether the first fully connected layer should have a learnable bias
        * `bias2` specified whether the second fully connected layer should have a learnable bias
        * `bias_gate` specified whether the fully connected layer for the gate should have a learnable bias
        """
        super().__init__()
        # Layer one parameterized by weight $W_1$ and bias $b_1$
        self.layer1 = torch.nn.Linear(d_model, d_ff, bias=bias1)
        # Layer one parameterized by weight $W_1$ and bias $b_1$
        self.layer2 = torch.nn.Linear(d_ff, d_model, bias=bias2)
        # Hidden layer dropout
        self.dropout = torch.nn.Dropout(dropout)
        # Activation function $f$
        self.activation = activation
        # Whether there is a gate
        self.is_gated = is_gated
        if is_gated:
            # If there is a gate the linear layer to transform inputs to
            # be multiplied by the gate, parameterized by weight $V$ and bias $c$
            self.linear_v = torch.nn.Linear(d_model, d_ff, bias=bias_gate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # $f(x W_1 + b_1)$
        g = self.activation(self.layer1(x))
        # If gated, $f(x W_1 + b_1) \otimes (x V + b) $
        if self.is_gated:
            x = g * self.linear_v(x)
        # Otherwise
        else:
            x = g
        # Apply dropout
        x = self.dropout(x)
        # $(f(x W_1 + b_1) \otimes (x V + b)) W_2 + b_2$ or $f(x W_1 + b_1) W_2 + b_2$
        # depending on whether it is gated
        return self.layer2(x)


class GFLU(torch.nn.Module):

    def __init__(
            self,
            n_features_in: int,
            n_stages: int,
            dropout: float = 0.0,):
        super().__init__()
        self.n_features_in = n_features_in
        self.n_features_out = n_features_in
        self._dropout = dropout
        self.n_stages = n_stages

        self.w_in = torch.nn.ModuleList([
            torch.nn.Linear(2 * self.n_features_in, 2 * self.n_features_in)
            for _ in range(self.n_stages)
        ])
        self.w_out = torch.nn.ModuleList([
            torch.nn.Linear(2 * self.n_features_in, self.n_features_in)
            for _ in range(self.n_stages)
        ])

        self.feature_masks = torch.nn.parameter.Parameter(
            torch.cat(
                [
                    torch.distributions.Beta(
                        torch.tensor([random.uniform(0.5, 10.0)]),
                        torch.tensor([random.uniform(0.5, 10.0)]),
                    )
                    .sample(torch.Size([self.n_features_in,]))
                    .squeeze(-1)
                    for _ in range(self.n_stages)
                ]
            ).reshape(self.n_stages, self.n_features_in),
            requires_grad=True)
        self.dropout = torch.nn.Dropout(self._dropout)

    def _mask_feat_for_stage(self, x: torch.Tensor, d: int) -> torch.Tensor:
        raise NotImplementedError()

    def _forward_stage(self, x: torch.Tensor, h: torch.Tensor, d: int) -> torch.Tensor:
        feature = self._mask_feat_for_stage(x, d)
        h_in = self.w_in[d](torch.cat([feature, h], dim=-1))
        z = torch.sigmoid(h_in[:, : self.n_features_in])
        r = torch.sigmoid(h_in[:, self.n_features_in :])  # noqa: E203
        h_out = torch.tanh(self.w_out[d](torch.cat([r * h, x], dim=-1)))
        h = self.dropout((1 - z) * h + z * h_out)
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for d in range(self.n_stages):
            h = self._forward_stage(x, h, d)
        return h
