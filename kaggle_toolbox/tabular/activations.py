import typing as t

import torch
import torch.nn.functional as F
from torch.autograd import Function

from .utils import make_ix_like


class SparsemaxFunction(Function):
    """An implementation of sparsemax (Martins & Astudillo, 2016). See :cite:`DBLP:journals/corr/MartinsA16` for
    detailed description.

    By Ben Peters and Vlad Niculae
    """

    @staticmethod
    def forward(ctx, input, *args, dim: int = -1, **kwargs):  # type: ignore
        """Sparsemax: normalizing sparse transform (a la softmax)

        Parameters:
            input (Tensor): any shape
            dim (int): dimension along which to apply sparsemax

        Returns:
            output (Tensor): same shape as input
        """
        ctx.dim = dim
        max_val, _ = input.max(dim=dim, keepdim=True)
        input -= max_val  # same numerical stability trick as for softmax
        tau, supp_size = SparsemaxFunction._threshold_and_support(input, dim=dim)
        output = torch.clamp(input - tau, min=0)
        ctx.save_for_backward(supp_size, output)
        return output

    @staticmethod
    def backward(ctx: t.Any, grad_output: torch.Tensor) -> t.Tuple[torch.Tensor, None]:  # type: ignore
        supp_size, output = ctx.saved_tensors
        dim = ctx.dim
        grad_input = grad_output.clone()
        grad_input[output == 0] = 0

        v_hat = grad_input.sum(dim=dim) / supp_size.to(output.dtype).squeeze()
        v_hat = v_hat.unsqueeze(dim)
        grad_input = torch.where(output != 0, grad_input - v_hat, grad_input)
        return grad_input, None

    @staticmethod
    def _threshold_and_support(
            input: torch.Tensor,
            dim: int = -1) -> t.Tuple[torch.Tensor, torch.Tensor]:
        """Sparsemax building block: compute the threshold.

        Args:
            input: any dimension
            dim: dimension along which to apply the sparsemax

        Returns:
            the threshold value
        """

        input_srt, _ = torch.sort(input, descending=True, dim=dim)
        input_cumsum = input_srt.cumsum(dim) - 1
        rhos = make_ix_like(input, dim)
        support = rhos * input_srt > input_cumsum

        support_size = support.sum(dim=dim).unsqueeze(dim)
        tau = input_cumsum.gather(dim, support_size - 1)
        tau /= support_size.to(input.dtype)
        return tau, support_size


class Entmax15Function(Function):
    """An implementation of exact Entmax with alpha=1.5 (B.

    Peters, V. Niculae, A. Martins). See
    :cite:`https://arxiv.org/abs/1905.05702 for detailed description.
    Source: https://github.com/deep-spin/entmax
    """

    @staticmethod
    def forward(ctx, input: torch.Tensor, dim: int = -1) -> torch.Tensor:  # type: ignore
        ctx.dim = dim

        max_val, _ = input.max(dim=dim, keepdim=True)
        input = input - max_val  # same numerical stability trick as for softmax
        input = input / 2  # divide by 2 to solve actual Entmax

        tau_star, _ = Entmax15Function._threshold_and_support(input, dim)
        output = torch.clamp(input - tau_star, min=0) ** 2
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> t.Tuple[torch.Tensor, None]:  # type: ignore
        (Y,) = ctx.saved_tensors
        gppr = Y.sqrt()  # = 1 / g'' (Y)
        dX = grad_output * gppr
        q = dX.sum(ctx.dim) / gppr.sum(ctx.dim)
        q = q.unsqueeze(ctx.dim)
        dX -= q * gppr
        return dX, None

    @staticmethod
    def _threshold_and_support(
            input: torch.Tensor,
            dim: int = -1) -> t.Tuple[torch.Tensor, torch.Tensor]:
        x_srt, _ = torch.sort(input, descending=True, dim=dim)

        rho = make_ix_like(input, dim)
        mean = x_srt.cumsum(dim) / rho
        mean_sq = (x_srt ** 2).cumsum(dim) / rho
        ss = rho * (mean_sq - mean**2)
        delta = (1 - ss) / rho

        # NOTE this is not exactly the same as in reference algo
        # Fortunately it seems the clamped values never wrongly
        # get selected by tau <= sorted_z. Prove this!
        delta_nz = torch.clamp(delta, 0)
        tau = mean - torch.sqrt(delta_nz)

        support_size = (tau <= x_srt).sum(dim).unsqueeze(dim)
        tau_star = tau.gather(dim, support_size - 1)
        return tau_star, support_size


class Entmoid15Function(Function):
    """A highly optimized equivalent of labda x: Entmax15([x, 0])"""

    @staticmethod
    def forward(ctx: t.Any, input: torch.Tensor) -> torch.Tensor:  # type: ignore
        output = Entmoid15Function._forward(input)
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def _forward(input: torch.Tensor) -> torch.Tensor:
        input, is_pos = abs(input), input >= 0
        tau = (input + torch.sqrt(F.relu(8 - input**2))) / 2
        tau.masked_fill_(tau <= input, 2.0)
        y_neg = 0.25 * F.relu(tau - input, inplace=True) ** 2
        return torch.where(is_pos, 1 - y_neg, y_neg)

    @staticmethod
    def backward(ctx: t.Any, grad_output: torch.Tensor) -> torch.Tensor:  # type: ignore
        return Entmoid15Function._backward(ctx.saved_tensors[0], grad_output)

    @staticmethod
    def _backward(output: torch.Tensor, grad_output: torch.Tensor) -> torch.Tensor:
        gppr0, gppr1 = output.sqrt(), (1 - output).sqrt()
        grad_input = grad_output * gppr0
        q = grad_input / (gppr0 + gppr1)
        grad_input -= q * gppr0
        return grad_input


def t_softmax(
        x: torch.Tensor,
        t: t.Optional[torch.Tensor] = None,
        dim: int = -1) -> torch.Tensor:
    if t is None:
        t = torch.tensor(0.5, device=x.device)
    assert (t >= 0.0).all()
    maxes = torch.max(x, dim=dim, keepdim=True).values
    x_minus_maxes = x - maxes

    w = torch.relu(x_minus_maxes + t) + 1e-8
    return torch.softmax(x_minus_maxes + torch.log(w), dim=dim)


class TSoftmax(torch.nn.Module):

    def __init__(self, t: torch.Tensor, learn_t: bool = False, dim: int = -1):
        super().__init__()
        self._t = torch.nn.parameter.Parameter(t, requires_grad=learn_t)
        self._dim = dim

    @classmethod
    def t_from_r(
            cls,
            input: torch.Tensor,
            r: torch.Tensor,
            dim: int = -1,
            eps: float = 1e-8) -> torch.Tensor:
        # r represents what is the fraction of zero values that we want to have
        assert ((0.0 <= r) & (r <= 1.0)).all()

        maxes = torch.max(input, dim=dim, keepdim=True).values
        input_minus_maxes = input - maxes

        zeros_mask = torch.exp(input_minus_maxes) == 0.0
        zeros_frac = zeros_mask.sum(dim=dim, keepdim=True).float() / input_minus_maxes.shape[dim]

        q = torch.clamp((r - zeros_frac) / (1 - zeros_frac), min=0.0, max=1.0)
        x_minus_maxes = input_minus_maxes * (~zeros_mask).float()
        if q.ndim > 1:
            t = -torch.quantile(x_minus_maxes, q.view(-1), dim=dim, keepdim=True).detach()
            t = t.squeeze(dim).diagonal(dim1=-2, dim2=-1).unsqueeze(-1) + eps
        else:
            t = -torch.quantile(x_minus_maxes, q, dim=dim).detach() + eps
        return t

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return t_softmax(input, self._t, self._dim)


def sparsemax(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return SparsemaxFunction.apply(input, dim)


def sparsemoid(input: torch.Tensor) -> torch.Tensor:
    return (0.5 * input + 0.5).clamp_(0, 1)


def entmax15(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return Entmax15Function.apply(input, dim)


def entmoid15(input: torch.Tensor) -> torch.Tensor:
    return Entmoid15Function.apply(input)
