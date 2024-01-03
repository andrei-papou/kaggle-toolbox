import torch

from kaggle_toolbox.tabular.activations import t_softmax, TSoftmax
from kaggle_toolbox.tabular.layers import GFLU as BaseGFLU


class GFLU(BaseGFLU):

    def __init__(
            self,
            n_features_in: int,
            n_stages: int,
            feature_sparsity: float = 0.3,
            learnable_sparsity: bool = True,
            dropout: float = 0.0,):
        super().__init__(
            n_features_in=n_features_in,
            n_stages=n_stages,
            dropout=dropout)

        self._t = torch.nn.parameter.Parameter(
            TSoftmax.t_from_r(self.feature_masks, r=torch.tensor([feature_sparsity])),
            requires_grad=learnable_sparsity)

    def _mask_feat_for_stage(self, x: torch.Tensor, d: int) -> torch.Tensor:
        return t_softmax(self.feature_masks[d], torch.relu(self._t[d])) * x
