import torch

from kaggle_toolbox.tabular.activations import TSoftmax
from kaggle_toolbox.tabular.gate.components import GFLU as OriginalGFLU


class GFLU(OriginalGFLU):

    def __init__(
            self,
            n_features_in: int,
            n_stages: int,
            feature_sparsity: float = 0.3,
            learnable_sparsity: bool = True,
            dropout: float = 0.0,):
        # Let `feature_mask_function` be initialized by default.
        super().__init__(
            n_features_in=n_features_in,
            n_stages=n_stages,
            dropout=dropout)

        # Override `feature_mask_function` based on `feature_masks`.
        self.feature_mask_function = TSoftmax(
            t=TSoftmax.t_from_r(self.feature_masks, r=torch.tensor([feature_sparsity])),
            learn_t=learnable_sparsity)
