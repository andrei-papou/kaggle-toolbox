import typing as t

import torch

from kaggle_toolbox.data import LabeledDatasetItem
from kaggle_toolbox.tabular.activations import entmoid15

from .components import GFLU, NeuralDecisionTree, BinningAct, FeatureMaskFn


class GATEBackbone(torch.nn.Module):

    def __init__(
            self,
            input_dim: int,
            num_gflu_stages: int,
            num_trees: int,
            tree_depth: int,
            chain_trees: bool = True,
            tree_wise_attention: bool = False,
            tree_wise_attention_dropout: float = 0.0,
            gflu_dropout: float = 0.0,
            tree_dropout: float = 0.0,
            binning_activation: BinningAct = entmoid15,
            feature_mask_function: FeatureMaskFn = torch.nn.functional.softmax,):
        super().__init__()

        self.num_trees = num_trees
        self.tree_depth = tree_depth
        self.chain_trees = chain_trees
        self.tree_wise_attention = tree_wise_attention
        self.tree_wise_attention_dropout = tree_wise_attention_dropout
        self.gflu_dropout = gflu_dropout
        self.tree_dropout = tree_dropout
        self.binning_activation = binning_activation
        self.feature_mask_function = feature_mask_function
        self.n_features = input_dim
        self.output_dim = 2**self.tree_depth

        self.gflus = GFLU(
            n_features_in=self.n_features,
            n_stages=num_gflu_stages,
            feature_mask_function=self.feature_mask_function,
            dropout=self.gflu_dropout,
        ) if num_gflu_stages > 0 else None
        self.trees = torch.nn.ModuleList(
            [
                NeuralDecisionTree(
                    depth=self.tree_depth,
                    n_features=self.n_features + 2**self.tree_depth * t if self.chain_trees else self.n_features,
                    dropout=self.tree_dropout,
                    binning_activation=self.binning_activation,
                    feature_mask_function=self.feature_mask_function,
                )
                for t in range(self.num_trees)
            ]
        )
        if self.tree_wise_attention:
            self.tree_attention = torch.nn.MultiheadAttention(
                embed_dim=self.output_dim,
                num_heads=1,
                batch_first=False,
                dropout=self.tree_wise_attention_dropout,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.gflus is not None:
            x = self.gflus(x)
        # Decision Tree
        tree_output_list = []
        tree_feature_masks = []  # TODO make this optional and create feat importance
        tree_input = x
        for i in range(self.num_trees):
            tree_output, feat_masks = self.trees[i](tree_input)
            tree_output_list.append(tree_output.unsqueeze(-1))
            tree_feature_masks.append(feat_masks)
            if self.chain_trees:
                tree_input = torch.cat([tree_input, tree_output], 1)
        tree_output_tensor = torch.cat(tree_output_list, dim=-1)
        if self.tree_wise_attention:
            tree_output_tensor = tree_output_tensor.permute(2, 0, 1)
            tree_output_tensor, _ = self.tree_attention(tree_output_tensor, tree_output_tensor, tree_output_tensor)
            tree_output_tensor = tree_output_tensor.permute(1, 2, 0)
        return tree_output_tensor

    @property
    def feature_importance_(self) -> t.Optional[torch.Tensor]:
        return self.gflus.feature_mask_function(self.gflus.feature_masks)\
            .sum(dim=0).detach().cpu() if self.gflus is not None else None


class GATEHead(torch.nn.Module):
    """
    Custom Head for GATE.
    """

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            num_trees: int,
            is_regression: bool,
            share_head_weights: bool = True,):
        super().__init__()
        self.input_dim = input_dim
        if share_head_weights:
            self.head = self._build_head(input_dim, output_dim)
        else:
            self.head = torch.nn.ModuleList([
                self._build_head(input_dim, output_dim)
                for _ in range(num_trees)
            ])
        # random parameter with num_trees elements
        self.eta = torch.nn.parameter.Parameter(torch.rand(num_trees, requires_grad=True))
        self.t0 = torch.nn.parameter.Parameter(torch.rand(output_dim), requires_grad=True) \
            if is_regression \
            else torch.nn.parameter.Parameter(torch.zeros(torch.Size([output_dim])), requires_grad=False)

        self._is_regression = is_regression

    @classmethod
    def _build_head(cls, in_dim: int, out_dim: int) -> torch.nn.Module:
        return torch.nn.Linear(in_dim, out_dim)

    def init_from_data(self, data_loader: t.Iterable[LabeledDatasetItem[t.Any]]):
        if self._is_regression:
            batch = next(iter(data_loader))
            self.t0.data = torch.mean(batch.y, dim=0)

    def forward(self, backbone_features: torch.Tensor) -> torch.Tensor:
        # B x L x T
        if isinstance(self.head, torch.nn.ModuleList):
            # B x T X Output
            y_hat = torch.cat(
                [h(backbone_features[:, :, i]).unsqueeze(1) for i, h in enumerate(self.head)],
                dim=1,
            )
        else:
            # https://discuss.pytorch.org/t/how-to-pass-a-3d-tensor-to-linear-layer/908/6
            # B x T x L -> B x T x Output
            y_hat = self.head(backbone_features.transpose(2, 1))

        # applying weights to each tree and summing up
        # ETA
        y_hat = y_hat * self.eta.reshape(1, -1, 1)
        # summing up
        y_hat = y_hat.sum(dim=1)

        if self._is_regression:
            y_hat = y_hat + self.t0
        return y_hat
