import random
import typing as t

import torch

from kaggle_toolbox.tabular.activations import entmax15


BinningAct = t.Callable[[torch.Tensor], torch.Tensor]
FeatureMaskFn = t.Callable[[torch.Tensor], torch.Tensor]


class NeuralDecisionStump(torch.nn.Module):

    def __init__(
            self,
            n_features: int,
            binning_activation: BinningAct = entmax15,
            feature_mask_function: FeatureMaskFn = entmax15,):
        super().__init__()
        self._num_cutpoints = 1
        self._num_leaf = 2
        self.n_features = n_features
        self.binning_activation = binning_activation
        self.feature_mask_function = feature_mask_function

        # sampling a random beta distribution
        # random distribution helps with diversity in trees and feature splits
        alpha = random.uniform(0.5, 10.0)
        beta = random.uniform(0.5, 10.0)
        # with torch.no_grad():
        feature_mask = torch.distributions.Beta(
                torch.tensor([alpha]), torch.tensor([beta]))\
            .sample(torch.Size([self.n_features,]))\
            .squeeze(-1)
        self.feature_mask = torch.nn.parameter.Parameter(
            feature_mask, requires_grad=True)
        w = torch.linspace(
            1.0,
            self._num_cutpoints + 1.0,
            self._num_cutpoints + 1,
            requires_grad=False,
        ).reshape(1, 1, -1)
        self.register_buffer("W", w)

        cutpoints = torch.rand([self.n_features, self._num_cutpoints])
        # Append zeros to the beginning of each row
        cutpoints = torch.cat([
            torch.zeros([self.n_features, 1], device=cutpoints.device),
            cutpoints
        ], 1)
        self.cut_points = torch.nn.parameter.Parameter(
            cutpoints, requires_grad=True)
        self.leaf_responses = torch.nn.parameter.Parameter(
            torch.rand(self.n_features, self._num_leaf), requires_grad=True)

    def forward(self, x: torch.Tensor) -> t.Tuple[torch.Tensor, torch.Tensor]:
        feature_mask = self.feature_mask_function(self.feature_mask)
        # Repeat W for each batch size using broadcasting
        W = torch.ones(x.size(0), 1, 1, device=x.device) * self.W
        # Binning features
        x = torch.bmm(x.unsqueeze(-1), W) - self.cut_points.unsqueeze(0)
        x = self.binning_activation(x)  # , dim=-1)
        x = x * self.leaf_responses.unsqueeze(0)
        x = (x * feature_mask.reshape(1, -1, 1)).sum(dim=1)
        return x, feature_mask


class NeuralDecisionTree(torch.nn.Module):

    def __init__(
            self,
            depth: int,
            n_features: int,
            dropout: float = 0,
            binning_activation: BinningAct = entmax15,
            feature_mask_function: FeatureMaskFn = entmax15,):
        super().__init__()
        self.depth = depth
        self._num_cutpoints = 1
        self.n_features = n_features
        self._dropout = dropout
        self.binning_activation = binning_activation
        self.feature_mask_function = feature_mask_function

        for d in range(self.depth):
            for n in range(max(2 ** (d), 1)):
                self.add_module(
                    f"decision_stump_{d}_{n}",
                    NeuralDecisionStump(
                        self.n_features + (2 ** (d) if d > 0 else 0),
                        self.binning_activation,
                        self.feature_mask_function,
                    ),
                )
        self.dropout = torch.nn.Dropout(self._dropout)

    def forward(self, x: torch.Tensor) -> t.Tuple[torch.Tensor, t.List[t.List[torch.Tensor]]]:
        tree_input = x
        feature_masks: t.List[t.List[torch.Tensor]] = []
        last_level_layer_nodes: t.Optional[torch.Tensor] = None
        for d in range(self.depth):
            layer_node_list = []
            layer_feature_masks = []
            for n in range(max(2 ** (d), 1)):
                decision_stump = self._modules[f"decision_stump_{d}_{n}"]
                assert decision_stump is not None
                leaf_nodes, feature_mask = decision_stump(tree_input)
                layer_node_list.append(leaf_nodes)
                layer_feature_masks.append(feature_mask)
            layer_node_tensor = torch.cat(layer_node_list, dim=1)
            tree_input = torch.cat([x, layer_node_tensor], dim=1)
            feature_masks.append(layer_feature_masks)
            last_level_layer_nodes = layer_node_tensor
        assert last_level_layer_nodes is not None
        return self.dropout(last_level_layer_nodes), feature_masks


class GFLU(torch.nn.Module):

    def __init__(
            self,
            n_features_in: int,
            n_stages: int,
            feature_mask_function: FeatureMaskFn = entmax15,
            dropout: float = 0.0,):
        super().__init__()
        self.n_features_in = n_features_in
        self.n_features_out = n_features_in
        self.feature_mask_function = feature_mask_function
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for d in range(self.n_stages):
            feature = self.feature_mask_function(self.feature_masks[d]) * x
            h_in = self.w_in[d](torch.cat([feature, h], dim=-1))
            z = torch.sigmoid(h_in[:, : self.n_features_in])
            r = torch.sigmoid(h_in[:, self.n_features_in :])  # noqa: E203
            h_out = torch.tanh(self.w_out[d](torch.cat([r * h, x], dim=-1)))
            h = self.dropout((1 - z) * h + z * h_out)
        return h
