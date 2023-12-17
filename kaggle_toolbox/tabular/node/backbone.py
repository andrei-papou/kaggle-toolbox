import typing as t

import torch

from kaggle_toolbox.init import Initializer
from .activations import sparsemax, sparsemoid
from .odst import ODST, ChoiceFunction, BinFunction


class NODEBackbone(torch.nn.Sequential):

    def __init__(
            self,
            input_dim: int,
            num_trees: int,
            num_layers: int,
            depth: int = 6,
            tree_output_dim: int = 1,
            max_features: t.Optional[int] = None,
            input_dropout: float = 0.0,
            flatten_output: bool = False,
            choice_function: ChoiceFunction = sparsemax,
            bin_function: BinFunction = sparsemoid,
            initialize_response_: Initializer = torch.nn.init.normal_,
            initialize_selection_logits_: Initializer = torch.nn.init.uniform_,
            threshold_init_beta: float = 1.0,
            threshold_init_cutoff: float = 1.0,):
        layers = []
        for _ in range(num_layers):
            oddt = ODST(
                input_dim,
                num_trees,
                depth=depth,
                tree_output_dim=tree_output_dim,
                flatten_output=True,
                choice_function=choice_function,
                bin_function=bin_function,
                initialize_response_=initialize_response_,
                initialize_selection_logits_=initialize_selection_logits_,
                threshold_init_beta=threshold_init_beta,
                threshold_init_cutoff=threshold_init_cutoff,)
            input_dim = min(
                input_dim + num_trees * tree_output_dim,
                max_features or int(float("inf")))
            layers.append(oddt)

        super().__init__(*layers)
        self.num_layers, self.layer_dim, self.tree_dim = (
            num_layers,
            num_trees,
            tree_output_dim,
        )
        self.max_features, self.flatten_output = max_features, flatten_output
        self.input_dropout = input_dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        initial_features = x.shape[-1]
        for layer in self:
            layer_inp = x
            if self.max_features is not None:
                tail_features = min(self.max_features, layer_inp.shape[-1]) - initial_features
                if tail_features != 0:
                    layer_inp = torch.cat(
                        [
                            layer_inp[..., :initial_features],
                            layer_inp[..., -tail_features:],
                        ],
                        dim=-1,
                    )
            if self.training and self.input_dropout:
                layer_inp = torch.nn.functional.dropout(layer_inp, self.input_dropout)
            h = layer(layer_inp)
            x = torch.cat([x, h], dim=-1)

        outputs = x[..., initial_features:]
        if not self.flatten_output:
            outputs = outputs.view(*outputs.shape[:-1], self.num_layers * self.layer_dim, self.tree_dim)
        return outputs
