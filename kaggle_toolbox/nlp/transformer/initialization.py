import torch


def standard_init_linear(module: torch.nn.Linear, mean: float = 0.0, std: float = 0.02) -> torch.nn.Linear:
    module.weight.data.normal_(mean=mean, std=std)
    if module.bias is not None:
        module.bias.data.zero_()
    return module


def standard_init_layer_norm(module: torch.nn.LayerNorm) -> torch.nn.LayerNorm:
    module.bias.data.zero_()
    module.weight.data.fill_(1.0)
    return module
