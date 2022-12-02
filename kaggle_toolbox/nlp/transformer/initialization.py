import torch


def standard_init_linear(
        module: torch.nn.Linear,
        mean: float = 0.0,
        std: float = 0.02) -> torch.nn.Linear:
    module.weight.data.normal_(mean=mean, std=std)
    if module.bias is not None:
        module.bias.data.zero_()
    return module


def standard_init_layer_norm(module: torch.nn.LayerNorm) -> torch.nn.LayerNorm:
    module.bias.data.zero_()
    module.weight.data.fill_(1.0)
    return module


def standard_init_embedding(
        module: torch.nn.Embedding,
        mean: float = 0.0,
        std: float = 0.02) -> torch.nn.Embedding:
    module.weight.data.normal_(mean=mean, std=std)
    if module.padding_idx is not None:
        module.weight.data[module.padding_idx].zero_()
    return module


def standard_init_module(
        module: torch.nn.Module,
        mean: float = 0.0,
        std: float = 0.02) -> torch.nn.Module:
    for child_module in module.modules():
        if isinstance(child_module, torch.nn.Linear):
            standard_init_linear(child_module, mean=mean, std=std)
        if isinstance(child_module, torch.nn.LayerNorm):
            standard_init_layer_norm(child_module)
        if isinstance(child_module, torch.nn.Embedding):
            standard_init_embedding(child_module, mean=mean, std=std)
    return module
