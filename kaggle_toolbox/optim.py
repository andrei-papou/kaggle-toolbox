import typing as t

import torch


def freeze_params(
        named_param_it: t.Iterable[t.Tuple[str, torch.nn.parameter.Parameter]],
        by_prefix: t.Optional[str] = None):
    for name, param in named_param_it:
        if by_prefix is not None and not name.startswith(by_prefix):
            continue
        param.requires_grad = False
