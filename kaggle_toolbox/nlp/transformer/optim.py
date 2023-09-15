import typing as t

import torch
from torch.optim import Optimizer, AdamW

_NO_DECAY = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']


def create_encoder_decoder_optimizer(
        encoder_named_parameters: t.Iterator[t.Tuple[str, torch.nn.parameter.Parameter]],
        decoder_named_parameters: t.Iterator[t.Tuple[str, torch.nn.parameter.Parameter]],
        encoder_lr: float,
        decoder_lr: float,
        weight_decay: float = 1e-2,
        eps: float = 1e-6,
        betas: t.Tuple[float, float] = (0.9, 0.999)) -> Optimizer:
    backbone_wd_param_list, backbone_nowd_param_list = [], []
    for name, param in encoder_named_parameters:
        if any(no_decay_name in name for no_decay_name in _NO_DECAY):
            backbone_nowd_param_list.append(param)
        else:
            backbone_wd_param_list.append(param)
    optimizer_parameters = [
        {
            'params': backbone_wd_param_list,
            'lr': encoder_lr,
            'weight_decay': weight_decay
        },
        {
            'params': backbone_nowd_param_list,
            'lr': encoder_lr,
            'weight_decay': 0.0
        },
        {
            'params': [p for n, p in decoder_named_parameters],
            'lr': decoder_lr,
            'weight_decay': 0.0
        }
    ]
    return AdamW(
        optimizer_parameters,
        lr=encoder_lr,
        eps=eps,
        betas=betas)
