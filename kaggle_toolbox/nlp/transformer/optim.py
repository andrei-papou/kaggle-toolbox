import typing as t

from torch.optim import Optimizer, AdamW

from .model import Model


def create_nakama_optimizer(
        model: Model[t.Any],
        encoder_lr: float,
        decoder_lr: float,
        weight_decay: float = 1e-2,
        eps: float = 1e-6,
        betas: t.Tuple[float, float] = (0.9, 0.999)) -> Optimizer:
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {'params': [p for n, p in model.backbone_named_parameters if not any(nd in n for nd in no_decay)],
            'lr': encoder_lr, 'weight_decay': weight_decay},
        {'params': [p for n, p in model.backbone_named_parameters if any(nd in n for nd in no_decay)],
            'lr': encoder_lr, 'weight_decay': 0.0},
        {'params': [p for n, p in model.head_named_parameters],
            'lr': decoder_lr, 'weight_decay': 0.0}
    ]
    return AdamW(
        optimizer_parameters,
        lr=encoder_lr,
        eps=eps,
        betas=betas)
