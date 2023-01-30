import typing as t

import torch
from torch.optim import Optimizer

from kaggle_toolbox.data import Movable
from kaggle_toolbox.device import Device
from kaggle_toolbox.iter import Index
from kaggle_toolbox.loss import Loss
from kaggle_toolbox.model import Model
from kaggle_toolbox.trainer import StandardIterationTrainer, IterationHook

_X = t.TypeVar('_X', bound=Movable)


class AWP(t.Generic[_X]):

    def __init__(
            self,
            adv_param: str = 'weight',
            adv_lr: float = 1.0,
            adv_eps: float = 0.2,
            start_epoch: int = 0,
            accumultion: int = 1):
        self._adv_param = adv_param
        self._adv_lr = adv_lr
        self._adv_eps = adv_eps
        self._start_epoch = start_epoch
        self._accumultion = accumultion

        self._backup: t.Dict[str, torch.Tensor] = {}
        self._backup_eps: t.Dict[str, t.Tuple[torch.Tensor, torch.Tensor]] = {}

    def attack_backward(
            self,
            model: Model[_X],
            optimizer: Optimizer,
            criterion: Loss,
            device: Device,
            input_pair_list: t.List[t.Tuple[_X, torch.Tensor]],
            epoch: int):
        if (self._adv_lr == 0) or (epoch < self._start_epoch):
            return None

        self._save(model)
        self._attack_step(model)
        optimizer.zero_grad()
        for x, y in input_pair_list:
            x = x.to_device(device=device)
            y = y.to(device=device.as_torch)
            out = model(x)
            adv_loss = criterion(out, y)
            adv_loss = adv_loss / self._accumultion
            adv_loss.backward(
                gradient=torch.ones_like(adv_loss).to(device.as_torch) if adv_loss.size() else None)
        self._restore(model)

    def _attack_step(self, model: Model[_X]):
        e = 1e-6
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None and self._adv_param in name:
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())
                if norm1 != 0 and not torch.isnan(norm1):
                    r_at = self._adv_lr * param.grad / (norm1 + e) * (norm2 + e)
                    param.data.add_(r_at)
                    param.data = torch.min(
                        torch.max(param.data, self._backup_eps[name][0]), self._backup_eps[name][1]
                    )

    def _save(self, model: Model[_X]):
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None and self._adv_param in name:
                if name not in self._backup:
                    self._backup[name] = param.data.clone()
                    grad_eps = self._adv_eps * param.abs().detach()
                    self._backup_eps[name] = (
                        self._backup[name] - grad_eps,
                        self._backup[name] + grad_eps,
                    )

    def _restore(self, model: Model[_X]):
        for name, param in model.named_parameters():
            if name in self._backup:
                param.data = self._backup[name]
        self._backup = {}
        self._backup_eps = {}


class AWPIterationHook(IterationHook[_X]):

    def __init__(self, awp: AWP) -> None:
        self._awp = awp
        self._input_history: t.List[t.Tuple[_X, torch.Tensor]] = []
        self._current_index: t.Optional[Index] = None

    def after_forward_pass(self, trainer: StandardIterationTrainer, idx: Index, x: _X, y: torch.Tensor):
        self._input_history.append((x, y))
        self._current_index = idx

    def before_optimizer_step(self, trainer: StandardIterationTrainer, idx: Index, x: _X, y: torch.Tensor):
        assert self._current_index is not None
        self._awp.attack_backward(
            model=trainer.model,
            optimizer=trainer.optimizer,
            criterion=trainer.criterion,
            device=trainer.device,
            input_pair_list=self._input_history,
            epoch=self._current_index.global_step)
        self._input_history.clear()


class FGM(t.Generic[_X]):

    def __init__(
            self,
            eps: float = 1.0,
            enable_for_param_fn: t.Optional[t.Callable[[str], bool]] = None):
        self._eps = eps
        self._enable_for_param_fn = enable_for_param_fn

        self._backup: t.Dict[str, torch.Tensor] = {}

    def attack(self, model: Model[_X]):
        for name, param in model.named_parameters():
            if param.requires_grad and (self._enable_for_param_fn is None or self._enable_for_param_fn(name)):
                assert param.grad is not None
                self._backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = self._eps * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, model: Model[_X]):
        for name, param in model.named_parameters():
            if param.requires_grad and (self._enable_for_param_fn is None or self._enable_for_param_fn(name)):
                assert name in self._backup
                param.data = self._backup[name]
            self._backup = {}


class FGMIterationHook(IterationHook[_X]):

    def __init__(self, fgm: FGM) -> None:
        super().__init__()
        self._fgm = fgm

    def after_forward_pass(self, trainer: StandardIterationTrainer, idx: Index, x: _X, y: torch.Tensor):
        self._fgm.attack(model=trainer.model)
        with torch.cuda.amp.autocast_mode.autocast(enabled=trainer.grad_scaler is not None):
            y_preds = trainer.model(x)
            loss_adv = trainer.criterion(y_preds, y)
            loss_adv.backward()
        self._fgm.restore(model=trainer.model)
