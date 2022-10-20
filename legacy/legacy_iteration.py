import typing as t

import torch
from torch.cuda.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader
from torch_iter import Index, SizedIter
from tqdm.notebook import tqdm

from fp_ell.data import DatasetItem
from fp_ell.loss import Loss
from fp_ell.lr_scheduling import LRScheduler
from fp_ell.metrics import MeanMetric, MCRMSEMetric
from fp_ell.model import Model
from fp_ell.optim import Optimizer


def do_train_iteration(
        model: Model,
        data_iter: SizedIter[t.Tuple[Index, DatasetItem]],
        criterion: Loss,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        accumulate_gradient_steps: int,
        device: str,
        grad_scaler: t.Optional[GradScaler] = None,) -> t.Dict[str, float]:
    model.train()
    loss_metric = MeanMetric()
    mcrmse_metric = MCRMSEMetric()

    it = tqdm(data_iter, desc='Training.')
    batch: DatasetItem
    for idx, batch in it:
        model_input = batch.tokenizer_result.get_model_input(device=device)
        target = batch.target.to(device)

        with autocast(enabled=grad_scaler is not None):
            pred = model(**model_input).squeeze(-1)
            loss = criterion(pred, target)
        if accumulate_gradient_steps > 1:
            loss /= accumulate_gradient_steps
        if grad_scaler is not None:
            grad_scaler.scale(loss).backward()  # type: ignore
        else:
            loss.backward()

        if (idx.local_step_pos[0] + 1) % accumulate_gradient_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        loss_metric(loss.cpu() * accumulate_gradient_steps)
        mcrmse_metric(pred.cpu(), target.cpu())

        it.set_description(
            f'Training. '
            f'epoch: {idx.global_step} [{(idx.local_step_pos[0] + 1) / idx.local_step_pos[1]:.3f}] '
            f'loss: {loss_metric.compute():.4f} '
            f'mcrmse: {mcrmse_metric.compute():.4f}'
        )

    return {
        'train_loss': loss_metric.compute(),
        'train_mcrmse': mcrmse_metric.compute(),
    }


@torch.no_grad()
def do_valid_iteration(
        model: Model,
        data_loader: DataLoader,
        criterion: Loss,
        is_autocast_enabled: bool,
        device: str,) -> t.Tuple[t.Dict[str, float], t.Dict[str, t.List[float]]]:
    model.eval()
    loss_metric = MeanMetric()
    mcrmse_metric = MCRMSEMetric()
    pred_dict = {}

    it = tqdm(enumerate(data_loader), desc='Validating.', total=len(data_loader))
    batch: DatasetItem
    for step, batch in it:
        model_input = batch.tokenizer_result.get_model_input(device=device)
        target = batch.target.to(device)

        with autocast(enabled=is_autocast_enabled):
            pred = model(**model_input).squeeze(-1)
            loss = criterion(pred, target)

        loss_metric(loss.cpu())
        mcrmse_metric(pred.cpu(), target.cpu())
        pred_dict.update(dict(zip(batch.id, [x.tolist() for x in pred.cpu()])))

        it.set_description(
            f'Validating. '
            f'loss: {loss_metric.compute():.4f} '
            f'mcrmse: {mcrmse_metric.compute():.4f}'
        )

    return {
        'valid_loss': loss_metric.compute(),
        'valid_mcrmse': mcrmse_metric.compute(),
    }, pred_dict
