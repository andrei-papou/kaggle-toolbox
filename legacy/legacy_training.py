import typing as t
from pathlib import Path

import pandas as pd
import torch
from torch.cuda.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader, default_collate as default_collate_fn
from torch_iter import IterPlannerBuilder
from torch_iter.metric import MetricCriteria

from fp_ell.context import ContextManagerList
from fp_ell.data import Dataset, Collator, DatasetItem
from fp_ell.iteration import do_train_iteration, do_valid_iteration
from fp_ell.logging import Logger, LoggerBuilder
from fp_ell.loss import Loss
from fp_ell.model import ModelBuilder
from fp_ell.optim import OptimizerBuilder
from fp_ell.lr_scheduling import LRSchedulerBuilder
from fp_ell.tokenization import Tokenizer
from fp_ell.typing import pd_dataframe


def _is_gpu_device(device: t.Union[str, torch.device]) -> bool:
    return str(device).startswith('cuda')


def train_model(
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
        target_list: t.List[str],
        model_builder: ModelBuilder,
        tokenizer: Tokenizer,
        criterion: Loss,
        train_iter_planner_builder: IterPlannerBuilder[DatasetItem],
        device: str,
        to_checkpoint: str,
        num_epochs: int,
        optimizer_builder: OptimizerBuilder,
        lr_scheduler_builder: LRSchedulerBuilder,
        batch_size: int,
        max_len: int,
        num_workers: int,
        model_comparison_metric: str,
        model_comparison_metric_criteria: MetricCriteria,
        collator: t.Optional[Collator] = None,
        grad_scaler: t.Optional[GradScaler] = None,
        accumulate_gradient_steps: int = 1,
        logger_list: t.Optional[t.List[Logger]] = None,
        write_model_to_disk: bool = False,
        max_steps_no_improvement: t.Optional[int] = None,
        stop_at_epoch: t.Optional[int] = None) -> t.Tuple[float, t.Dict[str, t.List[float]]]:
    logger_list = logger_list if logger_list is not None else []

    train_dataset = Dataset(
        df=train_df,
        tokenizer=tokenizer,
        max_len=max_len,
        target_list=target_list)
    valid_dataset = Dataset(
        df=valid_df,
        tokenizer=tokenizer,
        max_len=max_len,
        target_list=target_list)
    valid_dataset.sort_by_tokenizer_input_len()

    collator = collator if collator is not None else Collator(default_collate_fn)
    train_data_loader = DataLoader(
        train_dataset,
        collate_fn=collator,  # type: ignore
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=_is_gpu_device(device))
    valid_data_loader = DataLoader(
        valid_dataset,
        collate_fn=collator,  # type: ignore
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=_is_gpu_device(device))
    train_data_iter_planner = train_iter_planner_builder.build(train_data_loader)
    model = model_builder.build().to(device)

    optimizer = optimizer_builder.build(model)
    scheduler = lr_scheduler_builder.build(
        optimizer=optimizer,
        num_training_steps=len(train_data_loader) // accumulate_gradient_steps * num_epochs)

    best_metric = model_comparison_metric_criteria.get_initial_value()
    step_metric = best_metric
    steps_no_improvement = 0
    pred_dict: t.Optional[t.Dict[str, t.List[float]]] = None
    with ContextManagerList(logger_list) as logger_list:
        while train_data_iter_planner.epoch < num_epochs \
                and (max_steps_no_improvement is None or steps_no_improvement < max_steps_no_improvement) \
                and (stop_at_epoch is None or train_data_iter_planner.epoch < stop_at_epoch):
            train_metrics_to_track = do_train_iteration(
                data_iter=train_data_iter_planner.get_next_iter(step_metric),
                model=model,
                criterion=criterion,
                grad_scaler=grad_scaler,
                device=device,
                optimizer=optimizer,
                scheduler=scheduler,
                accumulate_gradient_steps=accumulate_gradient_steps)
            valid_metrics_to_track, iter_pred_dict = do_valid_iteration(
                data_loader=valid_data_loader,
                model=model,
                criterion=criterion,
                is_autocast_enabled=grad_scaler is not None,
                device=device)
            step_metric = valid_metrics_to_track[model_comparison_metric]
            for logger in logger_list:
                logger.log_metrics(
                    step=train_data_iter_planner.step,
                    metrics={
                        **train_metrics_to_track,
                        **valid_metrics_to_track,
                    })
            if model_comparison_metric_criteria.is_improvement(best_metric, step_metric):
                print(f'Best metric improved from {best_metric} to {step_metric}. Saving the model.')
                if write_model_to_disk:
                    torch.save(model.state_dict(), to_checkpoint)
                best_metric = step_metric
                pred_dict = iter_pred_dict
                steps_no_improvement = 0
            else:
                steps_no_improvement += 1
    assert pred_dict is not None
    return best_metric, pred_dict


def train_k_fold_models(
        all_df: pd.DataFrame,
        fold_list: t.List[int],
        target_list: t.List[str],
        model_builder: ModelBuilder,
        tokenizer: Tokenizer,
        criterion: Loss,
        train_iter_planner_builder: IterPlannerBuilder[DatasetItem],
        device: str,
        to_checkpoint_template: str,
        num_epochs: int,
        optimizer_builder: OptimizerBuilder,
        lr_scheduler_builder: LRSchedulerBuilder,
        batch_size: int,
        max_len: int,
        num_workers: int,
        model_comparison_metric: str,
        model_comparison_metric_criteria: MetricCriteria,
        collator: t.Optional[Collator] = None,
        oof_pred_path_template: t.Optional[Path] = None,
        accumulate_gradient_steps: int = 1,
        grad_scaler: t.Optional[GradScaler] = None,
        logger_builder_list: t.Optional[t.List[LoggerBuilder]] = None,
        write_model_to_disk: bool = False,
        max_steps_no_improvement: t.Optional[int] = None,
        stop_at_epoch: t.Optional[int] = None) -> t.List[float]:
    logger_builder_list = logger_builder_list if logger_builder_list is not None else []
    score_list = []
    for fold in fold_list:
        train_df = pd_dataframe(all_df[all_df['fold'] != fold])
        valid_df = pd_dataframe(all_df[all_df['fold'] == fold])
        to_checkpoint = to_checkpoint_template.format(fold=fold)
        score, fold_pred_dict = train_model(
            train_df=train_df,
            valid_df=valid_df,
            target_list=target_list,
            model_builder=model_builder,
            tokenizer=tokenizer,
            criterion=criterion,
            train_iter_planner_builder=train_iter_planner_builder,
            device=device,
            to_checkpoint=to_checkpoint,
            num_epochs=num_epochs,
            optimizer_builder=optimizer_builder,
            lr_scheduler_builder=lr_scheduler_builder,
            batch_size=batch_size,
            max_len=max_len,
            num_workers=num_workers,
            model_comparison_metric=model_comparison_metric,
            model_comparison_metric_criteria=model_comparison_metric_criteria,
            collator=collator,
            grad_scaler=grad_scaler,
            accumulate_gradient_steps=accumulate_gradient_steps,
            logger_list=[lb.build(fold=fold) for lb in logger_builder_list],
            write_model_to_disk=write_model_to_disk,
            max_steps_no_improvement=max_steps_no_improvement,
            stop_at_epoch=stop_at_epoch)
        score_list.append(score)
        if oof_pred_path_template is not None:
            pd.DataFrame([
                {
                    'id': id,
                    **{
                        f'{target}_pred': score
                    for target, score in zip(target_list, score_list)
                    }
                }
                for id, score_list in fold_pred_dict.items()
            ]).to_csv(str(oof_pred_path_template).format(fold=fold), index=False)
    return score_list
