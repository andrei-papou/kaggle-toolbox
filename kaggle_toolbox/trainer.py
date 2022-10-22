import typing as t
from pathlib import Path

import torch
from torch.cuda.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader, default_collate as default_collate_fn

from kaggle_toolbox.context import ContextManagerList
from kaggle_toolbox.data import DatasetItem, Movable
from kaggle_toolbox.device import Device
from kaggle_toolbox.iter import Index, SizedIter, IterPlannerBuilder, FixedSubsetIterPlannerBuilder, \
    FracSubsetSize
from kaggle_toolbox.logging.base import Logger
from kaggle_toolbox.loss import Loss
from kaggle_toolbox.lr_scheduling import LRScheduler
from kaggle_toolbox.metrics import MeanMetric, PredQualityMetric, MetricCriteria
from kaggle_toolbox.model import Model
from kaggle_toolbox.oof import OOFPredDict
from kaggle_toolbox.progress import ProgressBar, ASCIIProgressBar

_X = t.TypeVar('_X', bound=Movable)

IterationMetricDict = t.Dict[str, float]


class IterationTrainer(t.Generic[_X]):

    def __init__(self, device: Device):
        self._device = device

    @property
    def device(self) -> Device:
        return self._device

    def do_train_iteration(
            self,
            data_iter: SizedIter[t.Tuple[Index, DatasetItem[_X]]]) -> IterationMetricDict:
        raise NotImplementedError()

    def do_valid_iteration(
            self,
            data_loader: DataLoader[DatasetItem[_X]]) -> t.Tuple[IterationMetricDict, OOFPredDict]:
        raise NotImplementedError()

    def save_result(self, to_path: Path):
        raise NotImplementedError()


class StandardIterationTrainer(IterationTrainer[_X]):

    def __init__(
            self,
            model: Model[_X],
            criterion: Loss,
            optimizer: Optimizer,
            scheduler: LRScheduler,
            accumulate_gradient_steps: int,
            device: Device,
            grad_scaler: t.Optional[GradScaler] = None,
            pred_quality_metric_list: t.Optional[t.List[PredQualityMetric]] = None,
            map_output_to_pred: t.Callable[[torch.Tensor], torch.Tensor] = lambda x: x,
            progress_bar: t.Optional[ProgressBar] = None):
        super().__init__(device=device)
        self._model = model.to(device.as_torch)
        self._criterion = criterion
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._accumulate_gradient_steps = accumulate_gradient_steps
        self._device = device
        self._grad_scaler = grad_scaler
        self._loss_metric: MeanMetric = MeanMetric()
        self._pred_quality_metric_list = pred_quality_metric_list if pred_quality_metric_list is not None else []
        self._map_output_to_pred = map_output_to_pred
        self._progress_bar: ProgressBar = progress_bar if progress_bar is not None else ASCIIProgressBar()

    @property
    def device(self) -> Device:
        return self._device

    def do_train_iteration(
            self,
            data_iter: SizedIter[t.Tuple[Index, DatasetItem[_X]]]) -> IterationMetricDict:
        self._model.train()

        it = self._progress_bar(data_iter, desc='Training.')
        with \
                self._loss_metric as loss_metric, \
                ContextManagerList(self._pred_quality_metric_list) as pred_quality_metric_list:
            for idx, batch in it:
                x, y = batch.x, batch.y
                x = x.to(device=self._device)
                y = y.to(device=self._device.as_str)

                with autocast(enabled=self._grad_scaler is not None):
                    pred = self._model(x)
                    loss = self._criterion(pred, y)
                if self._accumulate_gradient_steps > 1:
                    loss /= self._accumulate_gradient_steps
                if self._grad_scaler is not None:
                    grad_scaler.scale(loss).backward()  # type: ignore
                else:
                    loss.backward()

                if (idx.local_step_pos[0] + 1) % self._accumulate_gradient_steps == 0:
                    self._optimizer.step()
                    self._optimizer.zero_grad()
                    self._scheduler.step()

                pred = self._map_output_to_pred(pred)
                loss_metric(loss.cpu() * self._accumulate_gradient_steps)
                pred_cpu, y_cpu = pred.cpu(), y.cpu()
                for metric in pred_quality_metric_list:
                    metric(pred_cpu, y_cpu)

                metric_str = ' '.join([f'{m.name}: {m.compute():.4f}' for m in pred_quality_metric_list])
                it.set_description(
                    f'Training. '
                    f'epoch: {idx.global_step} [{(idx.local_step_pos[0] + 1) / idx.local_step_pos[1]:.3f}] '
                    f'loss: {loss_metric.compute():.4f} {metric_str}')

            return {
                'train_loss': loss_metric.compute(),
                **{
                    f'train_{m.name}': m.compute() for m in pred_quality_metric_list
                }
            }

    @torch.no_grad()
    def do_valid_iteration(
            self,
            data_loader: DataLoader[DatasetItem[_X]]) -> t.Tuple[IterationMetricDict, OOFPredDict]:
        self._model.eval()

        pred_dict = OOFPredDict()
        it = self._progress_bar(data_loader, desc='Validating.', total=len(data_loader))
        with \
                self._loss_metric as loss_metric, \
                ContextManagerList(self._pred_quality_metric_list) as pred_quality_metric_list:
            batch: DatasetItem[_X]
            for batch in it:
                x, y = batch.x, batch.y
                x = x.to(device=self._device)
                y = y.to(device=self._device.as_torch)

                with autocast(enabled=self._grad_scaler is not None):
                    pred = self._model(x)
                    loss = self._criterion(pred, y)

                pred = self._map_output_to_pred(pred)
                loss_metric(loss.cpu())
                pred_cpu, y_cpu = pred.cpu(), y.cpu()
                for metric in pred_quality_metric_list:
                    metric(pred_cpu, y_cpu)
                pred_dict.update(dict(zip(batch.id, [x.tolist() for x in pred.cpu()])))

                metric_str = ' '.join([f'{m.name}: {m.compute():.4f}' for m in pred_quality_metric_list])
                it.set_description(
                    f'Validating. '
                    f'loss: {loss_metric.compute():.4f} {metric_str}')

            return {
                'valid_loss': loss_metric.compute(),
                **{
                    f'valid_{m.name}': m.compute() for m in pred_quality_metric_list
                }
            }, pred_dict

    def save_result(self, to_path: Path):
        torch.save(self._model.state_dict(), to_path)


class FullCycleTrainer(t.Generic[_X]):

    def __init__(
            self,
            iteration_trainer: IterationTrainer[_X],
            num_epochs: int,
            batch_size: int,
            num_workers: int,
            model_comparison_metric: str,
            model_comparison_metric_criteria: MetricCriteria,
            train_iter_planner_builder: t.Optional[IterPlannerBuilder] = None,
            collator: t.Optional[t.Callable[[t.List[DatasetItem[_X]]], DatasetItem[_X]]] = None,
            save_model_to_path: t.Optional[Path] = None,
            logger_list: t.Optional[t.List[Logger]] = None,
            max_steps_no_improvement: t.Optional[int] = None,
            stop_at_epoch: t.Optional[int] = None):
        self._iteration_trainer = iteration_trainer
        self._num_epochs = num_epochs
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._model_comparison_metric = model_comparison_metric
        self._model_comparison_metric_criteria = model_comparison_metric_criteria
        self._train_iter_planner_builder: IterPlannerBuilder = train_iter_planner_builder \
            if train_iter_planner_builder is not None \
            else FixedSubsetIterPlannerBuilder(subset_size=FracSubsetSize(1.0))
        self._collator = collator if collator is not None else default_collate_fn
        self._save_model_to_path = save_model_to_path
        self._logger_list = logger_list if logger_list is not None else []
        self._max_steps_no_improvement = max_steps_no_improvement
        self._stop_at_epoch = stop_at_epoch

    def do_full_cycle(
            self,
            train_dataset: Dataset[DatasetItem[_X]],
            valid_dataset: Dataset[DatasetItem[_X]]) -> t.Tuple[float, OOFPredDict]:
        train_data_loader = DataLoader(
            train_dataset,
            collate_fn=self._collator,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            pin_memory=self._iteration_trainer.device.is_gpu)
        valid_data_loader = DataLoader(
            valid_dataset,
            collate_fn=self._collator,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_workers,
            pin_memory=self._iteration_trainer.device.is_gpu)
        train_data_iter_planner = self._train_iter_planner_builder.build(train_data_loader)

        best_metric = self._model_comparison_metric_criteria.get_initial_value()
        step_metric = best_metric
        steps_no_improvement = 0
        pred_dict: t.Optional[OOFPredDict] = None
        with ContextManagerList(self._logger_list) as logger_list:
            while train_data_iter_planner.epoch < self._num_epochs \
                    and (self._max_steps_no_improvement is None or steps_no_improvement < self._max_steps_no_improvement) \
                    and (self._stop_at_epoch is None or train_data_iter_planner.epoch < self._stop_at_epoch):
                train_metrics_to_track = self._iteration_trainer.do_train_iteration(
                    data_iter=train_data_iter_planner.get_next_iter(step_metric))
                valid_metrics_to_track, iter_pred_dict = self._iteration_trainer.do_valid_iteration(
                    data_loader=valid_data_loader)
                step_metric = valid_metrics_to_track[self._model_comparison_metric]
                for logger in logger_list:
                    logger.log_metrics(
                        step=train_data_iter_planner.step,
                        metrics={
                            **train_metrics_to_track,
                            **valid_metrics_to_track,
                        })
                if self._model_comparison_metric_criteria.is_improvement(best_metric, step_metric):
                    print(f'Best metric improved from {best_metric} to {step_metric}. Saving the model.')
                    if self._save_model_to_path is not None:
                        self._iteration_trainer.save_result(to_path=self._save_model_to_path)
                    best_metric = step_metric
                    pred_dict = iter_pred_dict
                    steps_no_improvement = 0
                else:
                    steps_no_improvement += 1
        assert pred_dict is not None
        return best_metric, pred_dict


def train_kfold_model(
        train_model_fn: t.Callable[[int], t.Tuple[float, OOFPredDict]],
        fold_list: t.List[int],) -> t.Tuple[t.List[float], OOFPredDict]:
    score_list = []
    oof_pred_dict = OOFPredDict()
    for fold in fold_list:
        model_score, model_oof_pred_dict = train_model_fn(fold)
        score_list.append(model_score)
        oof_pred_dict.update(model_oof_pred_dict)
    return score_list, oof_pred_dict
