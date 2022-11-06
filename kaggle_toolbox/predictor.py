import typing as t

import torch
from torch.cuda.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler
from torch.utils.data import Dataset, DataLoader, default_collate as default_collate_fn

from kaggle_toolbox.data import DatasetItem, Movable
from kaggle_toolbox.device import Device
from kaggle_toolbox.model import Model
from kaggle_toolbox.prediction import PredDict
from kaggle_toolbox.progress import ProgressBar, ASCIIProgressBar
from kaggle_toolbox.typing import ensure_list

_X = t.TypeVar('_X', bound=Movable)


class Predictor(t.Generic[_X]):

    def __init__(self, device: Device):
        self._device = device

    @property
    def device(self) -> Device:
        return self._device

    def predict(self, dataset: Dataset[DatasetItem[_X]]) -> PredDict:
        raise NotImplementedError()


class StandardPredictor(Predictor[_X]):

    def __init__(
            self,
            model: Model[_X],
            batch_size: int,
            num_workers: int,
            device: Device,
            grad_scaler: t.Optional[GradScaler] = None,
            collator: t.Optional[t.Callable[[t.List[DatasetItem[_X]]], DatasetItem[_X]]] = default_collate_fn,
            map_output_to_pred: t.Callable[[torch.Tensor], torch.Tensor] = lambda x: x,
            progress_bar: t.Optional[ProgressBar] = None):
        super().__init__(device=device)
        self._model = model.to(device.as_torch)
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._grad_scaler = grad_scaler
        self._collator = collator
        self._map_output_to_pred = map_output_to_pred
        self._progress_bar: ProgressBar = progress_bar if progress_bar is not None else ASCIIProgressBar()

    @torch.no_grad()
    def predict(self, dataset: Dataset[DatasetItem[_X]]) -> PredDict:
        data_loader = DataLoader(
            dataset,
            collate_fn=self._collator,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_workers,
            pin_memory=self._device.is_gpu)
        self._model.eval()
        pred_dict = PredDict()
        it = self._progress_bar(data_loader, desc='Predicting.', total=len(data_loader))
        for batch in it:
            x = batch.x.to(self._device)

            with autocast(enabled=self._grad_scaler is not None):  # type: ignore
                pred = self._model(x)

            pred = self._map_output_to_pred(pred)
            pred_dict.update(dict(zip(batch.id, [ensure_list(x.tolist()) for x in pred.cpu()])))
        return pred_dict
