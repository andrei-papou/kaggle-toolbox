import itertools
import typing as t

import numpy as np

from kaggle_toolbox.prediction import PredDict


def _pred_list_to_array_id_list_pair(pred_list: t.List[PredDict]) -> t.Tuple[np.ndarray, t.List[str]]:
    assert len(pred_list) > 0
    id_list = list(pred_list[0].keys())
    assert len(id_list) > 0
    num_targets = len(pred_list[0][id_list[0]])
    arr = np.zeros((len(pred_list), len(id_list), num_targets))
    for pred_idx, (id_idx, id_val), target_idx in \
            itertools.product(range(len(pred_list)), enumerate(id_list), range(num_targets)):
        arr[pred_idx, id_idx, target_idx] = pred_list[pred_idx][id_val][target_idx]
    return arr, id_list


def _array_id_list_pair_to_pred(array: np.ndarray, id_list: t.List[str]) -> PredDict:
    return PredDict([(id, target_arr.tolist()) for id, target_arr in zip(id_list, array)])


class EnsemblingStrategy:

    def ensemble_array(self, array: np.ndarray, id_list: t.List[str]) -> np.ndarray:
        """
        Implementation of the ensembling method.

        :param array: Stacked predictions of the models to be ensembled.
            :shape: M x N x P, where:
            M - number of the models to be ensembled,
            N - number of the samples,
            P - number of the targets to be predicted.
        :param id_list: List of the sample ids.
        :return: The ensembled predictions array.
            :shape: N x P, where:
            N - number of the samples,
            P - number of the targets to be predicted.
        """
        raise NotImplementedError()

    def ensemble(self, pred_list: t.List[PredDict]) -> PredDict:
        arr, id_list = _pred_list_to_array_id_list_pair(pred_list=pred_list)
        ensembled_arr = self.ensemble_array(array=arr, id_list=id_list)
        return _array_id_list_pair_to_pred(array=ensembled_arr, id_list=id_list)


class MeanEnsemblingStrategy(EnsemblingStrategy):

    def ensemble_array(self, array: np.ndarray, id_list: t.List[str]) -> np.ndarray:
        return array.mean(axis=0)
