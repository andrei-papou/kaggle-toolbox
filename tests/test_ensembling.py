import itertools
import math

from kaggle_toolbox.ensembling import MeanEnsemblingStrategy, MulticlassMajorityVotingEnsemblingStrategy
from kaggle_toolbox.prediction import PredDict


def _pred_dict_is_close(l: PredDict, r: PredDict):
    for k in set(l.keys()) | set(r.keys()):
        for lx, rx in itertools.zip_longest(l[k], r[k]):
            if not math.isclose(lx, rx):
                print(f'`math.isclose` failed for {lx} and {rx}')
                return False
    return True


def test_mean():
    ens = MeanEnsemblingStrategy()

    pred = ens.ensemble([
        PredDict(a=[0.8, 0.2], b=[0.5, 0.4]),
        PredDict(a=[0.9, 0.1], b=[0.0, 0.5]),
        PredDict(a=[1.0, 0.0], b=[1.0, 0.6]),
    ])

    assert _pred_dict_is_close(pred, PredDict(a=[0.9, 0.1], b=[0.5, 0.5]))


def test_multiclass_majority_voting():
    ens = MulticlassMajorityVotingEnsemblingStrategy()

    pred = ens.ensemble([
        PredDict(a=[0.8, 0.2], b=[0.6, 0.4]),
        PredDict(a=[0.1, 0.9], b=[0.7, 0.3]),
        PredDict(a=[1.0, 0.0], b=[0.1, 0.9]),
    ])

    assert _pred_dict_is_close(pred, PredDict(a=[0.9, 0.1], b=[0.65, 0.35]))
