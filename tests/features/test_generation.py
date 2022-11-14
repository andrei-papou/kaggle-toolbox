import math

import numpy as np

from kaggle_toolbox.features.generation import Mean, Stdev, \
    FuncBinaryOp


def test_mean():
    gen = Mean(
        name='mean_score',
        feature_list=['fold_0_score', 'fold_1_score', 'fold_2_score'])
    feature_arr = gen({
        'fold_0_score': np.array([0.0]),
        'fold_1_score': np.array([6.0]),
        'fold_2_score': np.array([6.0]),
    })
    assert feature_arr == np.array([4.0])


def test_stdev():
    gen = Stdev(
        name='mean_score',
        feature_list=['fold_0_score', 'fold_1_score', 'fold_2_score'])
    feature_arr = gen({
        'fold_0_score': np.array([0.0]),
        'fold_1_score': np.array([3.0]),
        'fold_2_score': np.array([6.0]),
    })
    assert math.isclose(feature_arr, np.array([np.array([0.0, 3.0, 6.0]).std()]))


def test_func_binary_op():
    gen = FuncBinaryOp(
        name='a_plus_b',
        lhs_feature='a',
        rhs_feature='b',
        func=lambda x, y: x + y)
    feature_arr = gen({
        'a': np.array([1.0, 2.0]),
        'b': np.array([3.0, 4.0]),
    })
    assert (feature_arr == np.array([4.0, 6.0])).all()
