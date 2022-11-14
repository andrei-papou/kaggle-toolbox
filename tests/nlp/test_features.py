import numpy as np

from kaggle_toolbox.nlp.features import SubstrCount, Func


def test_substr_count():
    gen = SubstrCount(name='num_dots', substr='.')
    feature_arr = gen(['Hello. World.'], {})
    assert feature_arr == np.array([2.0])


def test_func():
    gen = Func(name='char_len', func=len)
    feature_arr = gen(['Hello World!'], {})
    assert feature_arr == np.array([12.0])
