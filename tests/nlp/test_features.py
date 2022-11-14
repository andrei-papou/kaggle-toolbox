import numpy as np

from kaggle_toolbox.nlp.features import SubstrCountFeatureGenerator, FuncFeatureGenerator


def test_substr_count():
    gen = SubstrCountFeatureGenerator(name='num_dots', substr='.')
    feature_arr = gen(['Hello. World.'], {})
    assert feature_arr == np.array([2.0])


def test_func():
    gen = FuncFeatureGenerator(name='char_len', func=len)
    feature_arr = gen(['Hello World!'], {})
    assert feature_arr == np.array([12.0])
