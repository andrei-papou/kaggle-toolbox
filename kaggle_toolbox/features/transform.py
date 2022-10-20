import functools
import operator
import typing as t

import numpy as np


def contiguous_to_categorical(feature_arr: np.ndarray, num_bins: t.Optional[int] = None) -> np.ndarray:
    num_samples: int
    num_features: int
    feature_axis_added = False
    if len(feature_arr.shape) > 1:
        num_samples = functools.reduce(operator.mul, list(feature_arr.shape)[:-1], 1)
        num_features = feature_arr.shape[-1]
    else:
        num_samples = feature_arr.shape[0]
        num_features = 1
        feature_arr = np.expand_dims(feature_arr, axis=-1)
        feature_axis_added = True
    cat_feature_arr_list = []
    for feature_idx in range(num_features):
        cont_feature_arr = feature_arr[..., feature_idx]
        num_bins = num_bins if num_bins is not None else int(np.floor(1 + np.log2(num_samples)))
        min_val, max_val = cont_feature_arr.min(), cont_feature_arr.max()
        bin_list = [min_val + i / num_bins * (max_val - min_val) for i in range(num_bins + 1)]
        cat_feature_arr_list.append(np.digitize(cont_feature_arr, bins=bin_list))
    cat_feature_arr = np.stack(cat_feature_arr_list, axis=-1)
    if feature_axis_added:
        cat_feature_arr = np.squeeze(cat_feature_arr, axis=-1)
    return cat_feature_arr
