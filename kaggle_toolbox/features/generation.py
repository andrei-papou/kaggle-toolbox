import typing as t

import numpy as np

FeatureDict = t.Dict[str, float]
FeatureArrayDict = t.Dict[str, np.ndarray]


class BaseFeatureGenerator:

    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name


class FeatureGenerator(BaseFeatureGenerator):

    def __call__(self, feature_array_dict: FeatureArrayDict) -> np.ndarray:
        raise NotImplementedError()


class _ListAggregationFeatureGenerator(FeatureGenerator):

    def __init__(self, name: str, feature_list: t.List[str]):
        super().__init__(name)
        self._feature_list = feature_list


class FuncList(_ListAggregationFeatureGenerator):

    def __init__(self, name: str, feature_list: t.List[str], func: t.Callable[[FeatureArrayDict], np.ndarray]):
        super().__init__(name, feature_list)
        self._func = func

    def __call__(self, feature_array_dict: FeatureArrayDict) -> np.ndarray:
        return self._func({feature: feature_array_dict[feature] for feature in self._feature_list})


class Mean(_ListAggregationFeatureGenerator):

    def __call__(self, feature_array_dict: FeatureArrayDict) -> np.ndarray:
        return np.stack([
            feature_array_dict[feature] for feature in self._feature_list
        ], axis=0).mean(axis=0)


class Stdev(_ListAggregationFeatureGenerator):

    def __call__(self, feature_array_dict: FeatureArrayDict) -> np.ndarray:
        return np.stack([
            feature_array_dict[feature] for feature in self._feature_list
        ], axis=0).std(axis=0)


_BF = t.TypeVar('_BF', bound='_BinaryOpFeatureGenerator')


class _BinaryOpFeatureGenerator(FeatureGenerator):

    def __init__(self, name: str, lhs_feature: str, rhs_feature: str):
        super().__init__(name)
        self._lhs_feature = lhs_feature
        self._rhs_feature = rhs_feature

    def _generate_from_arrays(
            self,
            lhs_feature_array: np.ndarray,
            rhs_feature_array: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def __call__(self, feature_array_dict: FeatureArrayDict) -> np.ndarray:
        return self._generate_from_arrays(
            lhs_feature_array=feature_array_dict[self._lhs_feature],
            rhs_feature_array=feature_array_dict[self._rhs_feature])

    @classmethod
    def pairwise_from_feature_list(cls: t.Type[_BF], feature_list: t.List[str]) -> t.List[_BF]:
        feature_generator_list = []
        for i, lhs_feature in enumerate(feature_list):
            for rhs_feature in feature_list[i:]:
                feature_generator_list.append(cls(
                    name=f'{lhs_feature}_{rhs_feature}_l1',
                    lhs_feature=lhs_feature,
                    rhs_feature=rhs_feature))
        return feature_generator_list


class FuncBinaryOp(_BinaryOpFeatureGenerator):

    def __init__(
            self,
            name: str,
            lhs_feature: str,
            rhs_feature: str,
            func: t.Callable[[np.ndarray, np.ndarray], np.ndarray]):
        super().__init__(name, lhs_feature, rhs_feature)
        self._func = func

    def _generate_from_arrays(
            self,
            lhs_feature_array: np.ndarray,
            rhs_feature_array: np.ndarray) -> np.ndarray:
        return self._func(lhs_feature_array, rhs_feature_array)


class Div(_BinaryOpFeatureGenerator):

    def _generate_from_arrays(
            self,
            lhs_feature_array: np.ndarray,
            rhs_feature_array: np.ndarray) -> np.ndarray:
        return lhs_feature_array / rhs_feature_array


class L1Distance(_BinaryOpFeatureGenerator):

    def _generate_from_arrays(
            self,
            lhs_feature_array: np.ndarray,
            rhs_feature_array: np.ndarray) -> np.ndarray:
        return np.abs(lhs_feature_array - rhs_feature_array)  # type: ignore
