import typing as t

import numpy as np

from kaggle_toolbox.features.generation import FeatureArrayDict, BaseFeatureGenerator, \
    FeatureGenerator
from kaggle_toolbox.progress import ProgressBar


class TextFeatureGenerator(BaseFeatureGenerator):

    def __call__(
            self,
            text_it: t.Iterable[str],
            feature_array_dict: FeatureArrayDict) -> np.ndarray:
        raise NotImplementedError()


class _ElementwiseTextFeatureGenerator(TextFeatureGenerator):

    def _generate(self, text: str, feature_dict: t.Dict[str, float]) -> t.Union[float, np.ndarray]:
        raise NotImplementedError()

    def __call__(
            self,
            text_it: t.Iterable[str],
            feature_array_dict: FeatureArrayDict) -> np.ndarray:
        return np.array([
            self._generate(
                text,
                {
                    feature_name: feature_arr[idx].item()
                    for feature_name, feature_arr in feature_array_dict.items()
                })
            for idx, text in enumerate(text_it)
        ])


def generate_text_features(
        generator_list: t.List[t.Union[FeatureGenerator, TextFeatureGenerator]],
        text_seq: t.Sequence[str],
        progress_bar: t.Optional[ProgressBar] = None,
        init_feature_array_dict: t.Optional[FeatureArrayDict] = None) -> FeatureArrayDict:
    feature_array_dict: FeatureArrayDict = init_feature_array_dict \
        if init_feature_array_dict is not None else {}

    for generator in generator_list:
        text_it = progress_bar(text_seq, desc=generator.name, total=len(text_seq)) \
            if progress_bar is not None else text_seq
        if isinstance(generator, FeatureGenerator):
            feature_array_dict[generator.name] = generator(feature_array_dict)
        elif isinstance(generator, TextFeatureGenerator):
            feature_array_dict[generator.name] = generator(text_it, feature_array_dict)

    return feature_array_dict


class SubstrCount(_ElementwiseTextFeatureGenerator):

    def __init__(self, name: str, substr: str):
        super().__init__(name)
        self._substr = substr

    def _generate(self, text: str, feature_dict: t.Dict[str, float]) -> t.Union[float, np.ndarray]:
        return float(text.count(self._substr))


class Func(_ElementwiseTextFeatureGenerator):

    def __init__(self, name: str, func: t.Callable[[str], t.Union[float, np.ndarray]]):
        super().__init__(name)
        self._func = func

    def _generate(self, text: str, feature_dict: t.Dict[str, float]) -> t.Union[float, np.ndarray]:
        return self._func(text)
