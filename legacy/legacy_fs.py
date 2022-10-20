import importlib
import os
import typing as t
from pathlib import Path

from fp_ell.environment import EnvironmentType

_CONTEST_NAME = 'feedback-prize-english-language-learning'


class FS:

    @property
    def dataset_dir(self) -> Path:
        raise NotImplementedError()

    @property
    def models_dir(self) -> Path:
        raise NotImplementedError()

    @property
    def tensorboard_dir(self) -> Path:
        raise NotImplementedError()

    @property
    def oof_dir(self) -> Path:
        raise NotImplementedError()

    def initialize(self):
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.tensorboard_dir, exist_ok=True)
        os.makedirs(self.oof_dir, exist_ok=True)


class LocalFS(FS):
    TARGET_ENV = EnvironmentType.local

    _ROOT = Path('/root')
    _CONTEST_DIR = _ROOT / 'fp-ell'

    @property
    def dataset_dir(self) -> Path:
        return self._CONTEST_DIR / 'data'

    @property
    def models_dir(self) -> Path:
        return self._CONTEST_DIR / 'models'

    @property
    def tensorboard_dir(self) -> Path:
        return self._CONTEST_DIR / 'tensorboard'

    @property
    def oof_dir(self) -> Path:
        return self._CONTEST_DIR / 'oof'


class ColabFS(FS):
    _GDRIVE_ROOT = Path('/content/drive')
    _GDRIVE_DIR = _GDRIVE_ROOT / 'MyDrive'

    def __init__(self):
        self._mount_gdrive: t.Callable[[str], None] = importlib.import_module('google.colab.drive').mount

    @property
    def dataset_dir(self) -> Path:
        return self._GDRIVE_DIR / f'data/{_CONTEST_NAME}'

    @property
    def models_dir(self) -> Path:
        return self._GDRIVE_DIR / f'models/{_CONTEST_NAME}'

    @property
    def tensorboard_dir(self) -> Path:
        return self._GDRIVE_DIR / f'tensorboard/{_CONTEST_NAME}'

    @property
    def oof_dir(self) -> Path:
        return self._GDRIVE_DIR / f'oof/{_CONTEST_NAME}'

    def initialize(self):
        self._mount_gdrive(str(self._GDRIVE_ROOT))
        super().initialize()


def create_fs_for_env_type(env_type: EnvironmentType) -> FS:
    if env_type == EnvironmentType.local:
        return LocalFS()
    elif env_type == EnvironmentType.colab:
        return ColabFS()
    EnvironmentType.raise_unknown(env_type)
