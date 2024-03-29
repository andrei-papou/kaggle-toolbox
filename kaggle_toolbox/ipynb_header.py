import os
from pathlib import Path


def kgltbx_try_init_kaggle(init_wandb: bool = False):
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore
    except ModuleNotFoundError:
        print('Kaggle initialization failed, probably not running on Kaggle...')
        return

    secrets_client = UserSecretsClient()
    env_var_dict = {
        '__KGLTBX_INSTALL_FROM_GITHUB': '1',
        '__KGLTBX_ENVIRONMENT': 'kaggle',
    }
    if init_wandb:
        env_var_dict['WANDB_TOKEN'] = secrets_client.get_secret('WANDB_TOKEN')
    os.environ.update(env_var_dict)


def kgltbx_try_init_colab(env_gdrive_rel_path: str):
    try:
        from google.colab import drive  # type: ignore
    except ModuleNotFoundError:
        print('Colab initialization failed, probably not running on Colab...')
        return

    drive.mount('/content/drive')

    env_var_dict = {
        '__KGLTBX_INSTALL_FROM_GITHUB': '1',
        '__KGLTBX_ENVIRONMENT': 'colab',
    }
    with open(Path('/content/drive/MyDrive') / env_gdrive_rel_path) as f:
        for line in f:
            line = line.strip()
            k, v, *_ = line.split('=')
            env_var_dict[k] = v
    os.environ.update(env_var_dict)
