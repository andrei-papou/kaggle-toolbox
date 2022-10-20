from transformers.trainer_utils import set_seed as set_huggingface_seed

from kaggle_toolbox.seed import seed_everything as base_seed_everything


def seed_everything(seed: int):
    base_seed_everything(seed=seed)
    set_huggingface_seed(seed)
