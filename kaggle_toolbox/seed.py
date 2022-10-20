import random

import numpy as np
import torch


def seed_everything(seed: int):
    np.random.seed(seed % (2 ** 32 - 1))
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = False  # type: ignore
