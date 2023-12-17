import typing as t

import torch


Initializer = t.Callable[[torch.Tensor], torch.Tensor]
