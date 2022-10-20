from pydantic import BaseModel
from torch.cuda.amp.grad_scaler import GradScaler


class FP16Config(BaseModel):
    init_scale: float = 2.**16
    growth_factor: float = 2.0
    backoff_factor: float = 0.5
    growth_interval: int = 2000

    def to_grad_scaler(self) -> GradScaler:
        return GradScaler(
            init_scale=self.init_scale,
            growth_factor=self.growth_factor,
            backoff_factor=self.backoff_factor,
            growth_interval=self.growth_interval)
