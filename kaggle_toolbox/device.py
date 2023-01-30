import torch


class Device:
    is_gpu: bool

    @property
    def as_str(self) -> str:
        raise NotImplementedError()

    def __str__(self) -> str:
        return self.as_str

    @property
    def as_torch(self) -> str:
        return self.as_str


class CPUDevice(Device):
    _str_repr = 'cpu'

    is_gpu = False

    @property
    def as_str(self) -> str:
        return self._str_repr


class UnexpectedGPUModelException(BaseException):
    pass


class CUDADevice(Device):
    _str_repr_prefix = 'cuda'
    _str_repr_template = f'{_str_repr_prefix}:{{id}}'

    is_gpu = True

    def __init__(self, id: int = 0):
        self._id = id

    @property
    def as_str(self) -> str:
        return self._str_repr_template.format(id=self._id)

    def get_name(self) -> str:
        return torch.cuda.get_device_name(self.as_str).lower().replace(' ', '_').replace('-', '_')

    def ensure_hardware_model(self, expected_model: str):
        actual_model = torch.cuda.get_device_name().lower().replace(' ', '_').replace('-', '_')
        if actual_model != expected_model:
            raise UnexpectedGPUModelException(f'Expected GPU {expected_model} but received {actual_model}.')
