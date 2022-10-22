class SubsetSize:
    
    def as_nat(self, num_total: int) -> int:
        raise NotImplementedError()


class NatSubsetSize(SubsetSize):

    def __init__(self, val: int) -> None:
        self._val = val

    def as_nat(self, num_total: int) -> int:
        return self._val


class FracSubsetSize(SubsetSize):

    def __init__(self, val: float) -> None:
        self._val = val

    def as_nat(self, num_total: int) -> int:
        return int(self._val * num_total)
