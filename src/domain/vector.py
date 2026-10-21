import numpy as np

class Vector:
    @property
    def dimension(self) -> int:
        return len(self.data)

    @property
    def length(self) -> float:
        return np.sqrt(sum(d[0] ** 2 for d in self.data))

    def __init__(self, data: np.ndarray):
        if data.ndim != 2 or data.shape[1] != 1:
            raise ValueError("Data shape is invalid. Has to be (m, 1)")
        self.data = data.copy()

    def __str__(self) -> str:
        return str(self.data)

    def __getitem__(self, index: int):
        if len(self.data) <= index or index < 0:
            raise IndexError(f"Vector does not have {index - 1} dimensions")
        return self.data[index][0]

    def __setitem__(self, index: int, value: float):
        if len(self.data) <= index or index < 0:
            raise IndexError(f"Vector does not have {index - 1} dimensions")
        self.data[index][0] = value
