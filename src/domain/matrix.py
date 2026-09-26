import numpy as np

class Matrix:
    def __init__(self, data: np.ndarray | None = None, rows: int = 2, cols: int = 2):
        if data is None:
            self.data = np.zeros((rows, cols))
        else:
            self.data = np.asarray(data)
            if self.data.ndim != 2:
                raise ValueError("Matrix data must be two-dimensional")

        self.rows, self.cols = self.data.shape

    def get_value(self, row: int, col: int):
        return self.data[row, col]

    def set_value(self, row: int, col: int, value: float):
        self.data[row, col] = value

class SquareMatrix(Matrix):
    def __init__(self, data: np.ndarray | None = None, unit: bool = False, size: int = 2):
        super().__init__(data)

        if unit:
            self.data = np.eye(size)

        if self.rows != self.cols:
            raise ValueError("Square matrix must have equal number of rows and columns")
