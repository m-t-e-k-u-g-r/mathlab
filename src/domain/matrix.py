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

    def get_value(self, row: int, col: int) -> float:
        return self.data[row, col]

    def get_row(self, row: int) -> np.ndarray:
        return self.data[row]

    def get_col(self, col: int) -> np.ndarray:
        column = [self.data[row, col] for row in range(self.rows)]
        return np.array(column)

    def set_value(self, row: int, col: int, value: float):
        self.data[row, col] = value

    def set_row(self, row: int, values: np.ndarray):
        if row >= self.rows | values.ndim != 1 | len(values) != self.cols:
            raise ValueError("Matrix must have equal number of rows and columns")
        self.data[row] = values

    def set_col(self, col: int, values: np.ndarray):
        if col >= self.cols | values.ndim != 1 | len(values) != self.rows:
            raise ValueError("Matrix must have equal number of columns and rows")
        for i in range(self.rows):
            self.data[i, col] = values[i]

class SquareMatrix(Matrix):
    def __init__(self, data: np.ndarray | None = None, unit: bool = False, size: int = 2):
        super().__init__(data, rows=size, cols=size)

        if unit:
            self.data = np.identity(size)

        if self.rows != self.cols:
            raise ValueError("Square matrix must have equal number of rows and columns")
