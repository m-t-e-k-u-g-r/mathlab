import numpy as np

class Matrix:
    @property
    def rows(self):
        return self.data.shape[0]

    @property
    def cols(self):
        return self.data.shape[1]

    def __init__(self, data: np.ndarray | None = None, rows: int = 2, cols: int = 2):
        if data is None:
            self.data = np.zeros((rows, cols), dtype=float)
        else:
            self.data = np.asarray(data, dtype=float).copy()
            if self.data.ndim != 2:
                raise ValueError("Matrix data must be two-dimensional")

    def copy(self) -> Matrix:
        return Matrix(self.data.copy())

    def get_value(self, row: int, col: int) -> float:
        return self.data[row, col]

    def get_row(self, row: int, copy: bool = False) -> np.ndarray:
        return self.data[row].copy() if copy else self.data[row]

    def get_col(self, col: int, copy: bool = False) -> np.ndarray:
        return self.data[:, col].copy() if copy else self.data[:, col]

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

    def switch_rows(self, i1: int, i2: int):
        if i1 == i2 or i1 >= self.rows or i2 >= self.rows or i1 < 0 or i2 < 0:
            raise ValueError("Invalid rows selected")

        r1_temp = self.get_row(i1).copy()

        self.set_row(i1, self.get_row(i2))
        self.set_row(i2, r1_temp)

    def switch_cols(self, i1: int, i2: int):
        if i1 == i2 or i1 >= self.cols or i2 >= self.cols or i1 < 0 or i2 < 0:
            raise ValueError("Invalid columns selected")

        c1_temp = self.get_col(i1).copy()

        self.set_col(i1, self.get_col(i2))
        self.set_col(i2, c1_temp)

    def extend(self, m: Matrix, overwrite: bool = False) -> Matrix:
        extended = np.concatenate((self.data, m.data), axis=1)

        if overwrite:
            self.data = extended
            return self

        return Matrix(extended)

class SquareMatrix(Matrix):
    def __init__(self, data: np.ndarray | None = None, unit: bool = False, size: int = 2):
        super().__init__(data, rows=size, cols=size)

        if unit:
            self.data = np.identity(size)

        if self.rows != self.cols:
            raise ValueError("Square matrix must have equal number of rows and columns")

    def copy(self) -> SquareMatrix:
        return SquareMatrix(self.data.copy())

    def extend(self, m, overwrite = False) -> Matrix:
        if overwrite:
            raise TypeError("A SquareMatrix cannot be extended in-place.")
        return super().extend(m)
