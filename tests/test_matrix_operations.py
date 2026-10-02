import numpy as np

from domain.matrix import Matrix, SquareMatrix
import matrix_operations as mo


class TestMatrixOperations:
    m2x3 = Matrix(np.array([
        [1, 2, 3],
        [4, 5, 6]
    ]))
    m3x2 = Matrix(np.array([
        [1, 2],
        [3, 4],
        [5, 6]
    ]))
    m3x3 = SquareMatrix(np.array([
        [1, 2, 0],
        [2, 4, 1],
        [2, 1, 0]
    ]))

    def test_extend_matrix(self):
        result = mo.extend_matrix(self.m3x2, self.m3x3).data

        expected = np.array([
            [1, 2, 1, 2, 0],
            [3, 4, 2, 4, 1],
            [5, 6, 2, 1, 0]
        ])
        assert np.array_equal(expected, result)

    def test_add_matrices(self):
        m1 = self.m2x3

        result = mo.add_matrices(m1, m1).data

        expected = Matrix(np.array([
            [2, 4, 6],
            [8, 10, 12]
        ])).data

        assert np.array_equal(expected, result)

    def test_subtract_matrices(self):
        m1 = self.m3x2

        result = mo.subtract_matrices(m1, m1).data

        expected = Matrix(np.array([
            [0, 0],
            [0, 0],
            [0, 0]
        ])).data

        assert np.array_equal(expected, result)

    def test_multiply_matrices(self):
        m1 = self.m2x3
        m2 = self.m3x2

        result = mo.multiply_matrices(m1, m2).data

        expected = Matrix(np.array([
            [22, 28],
            [49, 64]
        ])).data

        assert np.array_equal(expected, result)

    def test_invert_matrix(self):
        result = mo.invert_matrix(self.m3x3, None).data

        expected = SquareMatrix(np.array([
            [-1/3, 0, 2/3],
            [2/3, 0, -1/3],
            [-2, 1, 0]
        ])).data

        assert np.array_equal(expected, result)
