import numpy as np
import pytest

from domain.matrix import Matrix, SquareMatrix
import matrix_operations as mo


class TestMatrixOperations:
    rng = np.random.default_rng(42)
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
    test_sizes = pytest.mark.parametrize("recursive,size",
        [(True, size) for size in range(2, 31)]
        + [(False, size) for size in range(2, 101)]
    )

    def test_extend_matrix(self):
        result = mo.extend_matrix(self.m3x2, self.m3x3).data

        expected = np.array([
            [1, 2, 1, 2, 0],
            [3, 4, 2, 4, 1],
            [5, 6, 2, 1, 0]
        ])
        assert np.array_equal(expected, result)

    @test_sizes
    def test_add_matrices(self, recursive, size):
        for _ in range(100):
            m1 = Matrix(self.rng.integers(-10, 11, size=(size, size)))
            m2 = Matrix(self.rng.integers(-10, 11, size=(size, size)))

            result = mo.add_matrices(m1, m2, recursive).data

            expected = np.add(m1.data, m2.data)
            assert np.array_equal(expected, result)

    @test_sizes
    def test_subtract_matrices(self, recursive, size):
        for _ in range(100):
            m1 = Matrix(self.rng.integers(-10, 11, size=(size, size)))
            m2 = Matrix(self.rng.integers(-10, 11, size=(size, size)))

            result = mo.subtract_matrices(m1, m2, recursive).data

            expected = np.subtract(m1.data, m2.data)
            assert np.array_equal(expected, result)

    @test_sizes
    def test_multiply_matrices(self, recursive, size):
        for _ in range(100):
            m1 = Matrix(self.rng.integers(-10, 11, size=(size, size)))
            m2 = Matrix(self.rng.integers(-10, 11, size=(size, size)))

            result = mo.multiply_matrices(m1, m2, recursive).data

            expected = np.matmul(m1.data, m2.data)
            assert np.array_equal(expected, result)

    def test_invert_matrix(self):
        result = mo.invert_matrix(self.m3x3).data

        expected = SquareMatrix(np.array([
            [-1/3, 0, 2/3],
            [2/3, 0, -1/3],
            [-2, 1, 0]
        ])).data

        assert np.array_equal(expected, result)

    @pytest.mark.parametrize("size", list(range(2, 16)))
    def test_laplace_expansion(self, size):
        for _ in range(100):
            m = SquareMatrix(self.rng.integers(-10, 11, size=(size, size)))

            expected = np.linalg.det(m.data)
            result = mo.laplace_expansion(m)

            assert np.isclose(expected, result, rtol=1e-12, atol=1e-12)
