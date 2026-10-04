import numpy as np
import pytest

from domain.matrix import Matrix, SquareMatrix
import matrix_operations as mo


class TestMatrixOperations:
    rng = np.random.default_rng(42)
    test_sizes = pytest.mark.parametrize("recursive,size",
        [(True, size) for size in range(2, 31)]
        + [(False, size) for size in range(2, 101)]
    )


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

    @pytest.mark.parametrize("size", list(range(2, 16)))
    def test_laplace_expansion(self, size):
        for _ in range(100):
            m = SquareMatrix(self.rng.integers(-10, 11, size=(size, size)))

            expected = np.linalg.det(m.data)
            result = mo.laplace_expansion(m)

            assert np.isclose(expected, result, rtol=1e-12, atol=1e-12)
