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

    def generate_invertible_matrix(self, size):
        while True:
            try:
                data = self.rng.uniform(-10, 11, size=(size, size))
                np.linalg.inv(data)
                return SquareMatrix(data)
            except np.linalg.LinAlgError:
                pass


    @test_sizes
    def test_add_matrices(self, recursive, size):
        for _ in range(100):
            m1 = Matrix(self.rng.uniform(-10, 11, size=(size, size)))
            m2 = Matrix(self.rng.uniform(-10, 11, size=(size, size)))

            result = mo.add_matrices(m1, m2, recursive).data

            expected = np.add(m1.data, m2.data)
            assert np.allclose(expected, result, rtol=1e-12, atol=1e-12)

    @test_sizes
    def test_subtract_matrices(self, recursive, size):
        for _ in range(100):
            m1 = Matrix(self.rng.uniform(-10, 11, size=(size, size)))
            m2 = Matrix(self.rng.uniform(-10, 11, size=(size, size)))

            result = mo.subtract_matrices(m1, m2, recursive).data

            expected = np.subtract(m1.data, m2.data)
            assert np.allclose(expected, result, rtol=1e-12, atol=1e-12)

    @test_sizes
    def test_multiply_matrices(self, recursive, size):
        for _ in range(100):
            m1 = Matrix(self.rng.uniform(-10, 11, size=(size, size)))
            m2 = Matrix(self.rng.uniform(-10, 11, size=(size, size)))

            result = mo.multiply_matrices(m1, m2, recursive).data

            expected = np.matmul(m1.data, m2.data)
            assert np.allclose(expected, result, rtol=1e-12, atol=1e-12)

    @pytest.mark.parametrize("size", list(range(2, 16)))
    def test_laplace_expansion(self, size):
        for _ in range(100):
            m = SquareMatrix(self.rng.uniform(-10, 11, size=(size, size)))

            expected = np.linalg.det(m.data)
            result = mo.laplace_expansion(m)

            assert np.isclose(expected, result, rtol=1e-12, atol=1e-12)

    @pytest.mark.parametrize("size", list(range(2, 101)))
    def test_gauss_jordan_algorithm(self, size):
        for _ in range(100):
            a = self.generate_invertible_matrix(size)
            b = Matrix(self.rng.uniform(-10, 11, size=(size, 1)))

            result = mo.gauss_jordan_algorithm(a, b).data

            expected = np.linalg.solve(a.data, b.data)
            assert np.allclose(np.matmul(a.data, result), b.data, rtol=1e-12, atol=1e-12)
            assert np.allclose(expected, result, rtol=1e-10, atol=1e-10)
