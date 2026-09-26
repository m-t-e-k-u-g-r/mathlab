import unittest
import numpy as np

from domain.matrix import Matrix
from matrix_operations import multiply_matrices, add_matrices, subtract_matrices


class MatrixOperationsTests(unittest.TestCase):
    m2x3 = Matrix(np.array([
        [1, 2, 3],
        [4, 5, 6]
    ]))
    m3x2 = Matrix(np.array([
        [1, 2],
        [3, 4],
        [5, 6]
    ]))

    def test_add_matrices(self):
        m1 = self.m2x3

        result = add_matrices(m1, m1).data

        expected = Matrix(np.array([
            [2, 4, 6],
            [8, 10, 12]
        ])).data
        self.assertEqual(expected.all(), result.all())

    def test_subtract_matrices(self):
        m1 = self.m3x2

        result = subtract_matrices(m1, m1).data

        expected = Matrix(np.array([
            [0, 0],
            [0, 0],
            [0, 0]
        ])).data
        self.assertEqual(expected.all(), result.all())

    def test_multiply_matrices(self):
        m1 = self.m2x3
        m2 = self.m3x2

        result = multiply_matrices(m1, m2).data

        expected = Matrix(np.array([
            [22, 28],
            [49, 64]
        ])).data
        self.assertEqual(expected.all(), result.all())

if __name__ == '__main__':
    unittest.main()
