import logging
import unittest
import numpy as np

from domain.matrix import Matrix, SquareMatrix

logger = logging.getLogger("tests")

class MatrixClassTest(unittest.TestCase):
    m2x2zero = np.array([[0, 0], [0, 0]])
    m2x3zero = np.array([[0, 0, 0], [0, 0, 0]])
    unit = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    simple_2x2 = np.array([[1, 2], [3, 4]])

    def test_no_args_constructor(self):
        m = Matrix()

        self.assertEqual(2, m.rows)
        self.assertEqual(2, m.cols)
        self.assertTrue(np.array_equal(self.m2x2zero, m.data))

    def test_data_arg_constructor(self):
        data = np.array([[1, 2], [3, 4]])

        m = Matrix(data)

        self.assertEqual(2, m.rows)
        self.assertEqual(2, m.cols)
        self.assertTrue(np.array_equal(data, m.data))

    def test_shape_arg_constructor(self):
        rows, cols = 2, 3

        m = Matrix(rows=rows, cols=cols)


        self.assertEqual(rows, m.rows)
        self.assertEqual(cols, m.cols)
        self.assertTrue(np.array_equal(self.m2x3zero, m.data))

    def test_square_unit_constructor(self):
        size = 3
        unit = True

        m = SquareMatrix(unit=unit, size=size)

        self.assertEqual(size, m.rows)
        self.assertEqual(size, m.cols)
        self.assertTrue(np.array_equal(self.unit, m.data))

    def test_get_value(self):
        m = Matrix(self.simple_2x2)

        result = m.get_value(1, 0)

        self.assertTrue(3 == result)

    def test_set_value(self):
        m = Matrix(self.simple_2x2)

        m.set_value(1, 0, 5)

        self.assertTrue(5, m.get_value(1, 0))

    def test_get_row(self):
        m = Matrix(self.simple_2x2)

        result = m.get_row(0)

        equal = np.array_equal(self.simple_2x2[0], result)
        self.assertTrue(equal)

    def test_set_row(self):
        m = Matrix(self.simple_2x2)
        new_row = np.array([4, 5])

        m.set_row(0, new_row)

        self.assertTrue(np.array_equal(m.get_row(0), new_row))

    def test_get_col(self):
        m = Matrix(self.simple_2x2)

        result = m.get_col(0)

        equal = np.array_equal(np.array([1, 3]), result)
        self.assertTrue(equal)

    def test_set_col(self):
        m = Matrix(self.simple_2x2)
        new_col = np.array([10, 11])

        m.set_col(0, new_col)

        self.assertTrue(np.array_equal(m.get_col(0), new_col))

if __name__ == '__main__':
    unittest.main()
