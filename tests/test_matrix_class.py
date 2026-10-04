import numpy as np

from domain.matrix import Matrix, SquareMatrix

class TestMatrixClass:
    m2x2zero = np.array([[0, 0], [0, 0]])
    m2x3zero = np.array([[0, 0, 0], [0, 0, 0]])
    unit = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    simple_2x2 = np.array([[1, 2], [3, 4]])

    def test_no_args_constructor(self):
        m = Matrix()

        assert 2 == m.rows
        assert 2 == m.cols
        assert np.array_equal(self.m2x2zero, m.data)

    def test_data_arg_constructor(self):
        data = np.array([[1, 2], [3, 4]])

        m = Matrix(data)

        assert 2 == m.rows
        assert 2 == m.cols
        assert np.array_equal(data, m.data)

    def test_shape_arg_constructor(self):
        rows, cols = 2, 3

        m = Matrix(rows=rows, cols=cols)


        assert rows == m.rows
        assert cols == m.cols
        assert np.array_equal(self.m2x3zero, m.data)

    def test_square_unit_constructor(self):
        size = 3
        unit = True

        m = SquareMatrix(unit=unit, size=size)

        assert size == m.rows
        assert size == m.cols
        assert np.array_equal(self.unit, m.data)

    def test_get_value(self):
        m = Matrix(self.simple_2x2)

        result = m.get_value(1, 0)

        assert 3 == result

    def test_set_value(self):
        m = Matrix(self.simple_2x2.copy())

        m.set_value(1, 0, 5)

        assert 5 == m.get_value(1, 0)

    def test_get_row(self):
        m = Matrix(self.simple_2x2)

        result = m.get_row(0)

        assert np.array_equal(self.simple_2x2[0], result)

    def test_set_row(self):
        m = Matrix(self.simple_2x2.copy())
        new_row = np.array([4, 5])

        m.set_row(0, new_row)

        assert np.array_equal(m.get_row(0), new_row)

    def test_get_col(self):
        m = Matrix(self.simple_2x2)

        result = m.get_col(0)

        assert np.array_equal(np.array([1, 3]), result)

    def test_set_col(self):
        m = Matrix(self.simple_2x2.copy())
        new_col = np.array([10, 11])

        m.set_col(0, new_col)

        assert np.array_equal(m.get_col(0), new_col)

    def test_extend_matrix(self):
        m1 = Matrix(np.array([[1, 2], [3, 4], [5, 6]]))
        m2 = Matrix(np.array([[1, 2, 0], [2, 4, 1], [2, 1, 0]]))

        m1.extend(m2)

        expected = np.array([
            [1, 2, 1, 2, 0],
            [3, 4, 2, 4, 1],
            [5, 6, 2, 1, 0]
        ])
        assert np.array_equal(expected, m1.data)
