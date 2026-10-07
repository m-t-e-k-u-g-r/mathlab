import logging
import numpy as np

from domain.matrix import Matrix, SquareMatrix

logger = logging.getLogger(__name__)

type Interim = tuple[float, SquareMatrix]
# calculate determinant of matrix
def laplace_expansion(m: SquareMatrix, interims: list[Interim] | None = None) -> float:
    if interims is not None:
        v = 0
        for interim in interims:
            # calculate sum of interims
            added = interim[0] * laplace_expansion(interim[1])
            v += added
        return v
    else:
        # return determinant of 2x2 matrix
        if m.rows == 2:
            return (m.get_value(0, 0) * m.get_value(1, 1) -
                    m.get_value(0, 1) * m.get_value(1, 0))

        # count zeros in each row and column
        row_counts = np.count_nonzero(m.data == 0, axis=1)
        col_counts = np.count_nonzero(m.data == 0, axis=0)

        # find highest counts of zero values in rows and columns
        row_max = row_counts.max()
        col_max = col_counts.max()

        fixed_type = 'r' if row_max >= col_max else 'c'
        fixed_index = row_counts.argmax() if fixed_type == 'r' else col_counts.argmax()
        fixed = np.array(
            m.get_row(fixed_index) if fixed_type == 'r'
            else m.get_col(fixed_index),
            copy=True)

        new_interims = []
        # iterate through lines to build interim matrices
        for index, v in enumerate(fixed):
            # skip lines multiplied with 0
            if v == 0:
                continue
            # multiply coefficient with -1 if necessary
            coefficient = v * (-1) ** (fixed_index + index)

            # create interim square matrix
            interim = SquareMatrix(size=len(fixed) - 1)
            next_line = 0
            # iterate through the lines to build interims
            for l in range(len(fixed)):
                # skip the fixed line
                if l == fixed_index:
                    continue

                next_cross = 0
                # iterate through cross lines
                for c in range(interim.rows + 1):
                    if c == index:
                        continue

                    if fixed_type == 'r':
                        interim.set_value(next_line, next_cross, m.get_value(l, c))
                    else:
                        interim.set_value(next_cross, next_line, m.get_value(c, l))
                    next_cross += 1
                next_line += 1
            # append new interim
            new_interims.append((coefficient, interim))
        return laplace_expansion(m, new_interims)


def add_matrices(m1: Matrix, m2: Matrix, recursive: bool, subtract: bool = False) -> Matrix:
    if m1.rows != m2.rows | m1.cols != m2.cols:
        logger.error("Matrix dimensions do not match")
        raise Exception(f"Matrix dimensions do not match: {m1.rows}x{m1.cols} and {m2.rows}x{m2.cols}")

    logger.debug(f"Add matrices: {m1.rows}x{m1.cols} + {m2.rows}x{m2.cols}")
    result = Matrix(rows=m1.rows, cols=m1.cols)

    # recursive approach
    if recursive:
        operation = 's' if subtract else 'a'
        return calc_recursive(m1, m2, result, operation)

    # iterative approach
    for r in range(m1.rows):
        if subtract:
            result.set_row(r, m1.get_row(r) - m2.get_row(r))
        else:
            result.set_row(r, m1.get_row(r) + m2.get_row(r))
    return result


def subtract_matrices(m1: Matrix, m2: Matrix, recursive: bool) -> Matrix:
    return add_matrices(m1, m2, recursive, subtract=True)


def multiply_matrices(m1: Matrix, m2: Matrix, recursive: bool) -> Matrix:
    if m1.cols != m2.rows:
        logger.error("Matrix dimensions do not match")
        raise Exception(f"Matrix dimensions do not match: {m1.rows}x{m1.cols} and {m2.rows}x{m2.cols}")

    logger.debug(f"Multiply matrices: {m1.rows}x{m1.cols} * {m2.rows}x{m2.cols}")
    result = Matrix(rows=m1.rows, cols=m2.cols)
    # recursive approach
    if recursive:
        return calc_recursive(m1, m2, result, 'm')

    # iterative approach
    for r in range(m1.rows):
        new_row = []
        for c in range(m2.cols):
            new_row.append(sum(
                m1.get_row(r) * m2.get_col(c)
            ))
        result.set_row(r, np.array(new_row))
    return result


def calc_recursive(m1: Matrix, m2: Matrix, result: Matrix, operation: str, row: int = 0, col: int = 0) -> Matrix:
    if col >= m2.cols:
        return calc_recursive(m1, m2, result, operation, row + 1, 0)
    if row >= m1.rows:
        logger.debug("Recursive calculation completed")
        return result

    value: float
    match operation:
        case "a": # add two matrices
            value = m1.get_value(row, col) + m2.get_value(row, col)
        case "s": # subtract two matrices
            value = m1.get_value(row, col) - m2.get_value(row, col)
        case "m": # multiply two matrices
            value = sum(m1.get_value(row, i) * m2.get_value(i, col) for i in range(m1.cols))
        case _:
            logger.error("Invalid matrix operation")
            raise Exception(f"Invalid operation: {operation}")

    result.set_value(row, col, value)
    return calc_recursive(m1, m2, result, operation, row, col + 1)


def gauss_jordan_algorithm(a: SquareMatrix, b: Matrix) -> Matrix:
    if a.rows != b.rows:
        raise ValueError("Matrix dimensions do not match")

    augmented = a.extend(b)

    for col in range(a.cols):
        pivot_row = col
        pivot_value = abs(augmented.get_value(col, col))

        if pivot_value == 0:
            for i in range(col + 1, augmented.rows):
                candidate_value = abs(augmented.get_value(i, col))
                if candidate_value > pivot_value:
                    pivot_row = i
                    pivot_value = candidate_value

            if pivot_row != col:
                augmented.switch_rows(col, pivot_row)

        if abs(pivot_value) == 0:
            raise ValueError("Matrix cannot be inverted")

        pivot = augmented.get_value(col, col)
        augmented.set_row(col, augmented.get_row(col) / pivot)

        for i in range(col + 1, a.cols):
            factor = augmented.get_value(i, col)
            new_row = augmented.get_row(i) - factor * augmented.get_row(col)
            augmented.set_row(i, new_row)

    for row in range(a.rows - 1, -1, -1):
        for i in range(row - 1, -1, -1):
            factor = augmented.get_value(i, row)
            new_row = augmented.get_row(i) - factor * augmented.get_row(row)
            augmented.set_row(i, new_row)

    return Matrix(augmented.data[:,a.cols:])
