import logging

from domain.matrix import Matrix, SquareMatrix

logger = logging.getLogger(__name__)

def add_matrices(m1: Matrix, m2: Matrix, subtract: bool = False) -> Matrix:
    if m1.rows != m2.rows | m1.cols != m2.cols:
        logger.error("Matrix dimensions do not match")
        raise Exception(f"Matrix dimensions do not match: {m1.rows}x{m1.cols} and {m2.rows}x{m2.cols}")

    logger.debug(f"Add matrices: {m1.rows}x{m1.cols} + {m2.rows}x{m2.cols}")
    result = Matrix(rows=m1.rows, cols=m1.cols)

    operation = 's' if subtract else 'a'
    return calc_recursive(m1, m2, result, operation)

def subtract_matrices(m1: Matrix, m2: Matrix) -> Matrix:
    return add_matrices(m1, m2, subtract=True)

def multiply_matrices(m1: Matrix, m2: Matrix) -> Matrix:
    if m1.cols != m2.rows:
        logger.error("Matrix dimensions do not match")
        raise Exception(f"Matrix dimensions do not match: {m1.rows}x{m1.cols} and {m2.rows}x{m2.cols}")

    logger.debug(f"Multiply matrices: {m1.rows}x{m1.cols} * {m2.rows}x{m2.cols}")
    result = Matrix(rows=m1.rows, cols=m2.cols)
    return calc_recursive(m1, m2, result, 'm')

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

def invert_matrix(m1: SquareMatrix, m2: SquareMatrix | None) -> SquareMatrix:
    if m2 is None:
        m2 = SquareMatrix(np.identity(m1.rows))
    if m1.rows != m2.rows:
        logger.error("Matrix dimensions do not match")
        raise Exception("Matrix dimensions do not match")

    return invert_recursive(m1, m2)

def invert_recursive(m1: SquareMatrix, m2: SquareMatrix, row: int = 0, col: int = 0) -> SquareMatrix:
    # todo: implement inversion
    pass
