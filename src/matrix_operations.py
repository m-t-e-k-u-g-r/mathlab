import logging

from domain.matrix import Matrix

logger = logging.getLogger(__name__)

def add_matrices(m1: Matrix, m2: Matrix) -> Matrix:
    # todo: implement addition
    pass

def subtract_matrices(m1: Matrix, m2: Matrix) -> Matrix:
    # todo: implement subtraction
    pass

def multiply_matrices(m1: Matrix, m2: Matrix) -> Matrix:
    # todo: implement multiplication
    pass

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
