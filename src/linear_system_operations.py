import numpy as np

from domain.lse_solution import LinearSystemEquationSolution, SolutionType
from domain.matrix import Matrix, SquareMatrix


def solve_lse(coefficient_matrix: SquareMatrix, rhs: Matrix, recursive: bool = False) -> LinearSystemEquationSolution:
    if coefficient_matrix.rows != rhs.rows or rhs.cols != 1:
        raise ValueError("Invalid matrix")

    extended = coefficient_matrix.extend(rhs)
    if recursive:
        return gauss_algorithm_recursive(extended)
    return gauss_algorithm_iterative(extended)


def gauss_algorithm_iterative(equations: Matrix) -> LinearSystemEquationSolution:
    # iterate through all rows
    # of the coefficient matrix
    for row in range(equations.rows):
        pivot_row = row
        pivot_value = equations.get_value(row, row)

        # choose the row with the highest
        # absolute value as pivot
        for candidate in range(row + 1, equations.rows):
            if abs(equations.get_value(candidate, row)) > abs(pivot_value):
                pivot_row = candidate
                pivot_value = equations.get_value(candidate, row)

        # swap the pivot
        if pivot_row != row:
            equations.switch_rows(row, pivot_row)

        # no solution
        if pivot_value == 0:
            continue

        # divide pivot row by pivot value
        new_row = equations.get_row(row) / pivot_value
        equations.set_row(row, new_row)

        for i in range(row + 1, equations.rows):
            value = equations.get_value(i, row)

            pivot_row = equations.get_row(row)
            # subtract the pivot row multiplied by the
            # pivot element in the row to get 0
            new_row = equations.get_row(i) - value * pivot_row
            equations.set_row(i, new_row)

    return backward_substitution(equations)


def gauss_algorithm_recursive(equations: Matrix, row: int = 0, candidate: int = -1, next_index: int = -1) -> LinearSystemEquationSolution:
    if row >= equations.rows:
        return backward_substitution(equations)

    candidate = candidate if candidate != -1 else row + 1
    next_index = next_index if next_index != -1 else row + 1

    if candidate >= equations.rows:
        pivot_value = equations.get_value(row, row)

        # no solution
        if pivot_value == 0:
            return LinearSystemEquationSolution(SolutionType.NO_SOLUTION, None)

        # divide pivot row by pivot value
        pivot_row = equations.get_row(row)
        new_row = pivot_row / pivot_value
        equations.set_row(row, new_row)

        if next_index >= equations.rows:
            return gauss_algorithm_recursive(equations, row + 1)

        next_row = equations.get_row(next_index)
        next_value = next_row[row]
        new_row = next_row - next_value * pivot_row
        equations.set_row(next_index, new_row)
        return gauss_algorithm_recursive(equations, row, candidate, next_index + 1)
    else:
        candidate_value = equations.get_value(candidate, row)
        current = equations.get_value(row, row)
        if abs(candidate_value) > abs(current):
            equations.switch_rows(row, candidate)
        return gauss_algorithm_recursive(equations, row, candidate + 1)


def backward_substitution(augmented: Matrix, row: int | None = None, col: int | None = None, solution: np.ndarray | None = None) -> LinearSystemEquationSolution:
    # Calculate starting values
    row = row if row is not None else augmented.rows - 1
    col = col if col is not None else augmented.rows
    solution = solution if solution is not None else np.zeros((augmented.rows, 1))

    row_values = augmented.get_row(row, copy=True).tolist()
    rhs = row_values.pop()
    if all(v == 0 for v in row_values) and rhs != 0:
        return LinearSystemEquationSolution(SolutionType.NO_SOLUTION, None)

    if all(v == 0 for v in augmented.get_row(row)):
        return LinearSystemEquationSolution(SolutionType.INFINITE_SOLUTIONS, None)

    # return if the top row has been processed
    if row < 0:
        return LinearSystemEquationSolution(SolutionType.UNIQUE_SOLUTION, np.array(solution))

    # iterate through the coefficients of the known
    # variables and calculate the sum
    coefficient_product_sum = sum(
        solution[-i][0] * augmented.get_value(row, -i - 1)
        for i in range(1, augmented.rows - row)
    )

    # subtract the sum of the known
    # coefficients from the right-hand-side
    value = augmented.get_value(row, -1) - coefficient_product_sum

    # add the value to the solution
    solution[row][0] = value

    # move one row upwards
    return backward_substitution(augmented, row - 1, col - 1, solution)
