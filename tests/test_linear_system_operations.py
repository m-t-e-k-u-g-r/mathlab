import numpy as np
import pytest
from numpy.linalg import LinAlgError

from domain.lse_solution import LinearSystemEquationSolution, SolutionType
from domain.matrix import Matrix, SquareMatrix
from linear_system_operations import backward_substitution, gauss_algorithm


class TestLinearSystemOperations:
    rng = np.random.default_rng(42)
    test_sizes = pytest.mark.parametrize("recursive,size",
        [(True, size) for size in range(2, 16)]
        + [(False, size) for size in range(2, 51)]
    )
    matrix_a_solved = Matrix(np.array([
        [1, -3, -1,  -9],
        [0,  1,  3,  21],
        [0,  0,  1, -24]
    ]))

    @test_sizes
    def test_gauss_algorithm(self, recursive: bool, size: int):
        for _ in range(100):
            coefficients = SquareMatrix(self.rng.uniform(-10, 11, size=(size, size)))
            right_hand_side = Matrix(self.rng.uniform(-10, 11, size=(size, 1)))

            result = gauss_algorithm(coefficients.copy(), right_hand_side.copy(), recursive)

            try:
                expected = np.linalg.solve(coefficients.data, right_hand_side.data)
                assert result.s_type.name == SolutionType.UNIQUE_SOLUTION.name
                assert np.allclose(expected, result.solution, rtol=1e-9, atol=1e-9)
            except LinAlgError:
                assert result.s_type != SolutionType.UNIQUE_SOLUTION


    def test_backward_substitution(self):
        result = backward_substitution(self.matrix_a_solved)

        expected = LinearSystemEquationSolution(
            SolutionType.UNIQUE_SOLUTION,
            np.array([[246], [93], [-24]]))

        assert expected.s_type == result.s_type
        assert result.solution is not None
        assert np.array_equal(expected.solution, result.solution)
