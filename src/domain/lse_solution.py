from enum import Enum
import numpy as np

class SolutionType(Enum):
    NO_SOLUTION = "NO SOLUTION"
    INFINITE_SOLUTIONS = "INFINITE SOLUTIONS"
    UNIQUE_SOLUTION = "UNIQUE SOLUTION"

class LinearSystemEquationSolution:
    def __init__(self, s_type: SolutionType, solution: np.ndarray | None = None):
        if (s_type == SolutionType.UNIQUE_SOLUTION and solution is None) or \
                (s_type != SolutionType.UNIQUE_SOLUTION and solution is not None):
            raise ValueError(f"Inconsistent solution type: {s_type.name} | {solution}")
        self.s_type = s_type
        self.solution = solution
