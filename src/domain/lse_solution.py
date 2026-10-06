from enum import Enum
import numpy as np

class SolutionType(Enum):
    NO_SOLUTION = "NO SOLUTION"
    INFINITE_SOLUTIONS = "INFINITE SOLUTIONS"
    UNIQUE_SOLUTION = "UNIQUE SOLUTION"

class LinearSystemEquationSolution:
    def __init__(self, s_type: SolutionType, solution: np.ndarray | None):
        self.s_type = s_type
        self.solution = solution
