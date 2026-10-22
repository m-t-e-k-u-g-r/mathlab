import logging
import numpy as np

from domain.matrix import SquareMatrix, Matrix
from domain.vector import Vector
from linear_system_operations import solve

logger = logging.getLogger(__name__)

def add_vectors(vectors: list[Vector]) -> Vector:
    return calc_vectors(vectors, subtract=False)


def subtract_vectors(v1: Vector, v2: Vector) -> Vector:
    vectors = [v1, v2]
    return calc_vectors(vectors, subtract=True)


def calc_vectors(vectors: list[Vector], subtract: bool) -> Vector:
    if any(v.dimension != vectors[0].dimension for v in vectors):
        logger.error("Vectors do not have the same dimension")
        raise ValueError("Vectors must have the same dimension")

    result = []
    for d in range(vectors[0].dimension):
        d_sum: float = vectors[0][d].item()
        for v in vectors[1:]:
            if subtract:
                d_sum -= v[d].item()
            else:
                d_sum += v.data[d].item()

        result.append([d_sum])

    return Vector(data = np.array(result), points = None)


def calculate_scalar_product(v1: Vector, v2: Vector) -> float:
    if v1.dimension != v2.dimension:
        raise ValueError("Vectors do not have the same dimension to be scalar product")

    return np.sum([v1[i] * v2[i] for i in range(v1.dimension)])


def calculate_cross_product(v1: Vector, v2: Vector) -> Vector:
    if v1.dimension != v2.dimension:
        raise ValueError("Vectors do not have the same dimension to be cross product")
    if v1.dimension != (3 or 7):
        raise ValueError("Cross product only exists for 3 dimensions")

    result = Vector(size=v1.dimension)
    for d in range(1, v1.dimension + 1):
        index = d % v1.dimension
        next_index = (d + 1) % v1.dimension
        result[d-1] = v1[index] * v2[next_index] - v1[next_index] * v2[index]

    return result

def check_collinearity(v1: Vector, v2: Vector) -> bool:
    if v1.dimension != v2.dimension:
        logger.error("Vectors must have the same dimension to be collinear")
        raise ValueError("Vectors must have the same dimension")

    k_values = [v2.data[i] / v1.data[i] for i in range(v1.dimension)]
    return all(x == k_values[0] for x in k_values)


def check_linear_dependency(vectors: list[Vector]) -> bool:
    if any(v.dimension != len(vectors) for v in vectors):
        logger.error("Vectors do not have the same dimension to be linear dependent")
        raise ValueError("Vectors must have the same dimension")

    data = np.column_stack([v.data for v in vectors])
    equations = SquareMatrix(data)
    rhs = Matrix(rows=len(vectors), cols=1)

    result = solve(equations, rhs)
    return any(v != 0 for v in result.data)


def check_orthogonality(v1: Vector, v2: Vector) -> bool:
    return np.isclose(calculate_scalar_product(v1, v2), 0, rtol=1e-10, atol=1e-10)

