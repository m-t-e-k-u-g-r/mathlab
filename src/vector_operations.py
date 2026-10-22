import logging
import numpy as np

from domain.vector import Vector

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
