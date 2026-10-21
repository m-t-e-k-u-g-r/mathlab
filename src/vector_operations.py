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
