import numpy as np
import pytest

from domain.vector import Vector
import vector_operations as ops


class TestVectorOperations:
    rng = np.random.default_rng(42)
    test_cases = [
        (1, 1), (2, 3), (10, 5),
        (100, 10), (1000, 10),
        (10, 1000), (100, 100),
    ]
    test_sizes = pytest.mark.parametrize("size,number", test_cases)

    @test_sizes
    def test_add_vectors(self, size: int, number: int):
        for _ in range(100):
            vectors: list[Vector] = [Vector(
                data=np.array(self.rng.uniform(-100, 100, size=(size, 1)))
            ) for _ in range(number)]

            result = ops.add_vectors(vectors)

            expected = np.sum([v.data for v in vectors], axis=0)
            assert np.allclose(expected, result.data, rtol=1e-10, atol=1e-10)

    @pytest.mark.parametrize("size", list(range(1, 101)))
    def test_subtract_vectors(self, size: int):
        for _ in range(100):
            v1 = Vector(data=np.array(self.rng.uniform(-100, 100, size=(size, 1))))
            v2 = Vector(data=np.array(self.rng.uniform(-100, 100, size=(size, 1))))

            result = ops.subtract_vectors(v1, v2).data

            expected = v1.data - v2.data
            assert np.allclose(expected, result, rtol=1e-10, atol=1e-10)
