import numpy as np
import pytest

from domain.vector import Vector


class TestVectorClass:
    rng = np.random.default_rng(42)

    @pytest.mark.parametrize("size", list(range(1, 11)))
    def test_data_arg_constructor(self, size: int):
        for _ in range(100):
            data = np.array(self.rng.uniform(-100, 100, size=(size, 1)))

            v = Vector(data=data)

            assert np.all(v.data == data)
            assert v.dimension == size

    @pytest.mark.parametrize("size", list(range(1, 11)))
    def test_size_constructor(self, size: int):
        for _ in range(100):
            v = Vector(size=size)

            assert np.all(v.data == 0)

    @pytest.mark.parametrize("size", list(range(1, 101)))
    def test_vector_length(self, size: int):
        for _ in range(100):
            v = Vector(data=np.array(self.rng.uniform(-100, 100, size=(size, 1))))

            result = v.length

            expected = np.linalg.norm(v.data)
            assert np.isclose(expected, result, rtol=1e-10, atol=1e-10)
