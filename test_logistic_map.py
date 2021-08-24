import numpy as np
import pytest

from logistic_map import logistic_map


@pytest.mark.parametrize("x, r, expected", [(0.1, 2.2, 0.198), (0.2, 3.4, 0.544), (0.75, 1.7, 0.31875)])
def test_logistic_map(x, r, expected):
    output = logistic_map(x, r)
    assert np.isclose(output, expected)