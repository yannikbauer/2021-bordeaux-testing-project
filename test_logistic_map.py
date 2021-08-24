import numpy as np
import pytest

from logistic_map import logistic_map, iterate_f


@pytest.mark.parametrize("x, r, expected", [(0.1, 2.2, 0.198), (0.2, 3.4, 0.544), (0.75, 1.7, 0.31875)])
def test_logistic_map(x, r, expected):
    output = logistic_map(x, r)
    assert np.isclose(output, expected)



@pytest.mark.parametrize("x, r, it, expected", [(0.1, 2.2, 1, [0.198]), (0.2, 3.4, 4, [0.544, 0.843418, 0.449019, 0.841163]), (0.75, 1.7, 2, [0.31875, 0.369152])])
def test_iterate_f(x, r, it, expected):
    output = iterate_f(it, x, r)
    assert np.all(np.isclose(np.array(output), np.array(expected)))

    