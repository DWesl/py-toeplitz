"""Test implementation of `.solve()` method."""
import itertools

import numpy as np
import numpy.testing as np_tst
from scipy.linalg import solve_toeplitz

import pytest

from py_toeplitz import PyToeplitz, ConvolveToeplitz, FFTToeplitz
from py_toeplitz.cytoeplitz import CyToeplitz

OPERATOR_LIST = (PyToeplitz, CyToeplitz,
                 ConvolveToeplitz, FFTToeplitz)
ATOL_MIN = 1e-14
ODD_LENGTH_TEST_ARRAYS = [
    0.5 ** np.arange(4),
    0.333 ** np.arange(4),
    1. / np.arange(1, 5),
    np.exp(-np.arange(4) ** 2),
]
EVEN_LENGTH_TEST_ARRAYS = [
    0.5 ** np.arange(5),
    0.333 ** np.arange(5),
    1. / np.arange(1, 6),
    np.exp(-np.arange(5) ** 2),
]


@pytest.mark.parametrize("toep_cls", OPERATOR_LIST)
@pytest.mark.parametrize(
    "first_col,first_row",
    itertools.chain(
        itertools.product(ODD_LENGTH_TEST_ARRAYS, ODD_LENGTH_TEST_ARRAYS),
        itertools.product(EVEN_LENGTH_TEST_ARRAYS, EVEN_LENGTH_TEST_ARRAYS),
    )
)
def test_toeplitz_solve(toep_cls, first_col, first_row):
    """Test toeplitz for real inputs."""
    toeplitz_op = toep_cls(first_col, first_row)
    if first_col.dtype == np.float32:
        atol_frac = 1e-5
    elif first_col.dtype == np.float64:
        atol_frac = 1e-14
    max_el = np.max(np.abs(first_col))
    if len(first_row) > 1:
        max_el = max(max_el, np.max(np.abs(first_row[1:])))
    test = np.ones_like(first_col)
    mat_result = solve_toeplitz((first_col, first_row), test)
    op_result = toeplitz_op.solve(test)
    np_tst.assert_allclose(
        op_result,
        mat_result,
        atol=(atol_frac * max_el +
              ATOL_MIN * (len(test) + toeplitz_op.shape[0])),
        rtol=atol_frac
    )
