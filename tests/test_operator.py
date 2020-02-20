import numpy as np
import numpy.testing as np_tst
from scipy.linalg import toeplitz

from hypothesis import given, assume
from hypothesis.extra.numpy import arrays, floating_dtypes
from hypothesis.strategies import shared, integers, tuples, floats, sampled_from

from py_toeplitz import Toeplitz, ConvolveToeplitz
from py_toeplitz.cytoeplitz import CyToeplitz

MAX_ARRAY = 10
# no float16 loop for `np.dot`
FLOAT_SIZES = (32, 64, 128)

# @given(
#     arrays(
#         shared(floating_dtypes(sizes=FLOAT_SIZES), key="dtype"),
#         shared(integers(min_value=1, max_value=MAX_ARRAY), key="nrows")
#     ),
#     arrays(
#         shared(floating_dtypes(sizes=FLOAT_SIZES), key="dtype"),
#         shared(integers(min_value=1, max_value=MAX_ARRAY), key="ncols")
#     ),
#     arrays(
#         shared(floating_dtypes(sizes=FLOAT_SIZES), key="dtype"),
#         shared(integers(min_value=1, max_value=MAX_ARRAY), key="ncols")
#     )
# )
# def test_toeplitz_real_vec(first_col, first_row, test):
#     """Test toeplitz for real inputs."""
#     full_mat = toeplitz(first_col, first_row)
#     toeplitz_op = Toeplitz(first_col, first_row)
#     np_tst.assert_allclose(full_mat.dot(test), toeplitz_op.dot(test))


@given(
    integers(min_value=1, max_value=MAX_ARRAY),
    integers(min_value=1, max_value=MAX_ARRAY),
    sampled_from((Toeplitz, ConvolveToeplitz)),
)
def test_toeplitz_shape(n_rows, n_cols, toep_cls):
    first_col = np.empty(n_rows)
    first_row = np.empty(n_cols)
    assert toeplitz(first_col, first_row).shape == toep_cls(first_col, first_row).shape


@given(
    arrays(
        shared(floating_dtypes(sizes=FLOAT_SIZES, endianness="="), key="dtype"),
        shared(integers(min_value=1, max_value=MAX_ARRAY), key="nrows"),
        elements=floats(allow_infinity=False, allow_nan=False, width=32)
    ),
    arrays(
        shared(floating_dtypes(sizes=FLOAT_SIZES, endianness="="), key="dtype"),
        shared(integers(min_value=1, max_value=MAX_ARRAY), key="ncols"),
        elements=floats(allow_infinity=False, allow_nan=False, width=32)
    ),
    arrays(
        shared(floating_dtypes(sizes=FLOAT_SIZES, endianness="="), key="dtype"),
        tuples(
            shared(integers(min_value=1, max_value=MAX_ARRAY), key="ncols"),
            integers(min_value=1, max_value=MAX_ARRAY)
        ),
        elements=floats(allow_infinity=False, allow_nan=False, width=32)
    ),
    sampled_from((Toeplitz, ConvolveToeplitz, CyToeplitz)),
)
def test_toeplitz_real_mat(first_col, first_row, test, toep_cls):
    """Test toeplitz for real inputs."""
    full_mat = toeplitz(first_col, first_row)
    toeplitz_op = toep_cls(first_col, first_row)
    if first_col.dtype == np.float16:
        atol_frac = 1e-2
    elif first_col.dtype == np.float32:
        atol_frac = 1e-5
    elif first_col.dtype == np.float64:
        atol_frac = 1e-14
    elif first_col.dtype == np.float128:
        atol_frac = 1e-15
    max_el = max(np.max(np.abs(first_col)), np.max(np.abs(first_row)),
                 np.max(np.abs(test)))
    mat_result = full_mat.dot(test)
    if first_col.dtype == np.float32:
        # Apparently `np.dot` uses an extended-precision accumulator
        assume(np.all(np.isfinite(mat_result)))
    np_tst.assert_allclose(
        toeplitz_op.dot(test),
        mat_result,
        atol=atol_frac * max_el, rtol=atol_frac
    )
