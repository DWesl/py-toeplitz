import numpy as np
import numpy.testing as np_tst
from scipy.linalg import toeplitz

from hypothesis import given, assume
from hypothesis.extra.numpy import (arrays, floating_dtypes, integer_dtypes,
                                    complex_number_dtypes)
from hypothesis.strategies import (shared, integers, tuples, floats,
                                   sampled_from, complex_numbers)

from py_toeplitz import Toeplitz, ConvolveToeplitz, FFTToeplitz
from py_toeplitz.cytoeplitz import CyToeplitz

MAX_ARRAY = 10
# no float16 loop for `np.dot`
FLOAT_SIZES = (32, 64, 128)
INTEGER_SIZES = (8, 16, 32, 64)
COMPLEX_SIZES = (64, 128)
INT8_MAX = 128
OPERATOR_LIST = (Toeplitz, CyToeplitz,
                 ConvolveToeplitz, FFTToeplitz)
ATOL_MIN = 1e-14

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
    sampled_from(OPERATOR_LIST),
)
def test_toeplitz_shape_dtype(n_rows, n_cols, toep_cls):
    first_col = np.empty(n_rows)
    first_row = np.empty(n_cols)
    matrix = toeplitz(first_col, first_row)
    operator = toep_cls(first_col, first_row)
    assert matrix.shape == operator.shape
    assert matrix.dtype == operator.dtype


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
    sampled_from(OPERATOR_LIST),
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
        atol_frac = 1.1e-15
    max_el = np.max(np.abs(first_col))
    if len(first_row) > 1:
        max_el = max(max_el, np.max(np.abs(first_row[1:])))
    max_el *= np.max(np.abs(test))
    mat_result = full_mat.dot(test)
    if first_col.dtype == np.float32:
        # Apparently `np.dot` uses an extended-precision accumulator
        assume(np.all(np.isfinite(mat_result)))
    op_result = toeplitz_op.dot(test)
    if toep_cls == FFTToeplitz:
        assume(np.all(np.isfinite(op_result)))
    np_tst.assert_allclose(
        op_result,
        mat_result,
        atol=atol_frac * max_el + ATOL_MIN, rtol=atol_frac
    )


@given(
    arrays(
        shared(integer_dtypes(sizes=INTEGER_SIZES, endianness="="), key="dtype"),
        shared(integers(min_value=1, max_value=MAX_ARRAY), key="nrows"),
    ),
    arrays(
        shared(integer_dtypes(sizes=INTEGER_SIZES, endianness="="), key="dtype"),
        shared(integers(min_value=1, max_value=MAX_ARRAY), key="ncols"),
    ),
    arrays(
        shared(integer_dtypes(sizes=INTEGER_SIZES, endianness="="), key="dtype"),
        tuples(
            shared(integers(min_value=1, max_value=MAX_ARRAY), key="ncols"),
            integers(min_value=1, max_value=MAX_ARRAY)
        ),
    ),
    sampled_from(OPERATOR_LIST),
)
def test_toeplitz_int_mat(first_col, first_row, test, toep_cls):
    """Test toeplitz for real inputs."""
    full_mat = toeplitz(first_col, first_row)
    toeplitz_op = toep_cls(first_col, first_row)
    mat_result = full_mat.dot(test)
    if toep_cls in (ConvolveToeplitz, FFTToeplitz):
        rtol = 2e-6
    else:
        rtol = 0
    if toep_cls == FFTToeplitz:
        mat_result_long = toeplitz(
            first_col.astype(float),
            first_row.astype(float)
        ).dot(
            test.astype(float)
        )
        assume(np.allclose(
            mat_result,
            mat_result_long.astype(first_col.dtype)
        ))
    np_tst.assert_allclose(
        toeplitz_op.dot(test),
        mat_result,
        rtol=rtol
    )


@given(
    arrays(
        shared(complex_number_dtypes(sizes=COMPLEX_SIZES, endianness="="), key="dtype"),
        shared(integers(min_value=1, max_value=MAX_ARRAY), key="nrows"),
    ),
    arrays(
        shared(complex_number_dtypes(sizes=COMPLEX_SIZES, endianness="="), key="dtype"),
        shared(integers(min_value=1, max_value=MAX_ARRAY), key="ncols"),
    ),
    arrays(
        shared(complex_number_dtypes(sizes=COMPLEX_SIZES, endianness="="), key="dtype"),
        tuples(
            shared(integers(min_value=1, max_value=MAX_ARRAY), key="ncols"),
            integers(min_value=1, max_value=MAX_ARRAY)
        ),
    ),
    sampled_from(OPERATOR_LIST),
)
def test_toeplitz_complex_mat(first_col, first_row, test, toep_cls):
    """Test toeplitz for real inputs."""
    full_mat = toeplitz(first_col, first_row)
    toeplitz_op = toep_cls(first_col, first_row)
    if first_col.dtype == np.complex64:
        atol_frac = 1e-5
    elif first_col.dtype == np.complex128:
        atol_frac = 1e-14
    elif first_col.dtype == np.complex256:
        atol_frac = 1e-15
    max_el = np.max(np.abs(first_col))
    if len(first_row) > 1:
        max_el = max(max_el, np.max(np.abs(first_row[1:])))
    max_el *= np.max(np.abs(test))
    mat_result = full_mat.dot(test)
    # Apparently `np.dot` uses an extended-precision accumulator
    assume(np.all(np.isfinite(mat_result)))
    op_result = toeplitz_op.dot(test)
    # np.dot may give nan or zero depending on array rank.
    assume(~np.any(np.isnan(op_result)))
    np_tst.assert_allclose(
        op_result,
        mat_result,
        atol=atol_frac * max_el + ATOL_MIN, rtol=atol_frac
    )
