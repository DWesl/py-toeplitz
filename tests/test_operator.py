"""Test matrix-vector and matrix-matrix multiplications with Toeplitz classes.

This is primarily implemented with property-based testing, based on
comparisons with :fun:`scipy.linalg.toeplitz`
"""
import numpy as np
import numpy.testing as np_tst
from scipy.linalg import toeplitz

import pytest

from hypothesis import given, assume, target
from hypothesis.extra.numpy import (arrays, floating_dtypes, integer_dtypes,
                                    complex_number_dtypes)
from hypothesis.strategies import (shared, integers, tuples, floats,
                                   builds)

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


@pytest.mark.parametrize("toep_cls", OPERATOR_LIST)
@given(
    integers(min_value=1, max_value=MAX_ARRAY),
    integers(min_value=1, max_value=MAX_ARRAY),
)
def test_toeplitz_shape_dtype(toep_cls, n_rows, n_cols):
    """Test that the shape and dtype of the operators make sense."""
    first_col = np.empty(n_rows)
    first_row = np.empty(n_cols)
    matrix = toeplitz(first_col, first_row)
    operator = toep_cls(first_col, first_row)
    assert matrix.shape == operator.shape
    assert matrix.dtype == operator.dtype


@pytest.mark.parametrize("toep_cls", OPERATOR_LIST)
@given(
    arrays(
        shared(
            floating_dtypes(sizes=FLOAT_SIZES, endianness="="),
            key="dtype"
        ),
        shared(integers(min_value=1, max_value=MAX_ARRAY), key="nrows"),
        elements=floats(allow_infinity=False, allow_nan=False, width=32)
    ),
    arrays(
        shared(
            floating_dtypes(sizes=FLOAT_SIZES, endianness="="),
            key="dtype"
        ),
        shared(integers(min_value=1, max_value=MAX_ARRAY), key="ncols"),
        elements=floats(allow_infinity=False, allow_nan=False, width=32)
    ),
    arrays(
        shared(
            floating_dtypes(sizes=FLOAT_SIZES, endianness="="),
            key="dtype"
        ),
        tuples(
            shared(integers(min_value=1, max_value=MAX_ARRAY), key="ncols"),
            integers(min_value=1, max_value=MAX_ARRAY)
        ),
        elements=floats(allow_infinity=False, allow_nan=False, width=32)
    ),
)
def test_toeplitz_real_mat(toep_cls, first_col, first_row, test):
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
        if toep_cls == FFTToeplitz:
            atol_frac = 1e-14
    max_el = np.max(np.abs(first_col))
    if len(first_row) > 1:
        max_el = max(max_el, np.max(np.abs(first_row[1:])))
    max_test = np.max(np.abs(test))
    if max_el != 0 and max_test != 0:
        max_el *= max_test
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
        atol=(atol_frac * max_el +
              ATOL_MIN * (len(test) + toeplitz_op.shape[0])),
        rtol=atol_frac
    )


@pytest.mark.parametrize("toep_cls", OPERATOR_LIST)
@given(
    arrays(
        shared(
            integer_dtypes(sizes=INTEGER_SIZES, endianness="="),
            key="dtype"
        ),
        shared(integers(min_value=1, max_value=MAX_ARRAY), key="nrows"),
    ),
    arrays(
        shared(
            integer_dtypes(sizes=INTEGER_SIZES, endianness="="),
            key="dtype"
        ),
        shared(integers(min_value=1, max_value=MAX_ARRAY), key="ncols"),
    ),
    arrays(
        shared(
            integer_dtypes(sizes=INTEGER_SIZES, endianness="="),
            key="dtype"
        ),
        tuples(
            shared(integers(min_value=1, max_value=MAX_ARRAY), key="ncols"),
            integers(min_value=1, max_value=MAX_ARRAY)
        ),
    ),
)
def test_toeplitz_int_mat(toep_cls, first_col, first_row, test):
    """Test toeplitz for integer inputs."""
    full_mat = toeplitz(first_col, first_row)
    toeplitz_op = toep_cls(first_col, first_row)
    mat_result = full_mat.dot(test)
    if toep_cls in (ConvolveToeplitz, FFTToeplitz):
        rtol = 2e-6
        max_el = np.max(np.abs(first_col))
        if len(first_row) > 1:
            max_el = max(max_el, np.max(np.abs(first_row[1:])))
        # if max_el != 0:
        #     max_el *= np.max(np.abs(test))
        # assume(np.array(max_el, first_col.dtype)  == max_el)
        atol = abs(rtol * max_el * np.max(np.abs(test)))
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
    else:
        rtol = 0
        atol = 0
    np_tst.assert_allclose(
        toeplitz_op.dot(test),
        mat_result,
        rtol=rtol, atol=atol
    )


@pytest.mark.parametrize("toep_cls", OPERATOR_LIST)
@given(
    arrays(
        shared(
            complex_number_dtypes(sizes=COMPLEX_SIZES, endianness="="),
            key="dtype"
        ),
        shared(integers(min_value=1, max_value=MAX_ARRAY), key="nrows"),
        elements=builds(
            complex,
            floats(allow_infinity=False, allow_nan=False, width=32),
            floats(allow_infinity=False, allow_nan=False, width=32),
        ),
    ).filter(lambda x: np.all(np.isfinite(x))),
    arrays(
        shared(
            complex_number_dtypes(sizes=COMPLEX_SIZES, endianness="="),
            key="dtype"
        ),
        shared(integers(min_value=1, max_value=MAX_ARRAY), key="ncols"),
        elements=builds(
            complex,
            floats(allow_infinity=False, allow_nan=False, width=32),
            floats(allow_infinity=False, allow_nan=False, width=32),
        ),
    ).filter(lambda x: np.all(np.isfinite(x))),
    arrays(
        shared(
            complex_number_dtypes(sizes=COMPLEX_SIZES, endianness="="),
            key="dtype"
        ),
        tuples(
            shared(integers(min_value=1, max_value=MAX_ARRAY), key="ncols"),
            integers(min_value=1, max_value=MAX_ARRAY)
        ),
        elements=builds(
            complex,
            floats(allow_infinity=False, allow_nan=False, width=32),
            floats(allow_infinity=False, allow_nan=False, width=32),
        ),
    ).filter(lambda x: np.all(np.isfinite(x))),
)
def test_toeplitz_complex_mat(toep_cls, first_col, first_row, test):
    """Test toeplitz for complex inputs."""
    full_mat = toeplitz(first_col, first_row)
    toeplitz_op = toep_cls(first_col, first_row)
    if first_col.dtype == np.complex64:
        atol_frac = 1e-5
    elif first_col.dtype == np.complex128:
        atol_frac = 1e-14
    elif first_col.dtype == np.complex256:
        atol_frac = 1e-15
        if toep_cls == FFTToeplitz:
            atol_frac = 1e-14
    max_el = np.max(np.abs(first_col))
    if len(first_row) > 1:
        max_el = max(max_el, np.max(np.abs(first_row[1:])))
    max_test = np.max(np.abs(test))
    if max_el != 0 and max_test != 0:
        max_el *= max_test
    mat_result = full_mat.dot(test)
    # Apparently `np.dot` uses an extended-precision accumulator
    assume(np.all(np.isfinite(mat_result)))
    op_result = toeplitz_op.dot(test)
    # np.dot may give nan or zero depending on array rank.
    assume(~np.any(np.isnan(op_result)))
    assume(np.all(np.isfinite(np.abs(op_result))))
    atol = atol_frac * max_el + ATOL_MIN * (len(test) + toeplitz_op.shape[0])
    assume(atol < np.inf)
    assume(atol != np.inf)
    np_tst.assert_allclose(
        op_result,
        mat_result,
        atol=atol,
        rtol=atol_frac
    )


@pytest.mark.parametrize("toep_cls", OPERATOR_LIST)
@given(
    arrays(
        shared(
            floating_dtypes(sizes=FLOAT_SIZES, endianness="="),
            key="dtype"
        ),
        shared(integers(min_value=1, max_value=MAX_ARRAY), key="nrows"),
        elements=floats(allow_infinity=False, allow_nan=False, width=32)
    ),
    arrays(
        shared(
            floating_dtypes(sizes=FLOAT_SIZES, endianness="="),
            key="dtype"
        ),
        tuples(
            shared(integers(min_value=1, max_value=MAX_ARRAY), key="nrows"),
            integers(min_value=1, max_value=MAX_ARRAY)
        ),
        elements=floats(allow_infinity=False, allow_nan=False, width=32)
    ),
)
def test_toeplitz_only_col(toep_cls, first_col, test):
    """Test toeplitz for real inputs."""
    full_mat = toeplitz(first_col)
    toeplitz_op = toep_cls(first_col)
    if first_col.dtype == np.float16:
        atol_frac = 1e-2
    elif first_col.dtype == np.float32:
        atol_frac = 1e-5
    elif first_col.dtype == np.float64:
        atol_frac = 1e-14
    elif first_col.dtype == np.float128:
        atol_frac = 1.1e-14
    max_el = np.max(np.abs(first_col))
    if max_el != 0:
        max_el *= np.max(np.abs(test))
    mat_result = full_mat.dot(test)
    target(float(np.sum(np.isfinite(mat_result))), label="mat_result_finite")
    if first_col.dtype == np.float32:
        # Apparently `np.dot` uses an extended-precision accumulator
        assume(np.all(np.isfinite(mat_result)))
    op_result = toeplitz_op.dot(test)
    target(float(np.sum(np.isfinite(op_result))), label="op_result_finite")
    target(
        float(np.sum(np.isfinite(np.abs(op_result)))),
        label="op_result_mag_finite"
    )
    if toep_cls == FFTToeplitz:
        assume(np.all(np.isfinite(op_result)))
        assume(np.all(np.isfinite(np.abs(op_result))))
    atol = atol_frac * max_el + ATOL_MIN * (len(test) + toeplitz_op.shape[0])
    assume(atol < np.inf)
    np_tst.assert_allclose(
        op_result,
        mat_result,
        atol=atol,
        rtol=atol_frac
    )
