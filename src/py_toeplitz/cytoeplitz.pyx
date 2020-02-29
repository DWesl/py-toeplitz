# cythom: embedsignature=True
from numpy import dot, empty, conjugate
from numpy import float32, float64, float128
from numpy import int8, int16, int32, int64
from numpy import complex64, complex128
cimport numpy as np

# These require five arguments, and do not accept two.  I would need
# to write a wrapper to fill in the details.
from scipy.linalg cimport cython_blas  # sdot, ddot, cdotu, zdotu

from . import Toeplitz

ctypedef fused numeric_type:
    np.float32_t
    np.float64_t
    long double
    np.int8_t
    np.int16_t
    np.int32_t
    np.int64_t
    np.complex64_t
    np.complex128_t


ctypedef fused blas_types:
    np.float32_t
    np.float64_t
    np.complex64_t
    np.complex128_t


cdef blas_types blas_dot(blas_types[::1] vec1, blas_types[::1] vec2):
    """Compute BLAS-accelerated dot product.

    Parameters
    ----------
    vec1, vec2: blas_types[::1]

    Returns
    -------
    blas_types
    """
    cdef int n = len(vec1)
    cdef int inc1 = 1
    cdef int inc2 = 1
    cdef blas_types result
    if blas_types is np.float32_t:
        result = cython_blas.sdot(&n, &vec1[0], &inc1, &vec2[0], &inc2)
    elif blas_types is np.float64_t:
        result = cython_blas.ddot(&n, &vec1[0], &inc1, &vec2[0], &inc2)
    elif blas_types is np.complex64_t:
        result = cython_blas.cdotu(&n, &vec1[0], &inc1, &vec2[0], &inc2)
    elif blas_types is np.complex128_t:
        result = cython_blas.zdotu(&n, &vec1[0], &inc1, &vec2[0], &inc2)
    return result


class CyToeplitz(Toeplitz):
    """Class holding toeplitz data."""

    def _matmat(self, numeric_type[:, :] vec):
        """Calculate product of self with vec.

        Parameters
        ----------
        vec : array_like

        Returns
        -------
        product : array_like
        """
        cdef long int n_rows, n_columns, n_data, dot_start, i
        cdef numeric_type[::1] data
        cdef numeric_type[:, ::1] result
        cdef numeric_type[::1] tmp
        cdef long int n_cols_vec = vec.shape[1]
        n_rows, n_columns = self.shape
        data = self._data
        n_data = len(data)
        result = empty((n_rows, n_cols_vec), dtype=self.dtype, order="C")
        dot_start = n_rows - 1
        if numeric_type in blas_types:
            for j in range(n_cols_vec):
                tmp = vec[:, j].copy()
                for i in range(n_rows):
                    result[i, j] = blas_dot(data[dot_start - i: n_data - i], tmp)
        else:
            cy_dot = dot
            for i in range(n_rows):
                tmp = cy_dot(data[dot_start - i: n_data - i], vec)
                result[i, :] = tmp

        return result
