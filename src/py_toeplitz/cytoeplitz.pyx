# cythom: embedsignature=True
from numpy import dot, empty
from numpy import float32, float64, float128

from scipy.sparse.linalg.interface import LinearOperator
# from scipy.linalg cimport cython_blas  # sdot, ddot

ctypedef fused floating_type:
    float
    double
    long double


class CyToeplitz(LinearOperator):
    """Class holding toeplitz data."""

    def __init__(
            self,
            floating_type[:] first_column,
            floating_type[:] first_row=None
    ):
        """Construct a toeplitz operator.

        Parameters
        ----------
        first_column : array_like
            First column of the matrix
        first_row : array_like, optional
            First row of the matrix; first element must be same as
            that of `first_column`

        See Also
        --------
        scipy.linalg.toeplitz : Construct the full array
        """
        if isinstance(first_row, type(None)):
            first_row = first_column
        cdef int n_rows = len(first_column)
        cdef int n_cols = len(first_row)
        if floating_type is float:
            dtype = float32
        elif floating_type is double:
            dtype = float64
        elif floating_type is "long double":
            dtype = float128
        super(CyToeplitz, self).__init__(
            shape=(n_rows, n_cols),
            dtype=dtype
        )
        cdef floating_type[:] data = empty(n_rows + n_cols - 1, dtype=self.dtype)
        data[-n_cols:] = first_row
        data[:n_rows] = first_column[::-1]
        self._data = data

    def _matmat(self, floating_type[:, :] vec):
        """Calculate product of self with vec.

        Parameters
        ----------
        vec : array_like

        Returns
        -------
        product : array_like
        """
        cdef long int n_rows, n_columns, n_data, dot_start, i
        cdef floating_type[::1] data
        cdef floating_type[:, ::1] result
        cdef floating_type[::1] tmp
        n_rows, n_columns = self.shape
        data = self._data
        n_data = len(data)
        result = empty((n_rows, vec.shape[1]), dtype=self.dtype, order="C")
        dot_start = n_rows - 1
        for i in range(n_rows):
            tmp = data[dot_start - i: n_data - i]
            tmp = dot(tmp, vec)
            result[i, :] = tmp
        return result
