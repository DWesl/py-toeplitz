"""Classes for Toeplitz matrices.

Various implementations of operations involving toeplitz matrices.
"""
from functools import partial

from numpy import dot, empty, zeros, newaxis, around, asarray
from numpy.fft import fft, ifft, rfft, irfft

from scipy.sparse.linalg.interface import LinearOperator
from scipy.signal import convolve
from scipy.linalg import solve_toeplitz

from .__version__ import VERSION as __version__  # noqa: F401


class Toeplitz(LinearOperator):
    """Class holding toeplitz data.

    Examples
    --------
    >>> import numpy as np
    >>> toep_op1 = Toeplitz([0, 1, 2, 3, 4])
    >>> toep_op1.dot(np.eye(toep_op1.shape[1]))
    array([[0, 1, 2, 3, 4],
           [1, 0, 1, 2, 3],
           [2, 1, 0, 1, 2],
           [3, 2, 1, 0, 1],
           [4, 3, 2, 1, 0]])
    >>> toep_op2 = Toeplitz([0, 1, 2], [3, 4])
    >>> toep_op2.dot(np.eye(toep_op2.shape[1]))
    array([[0, 4],
           [1, 0],
           [2, 1]])
    """

    def __init__(self, first_column, first_row=None):
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
        first_column = asarray(first_column)
        if first_row is None:
            first_row = first_column
        n_rows = len(first_column)
        n_cols = len(first_row)
        super(Toeplitz, self).__init__(
            shape=(n_rows, n_cols),
            dtype=first_column.dtype
        )
        data = empty(n_rows + n_cols - 1, dtype=self.dtype)
        data[-n_cols:] = first_row
        data[:n_rows] = first_column[::-1]
        self._data = data

    def solve(self, b):
        """Solve self.dot(x) = b for x.

        Parameters
        ----------
        b: array_like

        Returns
        -------
        x: np.ndarray

        Examples
        --------
        >>> toep_op = Toeplitz([3, 2, 1])
        >>> toep_op.solve([1, 3, 2])
        array([-0.625,  1.5  , -0.125])
        >>> toep_op2 = Toeplitz([3, 2, 1], [6, 5, 4])
        >>> toep_op2.solve([1, 2, 3])
        array([-6.,  3.,  1.])

        See Also
        --------
        scipy.linalg.solve_toeplitz
        """
        diagonal_index = self.shape[0] - 1
        first_row = self._data[diagonal_index:]
        first_col = self._data[diagonal_index::-1]
        return solve_toeplitz((first_col, first_row), b)

    def _matvec(self, vec):
        """Calculate product of self with vec.

        Parameters
        ----------
        vec : array_like

        Returns
        -------
        product : array_like
        """
        n_rows, n_columns = self.shape
        data = self._data
        n_data = len(data)
        result = empty(n_rows, dtype=self.dtype, order="C")
        dot_start = n_rows - 1
        for i in range(n_rows):
            result[i] = dot(data[dot_start - i: n_data - i], vec)
        return result

    def _matmat(self, mat):
        """Calculate product of self with vec.

        Parameters
        ----------
        mat : array_like

        Returns
        -------
        product : array_like
        """
        n_rows, n_columns = self.shape
        data = self._data
        n_data = len(data)
        dot_start = n_rows - 1
        result = empty((n_rows, mat.shape[1]), dtype=self.dtype, order="C")
        for i in range(n_rows):
            # dot(data[dot_start - i:n_data - i], mat, out=result[i, :])
            result[i, :] = dot(data[dot_start - i:n_data - i], mat)
        return result


class ConvolveToeplitz(Toeplitz):
    """Toeplitz operator using convolve.

    Examples
    --------
    >>> import numpy as np
    >>> toep_op1 = ConvolveToeplitz([0, 1, 2, 3, 4])
    >>> toep_op1.dot(np.eye(toep_op1.shape[1]))
    array([[0., 1., 2., 3., 4.],
           [1., 0., 1., 2., 3.],
           [2., 1., 0., 1., 2.],
           [3., 2., 1., 0., 1.],
           [4., 3., 2., 1., 0.]])
    >>> toep_op2 = ConvolveToeplitz([0, 1, 2], [3, 4])
    >>> toep_op2.dot(np.eye(toep_op2.shape[1]))
    array([[0., 4.],
           [1., 0.],
           [2., 1.]])
    """

    def _matvec(self, vec):
        """Compute product of self with vec.

        Parameters
        ----------
        vec: ndarray

        Returns
        -------
        ndarray
        """
        n_rows, n_cols = self.shape
        expand_dims = False
        if vec.ndim == 2:  # pragma: no cover
            vec = vec[:, 0]
            expand_dims = True
        data = self._data
        result = convolve(vec, data[::-1], "full")
        result_start = n_cols - 1
        result_end = len(result) - n_cols + 1
        result = result[result_start:result_end]
        if expand_dims:  # pragma: no cover
            result = result[:, newaxis]
        return result


class FFTToeplitz(Toeplitz):
    """Toeplitz operator using FFT.

    Examples
    --------
    >>> import numpy as np
    >>> toep_op1 = FFTToeplitz([0, 1, 2, 3, 4])
    >>> toep_op1.dot(np.eye(toep_op1.shape[1]))
    array([[0, 1, 2, 3, 4],
           [1, 0, 1, 2, 3],
           [2, 1, 0, 1, 2],
           [3, 2, 1, 0, 1],
           [4, 3, 2, 1, 0]])
    >>> toep_op2 = FFTToeplitz([0, 1, 2], [3, 4])
    >>> toep_op2.dot(np.eye(toep_op2.shape[1]))
    array([[0, 4],
           [1, 0],
           [2, 1]])
    >>> toep_op3 = FFTToeplitz([1+2j, 3+4j, 5+6j])
    >>> toep_op3.dot(np.eye(toep_op3.shape[1]))
    array([[1.+2.j, 3.+4.j, 5.+6.j],
           [3.+4.j, 1.+2.j, 3.+4.j],
           [5.+6.j, 3.+4.j, 1.+2.j]])
    """

    def __init__(self, first_column, first_row=None):
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
        super(FFTToeplitz, self).__init__(
            first_column, first_row
        )
        n_rows, n_cols = self.shape
        dtype = self.dtype
        computational_shape = n_rows + n_cols - 1
        first_row = self._data[-n_cols:]

        if dtype.kind == "c":
            self._fft = partial(fft, n=computational_shape)
            self._ifft = partial(ifft, n=computational_shape)
        else:
            self._fft = partial(rfft, n=computational_shape)
            self._ifft = partial(irfft, n=computational_shape)
        data = empty(computational_shape, dtype=dtype)
        data[:n_rows] = first_column
        data[n_rows:] = first_row[-1:0:-1]
        spectrum = self._fft(data)
        self._spectrum = spectrum

    def _matvec(self, vec):
        """Compute product of self with vec.

        Parameters
        ----------
        vec: ndarray

        Returns
        -------
        ndarray
        """
        n_rows, n_cols = self.shape
        expand_dims = False
        if vec.ndim == 2:  # pragma: no cover
            vec = vec[:, 0]
            expand_dims = True
        to_transform = zeros(n_rows + n_cols - 1, dtype=vec.dtype)
        to_transform[:n_cols] = vec
        vec_spec = self._fft(to_transform)
        vec_spec *= self._spectrum
        result = self._ifft(vec_spec)
        result = result[:n_rows]
        if expand_dims:  # pragma: no cover
            result = result[:, newaxis]
        if self.dtype.kind != "i":
            return result.astype(self.dtype)
        return around(result).astype(self.dtype)
