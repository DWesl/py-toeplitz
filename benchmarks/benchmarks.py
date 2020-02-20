# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.
from numpy import arange
from scipy.linalg import toeplitz

from py_toeplitz import Toeplitz, ConvolveToeplitz
from py_toeplitz.cytoeplitz import CyToeplitz

TEST_SIZES = [10 ** i for i in range(3, 5)]


class ScipyToeplitzSuite:
    """Benchmark values for `scipy.linalg.toeplitz`."""
    params = TEST_SIZES
    param_names = ["size"]

    def setup(self, size):
        """Set up the matrix and test vector."""
        self._mat = toeplitz(arange(size, 0, -1))
        self._vec = arange(size)

    def time_dense(self, size):
        """Time the multiply."""
        self._mat.dot(self._vec)

    def peakmem_dense(self, size):
        """Check peak memory usage of multiply.

        Will also include setup.
        """
        self._mat.dot(self._vec)

    def mem_dense(self, size):
        """Check memory size of operator."""
        return self._mat


class PyToeplitzSuite:
    """Benchmark values for `py_toeplitz.Toeplitz`."""
    params = TEST_SIZES
    param_names = ["size"]

    def setup(self, size):
        """Set up the matrix and test vector."""
        self._mat = Toeplitz(arange(size, 0, -1))
        self._vec = arange(size)

    def time_py_nocopy(self, size):
        """Time the multiply."""
        self._mat.dot(self._vec)

    def peakmem_py_nocopy(self, size):
        """Check peak memory usage of multiply.

        Will also include setup.
        """
        self._mat.dot(self._vec)

    def mem_py_nocopy(self, size):
        """Check memory size of operator."""
        return self._mat


class ConvolveToeplitzSuite:
    """Benchmark values for `py_toeplitz.ConvolveToeplitz`."""
    params = TEST_SIZES
    param_names = ["size"]

    def setup(self, size):
        """Set up the matrix and test vector."""
        self._mat = ConvolveToeplitz(arange(size, 0, -1))
        self._vec = arange(size)

    def time_cy_nocopy(self, size):
        """Time the multiply."""
        self._mat.dot(self._vec)

    def peakmem_cy_nocopy(self, size):
        """Check peak memory usage of multiply.

        Will also include setup.
        """
        self._mat.dot(self._vec)

    def mem_cy_nocopy(self, size):
        """Check memory size of operator."""
        return self._mat
