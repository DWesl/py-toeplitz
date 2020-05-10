"""Benchmarks comparing the Toeplitz operators.

Run from root directory with `python -m asv dev`.
"""
# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.
from numpy import arange
from scipy.linalg import toeplitz

from py_toeplitz import (
    ConvolveToeplitz, FFTToeplitz,
    # stride_tricks_toeplitz
)
try:
    from py_toeplitz import PyToeplitz
except ImportError:
    from py_toeplitz import Toeplitz as PyToeplitz
from py_toeplitz.cytoeplitz import CyToeplitz

TEST_SIZES = [float(2 ** i) for i in range(3, 15)]
TOEPLITZ_IMPLEMENTATIONS = {
    impl.__name__: impl
    for impl in (
        toeplitz,
        # dot makes everything contiguous, I think.
        # stride_tricks_toeplitz,
        PyToeplitz, CyToeplitz, ConvolveToeplitz, FFTToeplitz
    )
}


class CombinedToeplitzSuite:
    """Benchmark operations for toeplitz implementations."""

    params = (TEST_SIZES, TOEPLITZ_IMPLEMENTATIONS.keys())
    param_names = ("size", "implementation")

    def setup(self, size, impl):
        """Set up the matrix and test vector."""
        self._mat = TOEPLITZ_IMPLEMENTATIONS[impl](arange(size, 0, -1))
        self._vec = arange(size)

    def time_product(self, size, impl):
        """Time the multiply."""
        self._mat.dot(self._vec)

    def peakmem_product(self, size, impl):
        """Check peak memory usage of multiply.

        Will also include setup.
        """
        self._mat.dot(self._vec)

    def mem_copy_operator(self, size, impl):
        """Check memory size of operator."""
        return self._mat
