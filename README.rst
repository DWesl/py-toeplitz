===========
py-toeplitz
===========
Implementation of Toeplitz matricies using several algorithms and SciPy's LinearOperators.

- Two of the available operators use an implementation that forms a vector of the elements 
  of the first row and column and indexes out the subsets corresponding to the rows as 
  needed.  

  - One of these operators uses Cython_ and SciPy's Cython wrappers for BLAS to 
    speed up the floating-point cases.  

- Another operator uses `scipy.signal.convolve`_.

- The last uses NumPy's FFTs.

All implementations should be lower-memory than `scipy.linalg.toeplitz`_, and the last two 
implementations also have algorithmic speedups.  The first two implementations may be 
slightly faster due to cache interaction with the smaller memory footprint, but this effect 
will be small both for matrices small enough to fit entirely in cache and for matrices 
large enough that even the smaller representation doesn't fit in cache.

Related Software
================
- fastmat_
- pyoperators_
- pylops_
- `netlib/toeplitz`_
- `python wrapping of netlib/toeplitz`_
- `scipy.linalg.toeplitz`_

.. _Cython: https://cython.org
.. _fastmat: https://fastmat.readthedocs.io/en/latest/classes/Toeplitz.html
.. _pyoperators: http://pchanial.github.io/pyoperators/2000/doc-operators/#list
.. _pylops: https://pylops.readthedocs.io/en/latest/
.. _netlib/toeplitz: http://netlib.org/toeplitz/
.. _python wrapping of netlib/toeplitz: https://github.com/trichter/toeplitz
.. _scipy.linalg.toeplitz: https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.toeplitz.html
.. _scipy.signal.convolve: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.convolve.html

Performance Comparison
======================

========= ========== ======================== ========== ============ ================== =============
--                                               implementation
--------- --------------------------------------------------------------------------------------------
   size    toeplitz   stride_tricks_toeplitz   Toeplitz   CyToeplitz   ConvolveToeplitz   FFTToeplitz
========= ========== ======================== ========== ============ ================== =============
   8.0     49.7±0μs          140±0μs           395±0μs     571±0μs         430±0μs          582±0μs
   16.0    52.1±0μs          80.0±0μs          282±0μs     511±0μs         483±0μs          589±0μs
   32.0    53.3±0μs          95.0±0μs          475±0μs     559±0μs         508±0μs          590±0μs
   64.0    54.0±0μs          75.0±0μs          420±0μs     380±0μs         354±0μs          411±0μs
  128.0    844±0μs           937±0μs           453±0μs     385±0μs         498±0μs          405±0μs
  256.0    874±0μs           1.92±0ms          1.26±0ms    406±0μs         392±0μs          437±0μs
  512.0    921±0μs           4.01±0ms          2.11±0ms    849±0μs         1.59±0ms         816±0μs
  1024.0   1.58±0ms          18.4±0ms          3.01±0ms    943±0μs         1.74±0ms         634±0μs
  2048.0   2.55±0ms          38.0±0ms          6.07±0ms    1.64±0ms        2.34±0ms         918±0μs
  4096.0   7.29±0ms          152±0ms           10.7±0ms    4.48±0ms        2.98±0ms         148±0ms
  8192.0   28.5±0ms          613±0ms           34.0±0ms    18.9±0ms        4.55±0ms         4.48±0ms
 16384.0   122±0ms           2.34±0s           620±0ms     489±0ms         12.1±0ms         14.9±0ms
========= ========== ======================== ========== ============ ================== =============
