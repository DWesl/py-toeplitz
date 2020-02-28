===========
py-toeplitz
===========
Implementation of Toeplitz matricies using several algorithms and SciPy's LinearOperators.
Two of the available operators use an implementation that forms a vector of the elements 
of the first row and column and indexes out the subsets corresponding to the rows as 
needed.  One of these operators uses Cython_ and SciPy's Cython wrappers for BLAS to 
speed up the floating-point cases.  Another operator uses `scipy.signal.convolve`_, and 
the last uses NumPy's FFTs.

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
