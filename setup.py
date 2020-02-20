#!/usr/bin/env python
# ~*~ coding: utf8 ~*~
# cython: embedsignature=True
from setuptools import setup, Extension
from Cython.Build import cythonize

# import numpy as np

with open("VERSION", "r") as in_file:
    with open("src/py_toeplitz/__version__.py", "w") as out_file:
        out_file.write("""\"\"\"Version for the package.\"\"\"
VERSION = "{ver:s}"
""".format(ver=in_file.read()))

setup(
    package_dir={"": "src"},
    ext_modules=cythonize(
        Extension(
            "py_toeplitz.cytoeplitz",
            ["src/py_toeplitz/cytoeplitz.pyx"],
            # include_dirs=[np.get_include()]
        ),
        # include_path=[np.get_include()],
        compiler_directives=dict(embedsignature=True)
    )
)
