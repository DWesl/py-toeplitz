#!/usr/bin/env python
# ~*~ coding: utf8 ~*~
from setuptools import setup
# import numpy as np

with open("VERSION", "r") as in_file:
    with open("src/py_toeplitz/__version__.py", "w") as out_file:
        out_file.write("""\"\"\"Version for the package.\"\"\"
VERSION = "{ver:s}"
""".format(ver=in_file.read()))

setup(
    package_dir={"": "src"},
)
