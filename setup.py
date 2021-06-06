from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    name='sparseoperations',
    ext_modules=cythonize("sparseoperations.pyx"),
    include_dirs=[numpy.get_include()],
    zip_safe=False,
)
