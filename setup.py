#!/usr/bin/env python

from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy


extensions = [Extension("batch_gemm",
                        sources=['batch_gemm.pyx'],
                        extra_compile_args=['-fopenmp'],
                        extra_link_args=['-fopenmp'],
                        include_dirs=[numpy.get_include()])]

setup(name='',
      version='0.0',
      description='',
      author='',
      author_email='',
      packages=['batch_gemm'],
      install_requires=['numpy>=1.9.0'],
      keywords=[],
      url='',
      ext_modules=cythonize(extensions),
      )
