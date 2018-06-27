#!/usr/bin/env python3

from setuptools import setup, Extension

lib = Extension('poisson3d.support',
                sources=['poisson3d/support.cpp'],
                extra_compile_args=['--std=c++11',
                                    '-fPIC', '-O2', '-lfftw3', '-DNDEBUG'],
                )

setup(name='poisson3d',
      version='1.0.0',
      packages=['poisson3d'],
      install_requires=['numpy', 'pymatgen', 'numba'],
      ext_modules=[lib],
      )
