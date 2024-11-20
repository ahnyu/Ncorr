from setuptools import setup, Extension
import numpy
import pybind11
import os

# Set the build directory for the compiled Python module
build_dir = os.path.join(os.path.dirname(__file__), 'python')

ext_modules = [
    Extension(
        'ncorr_module',
        sources=[
            'source/ncorr_bindings.cpp',
            'source/counting.cpp',
            'source/grid_particles.cpp',
            'source/utils.cpp',
            # Add other source files as necessary
        ],
        include_dirs=[
            'include',
            numpy.get_include(),
            pybind11.get_include(),
            pybind11.get_include(user=True),
        ],
        language='c++',
        extra_compile_args=['-O3', '-std=c++11', '-fopenmp', '-march=native'],
        extra_link_args=['-fopenmp'],
    ),
]

setup(
    name='ncorr_module',
    version='0.1',
    author='Hanyu Zhang',
    author_email='hanyu.zhang@uwaterloo.ca',
    description='Optimized triangle counting module',
    ext_modules=ext_modules,
    install_requires=['numpy', 'pybind11'],
    # Specify the build directory
    options={
        'build_ext': {
            'build_lib': build_dir,
        }
    },
)
