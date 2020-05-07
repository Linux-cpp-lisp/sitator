from setuptools import setup, find_packages
from Cython.Build import cythonize
import Cython.Compiler
import numpy as np

# Allows cimport'ing PBCCalculator
Cython.Compiler.Options.cimport_from_pyx = True

setup(
    # == Package info ==
    name = 'sitator',
    version = '2.0.0',
    description = 'Unsupervised landmark analysis for jump detection in molecular dynamics simulations.',
    download_url = "https://github.com/Linux-cpp-lisp/sitator",
    author = 'Alby Musaelian',
    license = "MIT",
    python_requires = '>=3.2',
    packages = find_packages(),
    zip_safe = True,
    # == Extensions and Cythons ==
    ext_modules = cythonize(
        [
            "sitator/landmark/helpers.pyx",
            "sitator/util/*.pyx",
            "sitator/dynamics/*.pyx",
            "sitator/misc/*.pyx",
            "sitator/site_analysis/dpc/*.pyx"
        ],
        language_level = 3
    ),
    include_dirs=[np.get_include()],
    # == Dependencies ==
    install_requires = [
        "numpy",
        "scipy",
        "matplotlib",
        "ase",
        "tqdm",
        "sklearn"
    ],
    extras_require = {
        "SiteTypeAnalysis" : [
            "pydpc",
            "dscribe"
        ]
    },
)
