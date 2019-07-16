from setuptools import setup, find_packages
from Cython.Build import cythonize
import numpy as np

setup(name = 'sitator',
      version = '2.0.0',
      description = 'Unsupervised landmark analysis for jump detection in molecular dynamics simulations.',
      download_url = "https://github.com/Linux-cpp-lisp/sitator",
      author = 'Alby Musaelian',
      license = "MIT",
      python_requires = '>=3.2',
      packages = find_packages(),
      ext_modules = cythonize([
        "sitator/landmark/helpers.pyx",
        "sitator/util/*.pyx"
      ]),
      include_dirs=[np.get_include()],
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
      zip_safe = True)
