from setuptools import setup, find_packages
from Cython.Build import cythonize
import numpy as np

setup(name = 'sitator',
      version = '1.0.1',
      description = 'Unsupervised landmark analysis for jump detection in molecular dynamics simulations.',
      download_url = "https://github.com/Linux-cpp-lisp/sitator",
      author = 'Alby Musaelian',
      license = "MIT",
      python_requires = '>=2.7, <3',
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
        "backports.tempfile",
        "future",
        "sklearn"
      ],
      extras_require = {
        "SiteTypeAnalysis" : [
            "pydpc"
        ]
      },
      zip_safe = True)
