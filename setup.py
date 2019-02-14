from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(name = 'sitator',
      version = '1.0.0',
      description = 'Unsupervised landmark analysis for jump detection in molecular dynamics simulations.',
      download_url = "https://github.com/Linux-cpp-lisp/sitator",
      author = 'Alby Musaelian',
      license = "MIT",
      packages = ['sitator'],
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
