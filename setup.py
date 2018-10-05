from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(name = 'sitator',
      version = '0.1.0',
      description = 'Advanced site analysis for molecular dynamics trajectories.',
      author = 'Alby Musaelian',
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
