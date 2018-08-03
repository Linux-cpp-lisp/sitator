from setuptools import setup
from Cython.Build import cythonize

setup(name = 'sitator',
      version = '0.1.0',
      description = 'Advanced site analysis for molecular dynamics trajectories.',
      author = 'Alby Musaelian',
      packages = ['sitator'],
      ext_modules = cythonize([
        "sitator/landmark/helpers.pyx",
        "sitator/util/*.pyx"
      ]),
      install_requires = [
        "numpy",
        "scipy",
        "matplotlib",
        "ase",
        "markov_clustering",
        "tqdm",
        "backports.tempfile"
      ],
      extras_require = {
        "SiteTypeAnalysis" : [
            "pydpc"
        ]
      },
      zip_safe = True)
