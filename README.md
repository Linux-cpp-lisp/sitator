# sitator

A modular framework for conducting and visualizing site analysis of molecular dynamics trajectories.

`sitator` contains an efficient implementation of our method, landmark analysis, as well as visualization tools, generic data structures for site analysis, pre- and post-processing tools, and more.

For details on the method and its application, please see our paper:

> L. Kahle, A. Musaelian, N. Marzari, and B. Kozinsky
> [Unsupervised landmark analysis for jump detection in molecular dynamics simulations](https://doi.org/10.1103/PhysRevMaterials.3.055404)
> Phys. Rev. Materials 3, 055404 – 21 May 2019

If you use `sitator` in your research, please consider citing this paper. The BibTex citation can be found in [`CITATION.bib`](CITATION.bib).

## Installation

`sitator` is currently built for Python 2.7. We recommend the use of a Python 2.7 virtual environment (`virtualenv`, `conda`, etc.). `sitator` has two external dependencies:

 - The `network` executable from [Zeo++](http://www.maciejharanczyk.info/Zeopp/examples.html) is required for computing the Voronoi decomposition. (It does *not* have to be installed in `PATH`; the path to it can be given with the `zeopp_path` option of `VoronoiSiteGenerator`.)
 - If you want to use the site type analysis features, a working installation of [Quippy](https://libatoms.github.io/QUIP/) with [GAP](http://www.libatoms.org/gap/gap_download.html) and Python bindings is required for computing SOAP descriptors.

After downloading, the package is installed with `pip`:

```
# git clone ... OR unzip ... OR ...
cd sitator
pip install .
```

To enable site type analysis, add the `[SiteTypeAnalysis]` option:

```
pip install ".[SiteTypeAnalysis]"
```

## Examples and Documentation

Two example Jupyter notebooks for conducting full landmark analyses of LiAlSiO4 and Li12La3Zr2O12, including data files, can be found [on Materials Cloud](https://archive.materialscloud.org/2019.0008/).

All individual classes and parameters are documented with docstrings in the source code.

## License

This software is made available under the MIT License. See `LICENSE` for more details.
