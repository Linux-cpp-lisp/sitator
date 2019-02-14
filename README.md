# sitator

A modular framework for conducting and visualizing site analysis of molecular dynamics trajectories.

`sitator` contains an efficient implementation of our method, landmark analysis, as well as visualization tools, generic data structures for site analysis, pre- and post-processing tools, and more.

For details on the method and its application, please see our pre-print paper: [Unsupervised landmark analysis for jump detection in molecular dynamics simulations](https://arxiv.org/abs/1902.02107). If you use `sitator` in your research, please consider citing this paper.

## Installation

`sitator` has two external dependencies:

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

## Examples

Two example Jupyter notebooks for conducting full landmark analyses of LiAlSiO4 and Li12La3Zr2O12, including data files, can be found [on Materials Cloud](https://archive.materialscloud.org/2019.0008/).

## License

This software is made available under the MIT License. See `LICENSE` for more details.
