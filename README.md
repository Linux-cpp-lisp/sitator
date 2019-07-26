# sitator

A modular framework for conducting and visualizing site analysis of molecular dynamics trajectories.

![](example.png)

<i> Visualizations of complete landmark site analyses, created with `sitator`, of the superionic conductors (a) LGPS, (b) LLZO, and (c) LASO. Source: figures 11, 14, and 18 from our paper, linked below. </i>


`sitator` contains an efficient implementation of our method, landmark analysis, as well as visualization tools, generic data structures for site analysis, pre- and post-processing tools, and more.

For details on landmark analysis and its application, please see our paper:

> L. Kahle, A. Musaelian, N. Marzari, and B. Kozinsky <br/>
> [Unsupervised landmark analysis for jump detection in molecular dynamics simulations](https://doi.org/10.1103/PhysRevMaterials.3.055404) <br/>
> Phys. Rev. Materials 3, 055404 – 21 May 2019

If you use `sitator` in your research, please consider citing this paper. The BibTeX citation can be found in [`CITATION.bib`](CITATION.bib).

## Installation

`sitator` is built for Python >=3.2 (the older version supports Python 2.7). We recommend the use of a virtual environment (`virtualenv`, `conda`, etc.). `sitator` has a number of optional dependencies that enable various features:

* Landmark Analysis
 * The `network` executable from [Zeo++](http://www.maciejharanczyk.info/Zeopp/examples.html) is required for computing the Voronoi decomposition. (It does not have to be installed in `PATH`; the path to it can be given with the `zeopp_path` option of `VoronoiSiteGenerator`.)
* Site Type Analysis
 * For SOAP-based site types: either the `quip` binary from [QUIP](https://libatoms.github.io/QUIP/) with [GAP](http://www.libatoms.org/gap/gap_download.html) **or** the [`DScribe`](https://singroup.github.io/dscribe/index.html) Python library.
    * The Python 2.7 bindings for QUIP (`quippy`) are **not** required. Generally, `DScribe` is much simpler to install than QUIP. **Please note**, however, that the SOAP descriptor vectors **differ** between QUIP and `DScribe` and one or the other may give better results depending on the system you are analyzing.
 * For coordination environment analysis (`sitator.site_descriptors.SiteCoordinationEnvironment`), we integrate the `pymatgen.analysis.chemenv` package; a somewhat recent installation of `pymatgen` is required.

After downloading, the package is installed with `pip`:

```bash
# git clone ... OR unzip ... OR ...
cd sitator
pip install .
```

To enable site type analysis, add the `[SiteTypeAnalysis]` option (this adds two dependencies -- Python packages `pydpc` and `dscribe`):

```bash
pip install ".[SiteTypeAnalysis]"
```

## Examples and Documentation

Two example Jupyter notebooks for conducting full landmark analyses of LiAlSiO<sub>4</sub> and Li<sub>12</sub>La<sub>3</sub>Zr<sub>2</sub>O<sub>12</sub> as in our paper, including data files, can be found [on Materials Cloud](https://archive.materialscloud.org/2019.0008/).

Full API documentation can be found at [ReadTheDocs](https://sitator.readthedocs.io/en/py3/).

`sitator` generally assumes units of femtoseconds for time, Angstroms for space,
and Cartesian (not crystal) coordinates.

## Global Options

`sitator` uses the `tqdm.autonotebook` tool to automatically produce the correct fancy progress bars for terminals and iPython notebooks. To disable all progress bars, run with the environment variable `SITATOR_PROGRESSBAR` set to `false`.

The `SITATOR_ZEO_PATH` and `SITATOR_QUIP_PATH` environment variables can set the default paths to the Zeo++ `network` and QUIP `quip` executables, respectively.

## License

This software is made available under the MIT License. See [`LICENSE`](LICENSE) for more details.
