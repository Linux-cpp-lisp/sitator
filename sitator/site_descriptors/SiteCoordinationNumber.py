import numpy as np

try:
    from pymatgen.io.ase import AseAtomsAdaptor
    import pymatgen.analysis.chemenv.coordination_environments.coordination_geometry_finder as cgf
    has_pymatgen = True
except ImportError:
    has_pymatgen = False


class SiteCoordinationAnalysis(object):
    """Determine site types based on local coordination environments.

    Determine site types using the method from the following paper:

        David Waroquiers, Xavier Gonze, Gian-Marco Rignanese, Cathrin Welker-Nieuwoudt, Frank Rosowski, Michael Goebel, Stephan Schenk, Peter Degelmann, Rute Andre, Robert Glaum, and Geoffroy Hautier,
        “Statistical analysis of coordination environments in oxides”,
        Chem. Mater., 2017, 29 (19), pp 8346–8360, DOI: 10.1021/acs.chemmater.7b02766

    as implement in `pymatgen`'s `pymatgen.analysis.chemenv.coordination_environments`.

    Args:
        **kwargs: passed to `compute_structure_environments`.
    """
    def __init__(self, **kwargs):
        if not has_pymatgen:
            raise ImportError("Pymatgen (or a recent enough version including `pymatgen.analysis.chemenv.coordination_environments`) cannot be imported.")
        self._kwargs = kwargs

    def run(self, sn):
        site_struct, site_species = sn.get_structure_with_sites()
        pymat_struct = AseAtomsAdaptor.get_structure(site_struct)
        lgf = cgf.LocalGeometryFinder()
        struct_envs = lgf.compute_structure_environments(
            structure = pymat_struct,
            indicies = np.where(sn.mobile_mask)[0],
            only_cations = False,
        )
