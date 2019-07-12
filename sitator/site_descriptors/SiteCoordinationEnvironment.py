import numpy as np

from sitator.util.progress import tqdm

try:
    from pymatgen.io.ase import AseAtomsAdaptor
    import pymatgen.analysis.chemenv.coordination_environments.coordination_geometry_finder as cgf
    from pymatgen.analysis.chemenv.coordination_environments.structure_environments import \
        LightStructureEnvironments
    from pymatgen.analysis.chemenv.utils.defs_utils import AdditionalConditions
    has_pymatgen = True
except ImportError:
    has_pymatgen = False

import logging
logger = logging.getLogger(__name__)


class SiteCoordinationEnvironment(object):
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
        # -- Determine local environments
        # Get an ASE structure with a single mobile site that we'll move around
        site_struct, idexes, site_species = sn[0:1].get_structure_with_sites()
        pymat_struct = AseAtomsAdaptor.get_structure(site_struct)
        lgf = cgf.LocalGeometryFinder()
        index = idexes[0]

        coord_envs = []
        vertices = []

        logger.info("Running site coordination environment analysis...")
        # Do this once.
        # __init__ here defaults to disabling structure refinement, so all this
        # method is doing is making a copy of the structure and setting some
        # variables to None.
        lgf.setup_structure(structure = pymat_struct)

        for site in tqdm(range(sn.n_sites), desc = 'Site'):
            # Update the position of the site
            lgf.structure[index].coords = sn.centers[site]
            # Compute structure environments for the site
            struct_envs = lgf.compute_structure_environments(only_indices = [index], **self._kwargs)
            struct_envs = LightStructureEnvironments.from_structure_environments(
                strategy=cgf.LocalGeometryFinder.DEFAULT_STRATEGY,
                structure_environments=struct_envs
            )
            # Store the results
            # We take the first environment for each site since it's the most likely
            coord_envs.append(struct_envs.coordination_environments[index][0])
            vertices.append(
                [n['index'] for n in struct_envs.neighbors_sets[index][0].neighb_indices_and_images]
            )

        del lgf
        del struct_envs

        # -- Postprocess
        # TODO: allow user to ask for full fractional breakdown
        unique_envs = list(set(env['ce_symbol'] for env in coord_envs))
        site_types = np.array([unique_envs.index(env['ce_symbol']) for env in coord_envs])
        # The closer to 1 this is, the better
        site_type_confidences = np.array([env['ce_fraction'] for env in coord_envs])
        coordination_numbers = np.array([int(env['ce_symbol'].split(':')[1]) for env in coord_envs])
        assert np.all(coordination_numbers == [len(v) for v in vertices])

        n_types = len(unique_envs)
        logger.info(("             " + "Type {:<2} " * n_types).format(*range(n_types)))
        logger.info(("# of sites   " + "{:<8}" * n_types).format(*np.bincount(site_types)))

        sn.site_types = site_types
        sn.vertices = vertices
        sn.add_site_attribute("site_type_confidences", site_type_confidences)
        sn.add_site_attribute("coordination_numbers", coordination_numbers)

        return sn
