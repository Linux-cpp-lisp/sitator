import numpy as np

from sitator.util.progress import tqdm

try:
    from pymatgen.io.ase import AseAtomsAdaptor
    import pymatgen.analysis.chemenv.coordination_environments.coordination_geometry_finder as cgf
    from pymatgen.analysis.chemenv.coordination_environments.structure_environments import \
        LightStructureEnvironments
    from pymatgen.analysis.chemenv.utils.defs_utils import AdditionalConditions
    from pymatgen.analysis.bond_valence import BVAnalyzer
    has_pymatgen = True
except ImportError:
    has_pymatgen = False

import logging
logger = logging.getLogger(__name__)


class SiteCoordinationEnvironment(object):
    """Determine site types based on local coordination environments.

    Determine site types using the method from the following paper:

        David Waroquiers, Xavier Gonze, Gian-Marco Rignanese,
        Cathrin Welker-Nieuwoudt, Frank Rosowski, Michael Goebel, Stephan Schenk,
        Peter Degelmann, Rute Andre, Robert Glaum, and Geoffroy Hautier

        Statistical analysis of coordination environments in oxides

        Chem. Mater., 2017, 29 (19), pp 8346–8360, DOI: 10.1021/acs.chemmater.7b02766

    as implement in ``pymatgen``'s
    ``pymatgen.analysis.chemenv.coordination_environments``.

    Adds three site attributes:
     - ``coordination_environments``: The name of the coordination environment,
        as returned by ``pymatgen``. Example: ``"T:4"`` (tetrahedral, coordination
        of 4).
     - ``site_type_confidences``: The ``ce_fraction`` of the best match chemical
        environment (from 0 to 1).
     - ``coordination_numbers``: The coordination number of the site.

    Args:
        only_ionic_bonds (bool): If True, assumes ``sn`` is an ``IonicSiteNetwork``
            and uses its charge information to only consider as neighbors
            atoms with compatable (anion and cation, ion and neutral) charges.
        full_chemenv_site_types (bool): If True, ``sitator`` site types on the
            final ``SiteNetwork`` will be assigned based on unique chemical
            environments, including shape. If False, they will be assigned
            solely based on coordination number. Either way, both sets of information
            are included in the ``SiteNetwork``, this just changes which determines
            the ``site_types``.
        **kwargs: passed to ``compute_structure_environments``.
    """
    def __init__(self,
                 only_ionic_bonds = True,
                 full_chemenv_site_types = False,
                 **kwargs):
        if not has_pymatgen:
            raise ImportError("Pymatgen (or a recent enough version including `pymatgen.analysis.chemenv.coordination_environments`) cannot be imported.")
        self._kwargs = kwargs
        self._only_ionic_bonds = only_ionic_bonds
        self._full_chemenv_site_types = full_chemenv_site_types

    def run(self, sn):
        """
        Args:
            sn (SiteNetwork)
        Returns:
            ``sn``, with type information.
        """
        # -- Determine local environments
        # Get an ASE structure with a single mobile site that we'll move around
        if sn.n_sites == 0:
            logger.warning("Site network had no sites.")
            return sn
        site_struct, site_species = sn[0:1].get_structure_with_sites()
        pymat_struct = AseAtomsAdaptor.get_structure(site_struct)
        lgf = cgf.LocalGeometryFinder()
        site_atom_index = len(site_struct) - 1

        coord_envs = []
        vertices = []

        valences = 'undefined'
        if self._only_ionic_bonds:
            valences = list(sn.static_charges) + [sn.mobile_charge]

        logger.info("Running site coordination environment analysis...")
        # Do this once.
        # __init__ here defaults to disabling structure refinement, so all this
        # method is doing is making a copy of the structure and setting some
        # variables to None.
        lgf.setup_structure(structure = pymat_struct)

        for site in tqdm(range(sn.n_sites), desc = 'Site'):
            # Update the position of the site
            lgf.structure[site_atom_index].coords = sn.centers[site]
            # Compute structure environments for the site
            struct_envs = lgf.compute_structure_environments(
                only_indices = [site_atom_index],
                valences = valences,
                additional_conditions = [AdditionalConditions.ONLY_ANION_CATION_BONDS],
                **self._kwargs
            )
            struct_envs = LightStructureEnvironments.from_structure_environments(
                strategy = cgf.LocalGeometryFinder.DEFAULT_STRATEGY,
                structure_environments = struct_envs
            )
            # Store the results
            # We take the first environment for each site since it's the most likely
            this_site_envs = struct_envs.coordination_environments[site_atom_index]
            if len(this_site_envs) > 0:
                coord_envs.append(this_site_envs[0])
                vertices.append(
                    [n['index'] for n in struct_envs.neighbors_sets[site_atom_index][0].neighb_indices_and_images]
                )
            else:
                coord_envs.append({'ce_symbol' : 'BAD:0', 'ce_fraction' : 0.})
                vertices.append([])

        del lgf
        del struct_envs

        # -- Postprocess
        # TODO: allow user to ask for full fractional breakdown
        str_coord_environments = [env['ce_symbol'] for env in coord_envs]
        # The closer to 1 this is, the better
        site_type_confidences = np.array([env['ce_fraction'] for env in coord_envs])
        coordination_numbers = np.array([int(env['ce_symbol'].split(':')[1]) for env in coord_envs])
        assert np.all(coordination_numbers == [len(v) for v in vertices])

        typearr = str_coord_environments if self._full_chemenv_site_types else coordination_numbers
        unique_envs = list(set(typearr))
        site_types = np.array([unique_envs.index(t) for t in typearr])

        n_types = len(unique_envs)
        logger.info(("Type         " + "{:<8}" * n_types).format(*unique_envs))
        logger.info(("# of sites   " + "{:<8}" * n_types).format(*np.bincount(site_types)))

        sn.site_types = site_types
        sn.vertices = vertices
        sn.add_site_attribute("coordination_environments", str_coord_environments)
        sn.add_site_attribute("site_type_confidences", site_type_confidences)
        sn.add_site_attribute("coordination_numbers", coordination_numbers)

        return sn
