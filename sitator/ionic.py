import numpy as np

from sitator import SiteNetwork

import ase.data

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


class IonicSiteNetwork(SiteNetwork):
    """Site network for a species of mobile charged ions in a static lattice.

    Imposes more restrictions than a plain ``SiteNetwork``:

      - Has a mobile species in place of an arbitrary mobile mask
      - Has a set of static species in place of an arbitraty static mask
      - All atoms of the mobile species must have the same charge

    And contains more information...

    Attributes:
        opposite_ion_mask (ndarray): A mask on ``structure`` indicating all
            anions and neutrally charged atoms if the mobile species is a cation,
            or vice versa if the mobile species is an anion.
        opposite_ion_structure (ase.Atoms): An ``Atoms`` containing the atoms
            indicated by ``opposite_ion_mask``.
        same_ion_mask (ndarray): A mask on ``structure`` indicating all
            atoms whose charge has the same sign as the mobile species.
        same_ion_structure (ase.Atoms): An ``Atoms`` containing the atoms
            indicated by ``same_ion_mask``.
        n_opposite_charge (int): The number of opposite charge static atoms.
        n_same_charge (int): The number of same charge static atoms.

    Args:
        structure (ase.Atoms)
        mobile_species (int): Atomic number of the mobile species.
        static_species (list of int): Atomic numbers of the static species.
        mobile_charge (int): Charge of mobile atoms. If ``None``,
            ``pymatgen``'s ``BVAnalyzer`` will be used to estimate valences.
        static_charges (ndarray int): Charges of the atoms in the static
            structure. If ``None``, ``sitator`` will try to use
            ``pymatgen``'s ``BVAnalyzer`` to estimate valences.
    """
    def __init__(self,
                 structure,
                 mobile_species,
                 static_species,
                 mobile_charge = None,
                 static_charges = None):
        if mobile_species in static_species:
            raise ValueError("Mobile species %i cannot also be one of static species %s" % (mobile_species, static_species))
        mobile_mask = structure.numbers == mobile_species
        static_mask = np.in1d(structure.numbers, static_species)
        super().__init__(
            structure = structure,
            mobile_mask = mobile_mask,
            static_mask = static_mask
        )

        self.mobile_species = mobile_species
        self.static_species = static_species
        # Estimate bond valences if necessary
        if mobile_charge is None or static_charges is None:
            if not has_pymatgen:
                raise ImportError("Pymatgen could not be imported, and is required for guessing charges.")
            sim_struct = AseAtomsAdaptor.get_structure(structure)
            bv = BVAnalyzer()
            struct_valences = np.asarray(bv.get_valences(sim_struct))
            if static_charges is None:
                static_charges = struct_valences[static_mask]
            if mobile_charge is None:
                mob_val = struct_valences[mobile_mask]
                if np.any(mob_val != mob_val[0]):
                    raise ValueError("Mobile atom estimated valences (%s) not uniform; arbitrarily taking first." % mob_val)
                mobile_charge = mob_val[0]
        self.mobile_charge = mobile_charge
        self.static_charges = static_charges

        # Create oposite ion stuff
        mobile_sign = np.sign(mobile_charge)
        static_signs = np.sign(static_charges)
        self.opposite_ion_mask = np.empty_like(static_mask)
        self.opposite_ion_mask.fill(False)
        self.opposite_ion_mask[static_mask] = static_signs != mobile_sign
        self.opposite_ion_structure = structure[self.opposite_ion_mask]

        self.same_ion_mask = np.empty_like(static_mask)
        self.same_ion_mask.fill(False)
        self.same_ion_mask[static_mask] = static_signs == mobile_sign
        self.same_ion_structure = structure[self.same_ion_mask]

    @property
    def n_opposite_charge(self):
        return np.sum(self.opposite_ion_mask)

    @property
    def n_same_charge(self):
        return np.sum(self.same_ion_mask)

    def __str__(self):
        out = super().__str__()
        static_nums = self.static_structure.numbers
        out += (
            "         Mobile species: {:2} (charge {:+d})\n"
            "         Static species: {}\n"
            "      # opposite charge: {}\n"
        ).format(
            ase.data.chemical_symbols[self.mobile_species],
            self.mobile_charge,
            ", ".join(
                "{} (avg. charge {:+.1f})".format(
                    ase.data.chemical_symbols[s],
                    np.mean(self.static_charges[static_nums == s])
                ) for s in self.static_species
            ),
            self.n_opposite_charge
        )
        return out
