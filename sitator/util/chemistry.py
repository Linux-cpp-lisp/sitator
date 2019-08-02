import numpy as np

import ase.data
try:
    from ase.formula import Formula
except ImportError:
    from ase.util.formula import Formula

from sitator.util import PBCCalculator

# Key is a central atom, value is a list of the formulas for the rest.
# See, http://www.fccj.us/PolyatomicIons/CompletePolyatomicIonList.htm
DEFAULT_POLYATOMIC_IONS = {
    'Ar' : ['O4', 'O3'],
    'C' : ['N', 'O3', 'O2', 'NO', 'SN'],
    'B' : ['O3','O2'],
    'Br' : ['O4', 'O3', 'O2', 'O'],
    'Fe' : ['O4'],
    'I' : ['O3', 'O4', 'O2', 'O'],
    'Si' : ['O4', 'O3'],
    'S' : ['O5', 'O4', 'O3', 'SO3', 'O2'],
    'Sb' : ['O4', 'O3'],
    'Se' : ['O4', 'O3'],
    'Sn' : ['O3', 'O2'],
    'N' : ['O3', 'O2', 'CO'],
    'Re' : ['O4'],
    'Cl' : ['O4', 'O3', 'O2', 'O'],
    'Mn' : ['O4'],
    'Mo' : ['O4'],
    'Cr' : ['O4', 'CrO7', 'O2'],
    'P' : ['O4', 'O3', 'O2'],
    'Tc' : ['O4'],
    'Te' : ['O4', 'O6', 'O3'],
    'Pb' : ['O3', 'O2'],
    'W' : ['O4']
}

def identify_polyatomic_ions(structure,
                             cutoff_factor = 1.0,
                             ion_definitions = DEFAULT_POLYATOMIC_IONS):
    """Find polyatomic ions in a structure.

    Goes to each potential polyatomic ion center and first checks if the nearest
    neighbor is of a viable species. If it is, all nearest neighbors within the
    maximum possible pairwise summed covalent radii are found and matched against
    the database.

    Args:
        structure (ase.Atoms)
        cutoff_factor (float): Coefficient for the cutoff. Allows tuning just
            how closely the polyatomic ions must be bound. Defaults to 1.0.
        ion_definitions (dict): Database of polyatomic ions where the key is a
            central atom species symbol and the value is a list of the formulas
            for the rest. Defaults to a list of common, non-hydrogen-containing
            polyatomic ions with Oxygen.
    Returns:
        list of tuples, one for each identified polyatomic anion containing its
        formula, the index of the central atom, and the indexes of the remaining
        atoms.
    """
    ion_definitions = {
        k : [Formula(f) for f in v]
        for k, v in ion_definitions.items()
    }
    out = []
    # Go to each possible center atom in data, check that nearest neighbor is
    # of right species, then get all within covalent distances and check if
    # composition matches database...
    pbcc = PBCCalculator(structure.cell)
    dmat = pbcc.pairwise_distances(structure.positions) # Precompute
    np.fill_diagonal(dmat, np.inf)
    for center_i, center_symbol in enumerate(structure.symbols):
        if center_symbol in ion_definitions:
            nn_sym = structure.symbols[np.argmin(dmat[center_i])]
            could_be = [f for f in ion_definitions[center_symbol] if nn_sym in f]
            if len(could_be) == 0:
                # Nearest neighbor isn't even a plausible polyatomic ion species,
                # so skip this potential center.
                continue
            # Take the largest possible other species covalent radius for all
            # other species it possibily could be.
            cutoff = max(
                ase.data.covalent_radii[ase.data.atomic_numbers[other_sym]]
                for form in could_be for other_sym in form
            )
            cutoff += ase.data.covalent_radii[ase.data.atomic_numbers[center_symbol]]
            cutoff *= cutoff_factor
            neighbors = dmat[center_i] <= cutoff
            neighbors = np.where(neighbors)[0]
            neighbor_formula = Formula.from_list(structure.symbols[neighbors])
            it_is = [f for f in could_be if f == neighbor_formula]
            if len(it_is) > 1:
                raise ValueError("Somehow identified single center %s (atom %i) as multiple polyatomic ions %s" % (center_symbol, center_i, it_is))
            elif len(it_is) == 1:
                out.append((
                    center_symbol + neighbor_formula.format('hill'),
                    center_i,
                    neighbors
                ))
    return out
