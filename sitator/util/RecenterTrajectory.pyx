import numpy as np

ctypedef double precision

import ase.data

from sitator.util import PBCCalculator

class RecenterTrajectory(object):

    def __init__(self):
        pass

    def run(self, structure, static_mask, positions, velocities = None, masses = None):
        """Recenter a trajectory.

        Recenters ``traj`` on the center of mass of the atoms indicated by
        ``static_mask``, IN PLACE.

        Args:
            structure (ase.Atoms): An atoms representing the structure of the
                simulation.
            static_mask (ndarray): Boolean mask indicating which atoms to recenter on
            positions (ndarray): (n_frames, n_atoms, 3), modified in place
            velocities (ndarray, optional): Same; modified in place if provided
            masses (None or dict or ndarray): The masses to use when computing
                the center of mass.
                 - If ``None``, masses from ``structure.get_masses()`` will
                    be used.
                 - If a ``dict``, expected to map chemical symbols to masses
                 - If an ``ndarray``, must have ``n_atoms`` elements giving the
                    masses of all atoms in the system.
        """

        assert np.any(static_mask), "Static mask all false; there must be static atoms to recenter on."

        factors = static_mask.astype(np.float)
        n_static = np.sum(static_mask)

        # -- Deal with masses
        atomnums = structure.get_atomic_numbers()
        if masses is None:
            # Take standard masses
            mass_arr = structure.get_masses()
        elif isinstance(masses, dict):
            mass_arr = np.zeros(shape = (n_static), dtype = np.float)
            for element in np.unique(atomnums):
                mass_arr[atomnums == element] = masses[ase.data.chemical_symbols[element]]
        elif isinstance(masses, np.ndarray):
            mass_arr = masses
        else:
            raise TypeError("Don't know how to interpret masses `%s`; must be None, dict, or ndarray" % masses)

        # -- Do recentering
        recenter_traj_array(positions, mass_arr, factors)
        # Bring to cell center
        pbcc = PBCCalculator(structure.cell)
        positions += pbcc.cell_centroid

        if not velocities is None:
            recenter_traj_array(velocities, mass_arr, factors)

        return None


cdef recenter_traj_array(precision [:, :, :] array,
                         const precision [:] masses,
                         const precision [:] factors):
    """Recenter any array IN PLACE.

    Args:
        array (ndarray): (n_frames, n_atoms, 3)
        masses (ndarray): (n_atoms)
        factors (ndarray): (n_atoms)
    """

    n_frames, n_atoms, n_dim = array.shape[0], array.shape[1], array.shape[2]

    assert n_dim == 3
    assert len(masses) == n_atoms
    assert len(factors) == n_atoms

    cdef precision total_mass_inverse = 0
    for j in xrange(n_atoms):
        total_mass_inverse += factors[j] * masses[j]
    total_mass_inverse = 1.0 / total_mass_inverse

    cdef precision com[3]

    for i in xrange(n_frames):
        # Compute center of mass this frame
        com[0] = 0.0; com[1] = 0.0; com[2] = 0.0
        for j in xrange(n_atoms):
            for dim in xrange(n_dim):
                com[dim] += total_mass_inverse * factors[j] * masses[j] * array[i, j, dim]

        # Recenter this frame
        for j in xrange(n_atoms):
            for dim in xrange(n_dim):
                array[i, j, dim] -= com[dim]
