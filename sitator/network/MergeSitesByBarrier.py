import numpy as np

import itertools

from scipy.sparse.csgraph import connected_components

from ase.calculators.calculator import all_changes

from sitator.util import PBCCalculator
from sitator.network.merging import MergeSites

import logging
logger = logging.getLogger(__name__)

class MergeSitesByBarrier(MergeSites):
    """Merge sites based on the energy barrier between them.

    Uses a cheap coordinate driving system; this may not be sophisticated enough
    for complex cases. For each pair of sites within the pairwise distance cutoff,
    a linear spatial interpolation is applied to produce `n_driven_images`.
    Two sites are considered mergable if their energies are within
    `final_initial_energy_threshold` and the barrier between them is below
    `barrier_threshold`. The barrier is defined as the maximum image energy minus
    the average of the initial and final energy.

    The energies of the mobile atom are calculated in a static lattice given
    by `coordinating_mask`; if `None`, this is set to the systems `static_mask`.

    For resonable performance, `calculator` should be something simple like
    `ase.calculators.lj.LennardJones`.

    Takes species of first mobile atom as mobile species.

    Args:
        - calculator (ase.Calculator): For computing total potential energies.
        - final_initial_energy_threshold (float, eV): The maximum difference in
            energies between two sites for them to be mergable.
        - barrier_threshold (float, eV): The barrier value above which two sites
            are not mergable.
        - n_driven_images (int, default: None): The number of evenly distributed
            driven images to use.
        - maximum_pairwise_distance (float, Angstrom): The maximum distance
            between two sites for them to be considered for merging.
        - maximum_merge_distance (float, Angstrom): The maxiumum pairwise distance
            among a group of sites chosed to be merged.
    """
    def __init__(self,
                 calculator,
                 final_initial_energy_threshold,
                 barrier_threshold,
                 n_driven_images = None,
                 maximum_pairwise_distance = 2,
                 maximum_merge_distance = 2):
        super().__init__(maximum_merge_distance)
        self.final_initial_energy_threshold = final_initial_energy_threshold
        self.barrier_threshold = barrier_threshold
        self.maximum_pairwise_distance = maximum_pairwise_distance
        self.n_driven_images = n_driven_images
        self.calculator = calculator


    def _get_sites_to_merge(self, st, coordinating_mask = None):
        sn = st.site_network
        pos = sn.centers
        if coordinating_mask is None:
            coordinating_mask = sn.static_mask
        else:
            assert not np.any(coordinating_mask & sn.mobile_mask)
        # -- Build images
        mobile_idex = np.where(sn.mobile_mask)[0][0]
        one_mobile_structure = sn.structure[coordinating_mask]
        one_mobile_structure.extend(sn.structure[mobile_idex])
        mobile_idex = -1
        #images = [one_mobile_structure.copy() for _ in range(self.n_driven_images)]
        interpolation_coeffs = np.linspace(0, 1, self.n_driven_images)
        energies = np.empty(shape = self.n_driven_images)

        # -- Decide on pairs to check
        pbcc = PBCCalculator(sn.structure.cell)
        dists = pbcc.pairwise_distances(pos)
        # At the start, all within distance cutoff are mergable
        mergable = dists <= self.maximum_pairwise_distance

        # -- Check pairs' barriers
        # Symmetric, and diagonal is trivially true. Combinations avoids those cases.
        jbuf = pos[0].copy()
        first_calculate = True
        for i, j in itertools.combinations(range(sn.n_sites)):
            jbuf[:] = pos[j]
            # Get minimage
            _ = pbcc.min_image(pos[i], jbuf)
            # Do coordinate driving
            vector = jbuf - pos[i]
            for image_i in range(self.n_driven_images):
                one_mobile_structure.positions[mobile_idex] = vector
                one_mobile_structure.positions[mobile_idex] *= interpolation_coeffs[image_i]
                one_mobile_structure.positions[mobile_idex] += pos[i]
                energies[image_i] = self.calculator.calculate(atoms = one_mobile_structure,
                                                              properties = ['energy'],
                                                              system_changes = (all_changes if first_calculate else ['positions']))
                first_calculate = False
            # Check barrier
            barrier_idex = np.argmax(energies)
            if np.abs(energies[0] - energies[-1]) > self.final_initial_energy_threshold:
                mergable[i, j] = mergable[j, i] = False
            # Average the initial and final states for a baseline
            baseline_energy = 0.5 * (energies[0] + energies[-1])
            barrier_height = energies[barrier_idex] - baseline_energy
            if barrier_height > self.barrier_threshold:
                mergable[i, j] = mergable[j, i] = False

        # Get mergable groups
        n_merged_sites, labels = connected_components(mergable)
        # MergeSites will check pairwise distances; we just need to make it the
        # right format.
        merge_groups = []
        for lbl in range(n_merged_sites):
            merge_groups.append(np.where(labels == lbl)[0])

        return merge_groups
