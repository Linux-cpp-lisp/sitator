import numpy as np

from sitator import SiteTrajectory
from sitator.util import PBCCalculator

from .dpc_helpers import dpc_assign, dpc_compute_vars

class DensityPeakSites(object):
    """
    Args:
        thresh_func (callable): A function to call to determine the delta
            and density cutoffs for cluster centers. The function must have a
            signature:
                func(density, delta, top_gammas, top_gammas_indexes)
            The highest ``func.n_gamma`` gamma values, and the linear indexes
            of the grid boxes that produced them. It must return a tuple of
                (density_threshold, delta_threshold)
    """
    def __init__(self,
                 include_halos = False,
                 n_boxes_max = 100,
                 centers_func = None):

        self.include_halos = include_halos
        self.n_boxes_max = n_boxes_max
        self.thresh_func = centers_func

    def run(self, sn, frames):
        pbcc = PBCCalculator(sn.structure.cell)

        density, point_box_assignments, \
            deltas, delta_targets, gamma_buf, \
            gamma_idex_buf = \
        dpc_compute_vars(
            frames,
            pbcc,
            self.n_boxes_max,
            self.thresh_func.n_gamma,
            self.d_cutoff,
            which_pts = sn.mobile_mask
        )

        # == Determine center cutoffs
        density_threshold, delta_threshold = self.thresh_func(
            density,
            deltas,
            gamma_buf,
            gamma_idex_buf
        )

        # Puts assignments in point_box_assignments in-place
        dpc_assign(
            density,
            point_box_assignments,
            deltas,
            delta_targets,
            density_threshold,
            delta_threshold,
            include_halos = self.include_halos
        )

        st = SiteTrajectory(
            sn,
            point_box_assignments,
        )
        return st

        # Test the updated RemoveUnoccupiedSites
