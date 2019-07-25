# cython: language_level=3

import numpy as np

from sitator import SiteTrajectory
from sitator.dynamics import RemoveUnoccupiedSites

import logging
logger = logging.getLogger(__name__)

ctypedef Py_ssize_t site_int

class SmoothSiteTrajectory(object):
    """"Smooth" a SiteTrajectory by applying a rolling mode.

    For each mobile particle, the assignmet at each frame is replaced by the
    mode of its site assignments over some number of frames centered around it.

    Can be thought of as a discrete lowpass filter.

    The ``set_unassigned_under_threshold`` parameter allows the user to control
    how the smoothing handles "transitions" vs. "attempts"; setting it to True,
    the default, will mark as unassigned transitional moments where neither
    the source nor destination site have a sufficient (``threshold``) majority
    in the window, while setting it to False will maintain the assignment to a
    transitional site.

    Args:
        window_threshold_factor (float): The total width of the rolling window,
            in terms of the threshold.
        remove_unoccupied_sites (bool): If True, sites that are unoccupied after
            the smoothing will be removed.
        set_unassigned_under_threshold (bool): If True, if the multiplicity of
            the mode is less than the threshold, the particle is marked
            unassigned at that frame. If False, the particle's assignment will
            not be modified.
    """
    def __init__(self,
                 window_threshold_factor = 2.1,
                 remove_unoccupied_sites = True,
                 set_unassigned_under_threshold = True):
        self.window_threshold_factor = window_threshold_factor
        self.remove_unoccupied_sites = remove_unoccupied_sites
        self.set_unassigned_under_threshold = set_unassigned_under_threshold

    def run(self,
            st,
            threshold):
        n_mobile = st.site_network.n_mobile
        n_frames = st.n_frames
        n_sites = st.site_network.n_sites

        traj = st.traj
        out = st.traj.copy()

        window = self.window_threshold_factor * threshold
        wleft, wright = int(np.floor(window / 2)), int(np.ceil(window / 2))

        running_windowed_mode(
            traj,
            out,
            wleft,
            wright,
            threshold,
            n_sites,
            self.set_unassigned_under_threshold
        )

        st = st.copy(with_computed = False)
        st._traj = out
        if self.remove_unoccupied_sites:
            # Removing short jumps could have made some sites completely unoccupied
            st = RemoveUnoccupiedSites().run(st)
        st.site_network.clear_attributes()

        return st


cpdef running_windowed_mode(site_int [:, :] traj,
                            site_int [:, :] out,
                            Py_ssize_t wleft,
                            Py_ssize_t wright,
                            Py_ssize_t threshold,
                            Py_ssize_t n_sites,
                            bint replace_no_winner_unknown):
    countbuf_np = np.zeros(shape = n_sites + 1, dtype = np.int)
    cdef Py_ssize_t [:] countbuf = countbuf_np
    cdef Py_ssize_t n_mobile = traj.shape[1]
    cdef Py_ssize_t n_frames = traj.shape[0]
    cdef site_int s_unknown = SiteTrajectory.SITE_UNKNOWN
    cdef site_int winner
    cdef Py_ssize_t best_count

    for mob in range(n_mobile):
        for frame in range(n_frames):
            for wi in range(max(frame - wleft, 0), min(frame + wright, n_frames)):
                countbuf[traj[wi, mob] + 1] += 1
            winner = 0 # THis is actually -1, so unknown by default
            best_count = 0
            for site in range(n_sites + 1):
                if countbuf[site] > best_count:
                    winner = site
                    best_count = countbuf[site]
            if best_count >= threshold:
                out[frame, mob] = winner - 1
            else:
                if replace_no_winner_unknown:
                    out[frame, mob] = s_unknown
                else:
                    out[frame, mob] = traj[frame, mob]
            countbuf[:] = 0
