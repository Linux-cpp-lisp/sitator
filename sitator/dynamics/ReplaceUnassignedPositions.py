import numpy as np

from sitator import SiteTrajectory
from sitator.util import PBCCalculator

import logging
logger = logging.getLogger(__name__)


# See https://stackoverflow.com/questions/1066758/find-length-of-sequences-of-identical-values-in-a-numpy-array-run-length-encodi
def rle(inarray):
        """ run length encoding. Partial credit to R rle function.
            Multi datatype arrays catered for including non Numpy
            returns: tuple (runlengths, startpositions, values) """
        ia = np.asarray(inarray)                  # force numpy
        n = len(ia)
        if n == 0:
            return (None, None, None)
        else:
            y = np.array(ia[1:] != ia[:-1])     # pairwise unequal (string safe)
            i = np.append(np.where(y), n - 1)   # must include last element posi
            z = np.diff(np.append(-1, i))       # run lengths
            p = np.cumsum(np.append(0, z))[:-1] # positions
            return(z, p, ia[i])

class ReplaceUnassignedPositions(object):
    """Fill in missing site assignments in a SiteTrajectory.

    Args:
        replacement_function (callable): Callable that takes
            ``(st, mobile_atom, before_site, start_frame, after_site, end_frame)`` and
            returns either:
             - A single site assignment with which the unassigned will be replaced
             - A timeseries of length ``end_frame - start_frame`` of site
                assignments with which the unassigned will be replaced.
            If ``None``, defaults to
            ``AssignUnassignedPositions.replace_with_from``.
    """
    def __init__(self,
                 replacement_function = None):
        if replacement_function is None:
            replacement_function = RemoveShortJumps.replace_with_last_known
        self.replacement_function = replacement_function

    @staticmethod
    def replace_with_last_known(st, mobile_atom, before_site, start_frame, after_site, end_frame):
        """Replace unassigned with the last known site."""
        return before_site

    @staticmethod
    def replace_with_next_known(st, mobile_atom, before_site, start_frame, after_site, end_frame):
        """Replace unassigned with the next known site."""
        return after_site

    @staticmethod
    def replace_with_closer():
        """Create function to replace unknown with closest site over time.

        Assigns each of the positions during an unassigned run to whichever of
        the before and after sites it is closer to in real space.
        """
        pbcc = None
        ptbuf = np.empty(shape = (2, 3))
        distbuf = np.empty(shape = 2)
        def replace_with_closer(st, mobile_atom, before_site, start_frame, after_site, end_frame):
            if before_site == SiteTrajectory.SITE_UNKNOWN or \
               after_site == SiteTrajectory.SITE_UNKNOWN:
               return SiteTrajectory.SITE_UNKNOWN

            if pbcc is None:
                pbcc = PBCCalculator(st.site_network.structure.cell)
            n_frames = end_frame - start_frame
            out = np.empty(shape = n_frames)
            for i in range(n_frames):
                ptbuf[0] = st.site_network.centers[before_site]
                ptbuf[1] = st.site_network.centers[after_site]
                pbcc.distances(
                    st.real_trajectory[start_frame + i, mobile_atom],
                    ptbuf,
                    in_place = True,
                    out = distbuf
                )
                if distbuf[0] < distbuf[1]:
                    out[i] = before_site
                else:
                    out[i] = after_site
            return out


    def run(self,
            st):
        n_mobile = st.site_network.n_mobile
        n_frames = st.n_frames
        n_sites = st.site_network.n_sites

        traj = st.traj
        out = st.traj.copy()

        for mob in range(n_mobile):
            runlen, start, runsites = rle(traj[:, mob])
            for runi in range(len(runlen)):
                if runsites[runi] == SiteTrajectory.SITE_UNKNOWN:
                    unknown_start = start[runi]
                    unknown_end = unknown_start + runlen[runi]
                    unknown_before = runsites[runi - 1] if runi > 0 else SiteTrajectory.SITE_UNKNOWN
                    unknown_after = runsites[runi + 1] if runi < len(runlen) - 1 else SiteTrajectory.SITE_UNKNOWN
                    replace = self.replacement_function(
                        st, mob,
                        unknown_before, unknown_start,
                        unknown_after, unknown_end
                    )
                    out[unknown_start:unknown_end, mob] = replace

        st = st.copy(with_computed = False)
        st._traj = out

        return st
