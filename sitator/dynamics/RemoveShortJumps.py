import numpy as np

from collections import defaultdict

from sitator import SiteTrajectory
from sitator.dynamics import RemoveUnoccupiedSites
from sitator.util import PBCCalculator

import logging
logger = logging.getLogger(__name__)

# From https://stackoverflow.com/questions/1066758/find-length-of-sequences-of-identical-values-in-a-numpy-array-run-length-encodi
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
        return (z, p, ia[i])


class RemoveShortJumps(object):
    """Remove "short" jumps in a SiteTrajectory.

    Remove jumps where the residence at the target is less than some threshold
    and, optionally, only where the mobile atom returns to the site it originally
    jumped from.

    It only counts as a short jump if

    Args:
        only_returning_jumps (bool): If True, only short jumps
            where the mobile atom returns to its initial site will be removed.
        remove_unoccupied_sites (bool): If True, sites that are unoccupied after
            removing short jumps will be removed.
        replacement_function (callable): Callable that takes
            ``(st, mobile_atom, from_site, start_frame, to_site, end_frame)`` and
            returns either:
             - A single site assignment with which the short jump will be replaced
             - A timeseries of length ``end_frame - start_frame`` of site
                assignments with which the short jump will be replaced.
            If ``None``, defaults to ``RemoveShortJumps.replace_with_from``.
    """
    def __init__(self,
                 only_returning_jumps = True,
                 remove_unoccupied_sites = True,
                 replacement_function = None):
        self.only_returning_jumps = only_returning_jumps
        self.remove_unoccupied_sites = remove_unoccupied_sites
        if replacement_function is None:
            replacement_function = RemoveShortJumps.replace_with_from
        self.replacement_function = replacement_function

    @staticmethod
    def replace_with_from(st, mobile_atom, from_site, start_frame, to_site, end_frame):
        """Replace a short jump with the site being jumped from."""
        return from_site

    @staticmethod
    def replace_with_to(st, mobile_atom, from_site, start_frame, to_site, end_frame):
        """Replace a short jump with the site being jumped to after the short jump."""
        return to_site

    @staticmethod
    def replace_with_unknown(st, mobile_atom, from_site, start_frame, to_site, end_frame):
        """Mark as unassigned during a short jump."""
        return SiteTrajectory.SITE_UNKNOWN

    @staticmethod
    def replace_with_closer():
        """Create function to replace short jump with closest site over time.

        Assigns the positions during a short jump to whichever of the from
        and to site it is closer to in real space.
        """
        pbcc = None
        ptbuf = np.empty(shape = (2, 3))
        distbuf = np.empty(shape = 2)
        def replace_with_closer(st, mobile_atom, from_site, start_frame, to_site, end_frame):
            if pbcc is None:
                pbcc = PBCCalculator(st.site_network.structure.cell)
            n_frames = end_frame - start_frame
            out = np.empty(shape = n_frames)
            for i in range(n_frames):
                ptbuf[0] = st.site_network.centers[from_site]
                ptbuf[1] = st.site_network.centers[to_site]
                pbcc.distances(
                    st.real_trajectory[start_frame + i, mobile_atom],
                    ptbuf,
                    in_place = True,
                    out = distbuf
                )
                if distbuf[0] < distbuf[1]:
                    out[i] = from_site
                else:
                    out[i] = to_site
            return out

    def run(self,
            st,
            threshold,
            return_stats = False):
        n_mobile = st.site_network.n_mobile
        n_frames = st.n_frames
        n_sites = st.site_network.n_sites

        st_no_un = st.copy(with_computed = False)
        st_no_un.assign_to_last_known_site(frame_threshold = np.inf)

        traj = st_no_un.traj
        out = st.traj.copy()

        # Dict of lists [sum_jump_times, n_short_jumps]
        short_jump_info = defaultdict(lambda: [0, 0])

        for mob in range(n_mobile):
            runlen, start, runsites = rle(traj[:, mob])
            # We pretend that the first and last run extend into infinity
            # Think Fourier transforms
            runlen[[0, -1]] = np.iinfo(runlen.dtype).max
            last_long_enough = 0
            short_start = None
            short_from = None
            short_transitionals = None
            for runi in range(0, len(runlen)):
                shortrun = runlen[runi] < threshold
                if shortrun:
                    if short_start is None:
                        short_start = start[runi]
                        short_from = runsites[runi - 1]
                        short_transitionals = [runsites[runi]]
                    else:
                        short_transitionals.append(runsites[runi])
                elif short_start is not None:
                    # Process short jump
                    short_to = runsites[runi]
                    short_end = start[runi]
                    # If we're only doing returning jumps, check that
                    do = (not self.only_returning_jumps) or (short_to == short_from)
                    if do:
                        sjkey = (short_from, tuple(short_transitionals), short_to)
                        short_jump_info[sjkey][0] += short_end - short_start
                        short_jump_info[sjkey][1] += 1
                        replace = self.replacement_function(
                            st, mob,
                            short_from, short_start,
                            short_to, short_end
                        )
                        out[short_start:short_end, mob] = replace
                    # Reset
                    short_start = None

        # Do average
        for k in short_jump_info.keys():
            short_jump_info[k][0] /= short_jump_info[k][1]
        logger.info(
            "Short jump statistics:\n" +
            "\n".join(
                "    removed {1[1]:3}x  {0[0]:2} -> {0[1]} -> {0[2]:2}; spent {1[0]:.1f} frames at {0[1]}".format(
                    k, v
                ) for k, v in short_jump_info.items()
            )
        )

        st = st.copy(with_computed = False)
        st._traj = out
        if self.remove_unoccupied_sites:
            # Removing short jumps could have made some sites completely unoccupied
            st = RemoveUnoccupiedSites().run(st)
        st.site_network.clear_attributes()

        if return_stats:
            return st, short_jump_info
        else:
            return st
