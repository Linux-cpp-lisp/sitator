import numpy as np

from collections import defaultdict

from sitator import SiteTrajectory
from sitator.dynamics import RemoveUnoccupiedSites

import logging
logger = logging.getLogger(__name__)

class RemoveShortJumps(object):
    """Remove "short" jumps in a SiteTrajectory.

    Remove jumps where the residence at the target is less than some threshold
    and, optionally, only where the mobile atom returns to the site it originally
    jumped from.

    Args:
        only_returning_jumps (bool): If True, only short jumps
            where the mobile atom returns to its initial site will be removed.
    """
    def __init__(self,
                 only_returning_jumps = True,
                 remove_unoccupied_sites = True):
        self.only_returning_jumps = only_returning_jumps
        self.remove_unoccupied_sites = remove_unoccupied_sites


    def run(self,
            st,
            threshold,
            return_stats = False):
        """Returns a copy of ``st`` with short jumps removed.

        Args:
            st (SiteTrajectory): Unassigned considered to be last known.
            threshold (int): The largest number of frames the mobile atom
                can spend at a site while the jump is still considered short.

        Returns:
            A ``SiteTrajectory``.
        """
        n_mobile = st.site_network.n_mobile
        n_frames = st.n_frames
        n_sites = st.site_network.n_sites

        previous_site = np.full(shape = n_mobile, fill_value = -2, dtype = np.int)
        last_known = np.empty(shape = n_mobile, dtype = np.int)
        np.copyto(last_known, st.traj[0])
        # Everything is at it's first position for at least one frame by definition
        time_at_current = np.ones(shape = n_mobile, dtype = np.int)

        framebuf = np.empty(shape = st.traj.shape[1:], dtype = st.traj.dtype)

        out = st.traj.copy()

        n_problems = 0
        n_short_jumps = 0

        # Dict of lists [sum_jump_times, n_short_jumps]
        short_jump_info = defaultdict(lambda: [0, 0])

        for i, frame in enumerate(st.traj):
            if i == 0:
                continue
            # -- Deal with unassigned
            # Don't screw up the SiteTrajectory
            np.copyto(framebuf, frame)
            frame = framebuf

            unassigned = frame == SiteTrajectory.SITE_UNKNOWN
            # Reassign unassigned
            frame[unassigned] = last_known[unassigned]
            fknown = frame >= 0

            if np.any(~fknown):
                logger.warning("At frame %i, %i uncorrectable unassigned particles" % (i, np.sum(~fknown)))
            # -- Update stats

            jumped = (frame != last_known) & fknown
            #problems = last_known[jumped] == -1
            #jumped[np.where(jumped)[0][problems]] = False
            problems = last_known == -1
            jumped[problems] = False
            n_problems += np.sum(problems)

            jump_froms = last_known[jumped]
            jump_tos = frame[jumped]

            # For all that didn't jump, increment time at current
            time_at_current[~jumped] += 1
            # For all that did, check if short
            short_mask = time_at_current[jumped] <= threshold
            if self.only_returning_jumps:
                short_mask &= jump_tos == previous_site[jumped]
            # Remove short jumps
            for sj_atom in np.arange(n_mobile)[jumped][short_mask]:
                # Bookkeeping
                sjkey = (previous_site[sj_atom], last_known[sj_atom], frame[sj_atom])
                short_jump_info[sjkey][0] += time_at_current[sj_atom]
                short_jump_info[sjkey][1] += 1
                n_short_jumps += 1
                # Remove short jump
                out[i - time_at_current[sj_atom]:i, sj_atom] = previous_site[sj_atom]

            previous_site[jumped] = last_known[jumped]

            # Reset for those that jumped
            time_at_current[jumped] = 1

            # Update last known assignment for anything that has one
            last_known[~unassigned] = frame[~unassigned]

        if n_problems != 0:
            logger.warning("Came across %i times where assignment and last known assignment were unassigned." % n_problems)
        logger.info("Removed %i short jumps" % n_short_jumps)
        # Do average
        for k in short_jump_info.keys():
            short_jump_info[k][0] /= short_jump_info[k][1]
        logger.info(
            "Short jump statistics:\n" +
            "\n".join(
                "    removed {1[1]:3}x {0[0]:2} -> {0[1]:2} -> {0[2]:2}; avg. residence at {0[1]:2} of {1[0]} frames".format(
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
