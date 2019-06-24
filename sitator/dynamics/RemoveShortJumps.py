import numpy as np

from sitator import SiteTrajectory

import logging
logger = logging.getLogger(__name__)

class RemoveShortJumps(object):
    """Remove "short" jumps in a SiteTrajectory.

    Remove jumps where the residence at the target is less than some threshold
    and, optionally, only where the mobile atom returns to the site it originally
    jumped from.

    Args:
        - only_returning_jumps (bool, default: True): If True, only short jumps
            where the mobile atom returns to its initial site will be removed.
    """
    def __init__(self, only_returning_jumps = True):
        self.only_returning_jumps = only_returning_jumps


    def run(self,
            st,
            threshold):
        """Returns a copy of `st` with short jumps removed.

        Args:
            - st (SiteTrajectory): Unassigned considered to be last known.
            - threshold (int): The largest number of frames the mobile atom
                can spend at a site while the jump is still considered short.
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

        for i, frame in enumerate(st.traj):
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
            problems = last_known[jumped] == -1
            jumped[np.where(jumped)[0][problems]] = False
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
                #print("atom %s removing %i -> %i (%i) -> %i" % (sj_atom, previous_site[sj_atom], last_known[sj_atom], time_at_current[sj_atom], frame[sj_atom]))
                n_short_jumps += 1
                out[i - time_at_current[sj_atom]:i+1, sj_atom] = previous_site[sj_atom]

            previous_site[jumped] = last_known[jumped]

            # Reset for those that jumped
            time_at_current[jumped] = 1

            # Update last known assignment for anything that has one
            last_known[~unassigned] = frame[~unassigned]

        if n_problems != 0:
            logger.warning("Came across %i times where assignment and last known assignment were unassigned." % n_problems)
        logger.info("Removed %i short jumps" % n_short_jumps)
        self.n_short_jumps = n_short_jumps

        st = st.copy()
        st._traj = out

        return st
