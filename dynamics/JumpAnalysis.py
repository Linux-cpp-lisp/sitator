import numpy as np

import itertools

from analysis import SiteNetwork, SiteTrajectory
from analysis.visualization import plotter, plot_atoms, layers

class JumpAnalysis(object):
    """Given a SiteTrajectory, compute various statistics about the jumps it contains.
    """
    def __init__(self, verbose = True):
        self.verbose = verbose

    def run(self, st):
        assert isinstance(st, SiteTrajectory)

        self._st = st

        n_mobile = st.site_network.n_mobile
        n_frames = st.n_frames
        n_sites = st.site_network.n_sites

        last_known = np.empty(shape = n_mobile, dtype = np.int)
        np.copyto(last_known, st.traj[0])
        # Everything is at it's first position for at least one frame by definition
        time_at_current = np.ones(shape = n_mobile, dtype = np.int)
        total_time_spent_at_site = np.zeros(shape = st.site_network.n_sites, dtype = np.int)

        avg_time_before_jump = np.zeros(shape = (n_sites, n_sites), dtype = np.float)
        avg_time_before_jump_n = np.zeros(shape = avg_time_before_jump.shape, dtype = np.int)
        total_jumps_frame = np.empty(shape = n_frames, dtype = np.int)

        framebuf = np.empty(shape = st.traj.shape[1:], dtype = st.traj.dtype)

        n_problems = 0

        for i, frame in enumerate(st.traj):
            # -- Deal with unassigned
            # Don't screw up the SiteTrajectory
            np.copyto(framebuf, frame)
            frame = framebuf

            unassigned = frame == SiteTrajectory.SITE_UNKNOWN
            # Reassign unassigned
            frame[unassigned] = last_known[unassigned]

            # -- Update stats
            total_time_spent_at_site[frame] += 1

            jumped = frame != last_known
            problems = last_known[jumped] == -1
            jumped[np.where(jumped)[0][problems]] = False
            n_problems += np.sum(problems)

            # Record number of jumps this frame
            total_jumps_frame[i] = np.sum(jumped)

            jump_froms = last_known[jumped]
            jump_tos = frame[jumped]

            avg_time_before_jump[jump_froms, jump_tos] += time_at_current[jumped]
            avg_time_before_jump_n[jump_froms, jump_tos] += 1

            # For all that didn't jump, increment time at current
            time_at_current[~jumped] += 1
            # For all that did, reset it
            time_at_current[jumped] = 1

            # Update last known assignment for anything that has one
            last_known[~unassigned] = frame[~unassigned]

        # The time before jumping to self should always be inf
        assert not np.any(np.nonzero(avg_time_before_jump.diagonal()))

        if self.verbose and n_problems != 0:
            print "Came across %i times where assignment and last known assignment were unassigned." % n_problems

        msk = avg_time_before_jump_n > 0
        # Zeros -- i.e. no jumps -- should actualy be infs
        avg_time_before_jump[~msk] = np.inf
        # Do mean
        avg_time_before_jump[msk] /= avg_time_before_jump_n[msk]

        self.jump_lag = avg_time_before_jump
        self.total_time_spent_at_site = total_time_spent_at_site

    @property
    def jump_lag_by_type(self):
        if self._st.site_network.site_types is None:
            raise ValueError("Associated SiteTrajectory's SiteNetwork has no type information.")

        n_types = self._st.site_network.n_types
        site_types = self._st.site_network.site_types
        all_types = self._st.site_network.types
        outmat = np.empty(shape = (n_types, n_types), dtype = self.jump_lag.dtype)

        for stype_from, stype_to in itertools.product(xrange(len(all_types)), repeat = 2):
            lags = self.jump_lag[site_types == all_types[stype_from]][:, site_types == all_types[stype_to]]
            # Only take things that aren't inf
            lags = lags[lags < np.inf]
            # If there aren't any, then avg is inf
            if len(lags) == 0:
                outmat[stype_from, stype_to] = np.inf
            else:
                outmat[stype_from, stype_to] = np.mean(lags)

        return outmat

    def p_ij(self, i, j):
        p = self.jump_lag[i, j]
        p /= float(self.total_time_spent_at_site[i])
        # p is now probability of i->j given i occupied
        p *= self._st.get_site_occupancies()[i]
        return p

    def get_P_ij(self):
        P = self.jump_lag.copy()
        P /= self.total_time_spent_at_site[:, np.newaxis]
        P *= self._st.get_site_occupancies()[:, np.newaxis]
        return P

    @plotter(is3D = False)
    def plot_jump_lag(self, mode = 'site', ax = None, fig = None, **kwargs):
        if mode == 'site':
            mat = self.jump_lag
        elif mode == 'type':
            mat = self.jump_lag_by_type
        else:
            raise ValueError("`%s` is invalid mode" % mode)

        # Show diagonal
        ax.plot(*zip([0.0, 0.0], mat.shape), color = 'k', alpha = 0.5, linewidth = 1, linestyle = '--')
        ax.grid()

        im = ax.matshow(mat, zorder = 10, cmap = 'plasma')
        ax.set_xlabel("Site B (to)")
        ax.set_ylabel("Site A (from)")

        if mode == 'type':
            # Hack from https://stackoverflow.com/questions/3529666/matplotlib-matshow-labels
            lbls = [''] + ["Type %i" % t for t in self._st.site_network.types]
            ax.set_xticklabels(lbls)
            ax.set_yticklabels(lbls)

        cbar = fig.colorbar(im, ax = ax, extend = 'max')
        cbar.set_label("Avg. time before A->B jump")

        ax.set_title("Jump Lag" if mode == 'site' else "Jump Lag by Type")
