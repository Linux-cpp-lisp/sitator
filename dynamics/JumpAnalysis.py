import numpy as np

import itertools

from analysis import SiteNetwork, SiteTrajectory
from analysis.visualization import plotter, plot_atoms, layers

class JumpAnalysis(object):
    """Given a SiteTrajectory, compute various statistics about the jumps it contains.

    Adds these edge attributes to the SiteTrajectory's SiteNetwork:
     - `n_ij`: total number of jumps from i to j.
     - `p_ij`: being at i, the probability of jumping to j.
     - `jump_lag`: The average number of frames a particle spends at i before jumping
        to j. Can be +inf if no such jumps every occur.
    And these site attributes:
     - `residence_times`: Avg. number of frames a particle spends at a site before jumping.
    """
    def __init__(self, verbose = True):
        self.verbose = verbose

    def run(self, st):
        """Run the analysis.

        Adds edge attributes to st's SiteNetwork and returns st.
        """
        assert isinstance(st, SiteTrajectory)

        if self.verbose:
            print "Running JumpAnalysis..."

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

        n_ij = np.zeros(shape = (n_sites, n_sites), dtype = np.float)

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
            fknown = frame >= 0

            if np.any(~fknown) and self.verbose:
                print "  at frame %i, %i uncorrectable unassigned particles" % (i, np.sum(~fknown))
            # -- Update stats
            total_time_spent_at_site[frame[fknown]] += 1

            jumped = (frame != last_known) & fknown
            problems = last_known[jumped] == -1
            jumped[np.where(jumped)[0][problems]] = False
            n_problems += np.sum(problems)

            n_ij[frame[fknown], last_known[fknown]] += 1

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

        st.site_network.add_edge_attribute('jump_lag', avg_time_before_jump)
        st.site_network.add_edge_attribute('n_ij', n_ij)
        st.site_network.add_edge_attribute('p_ij', n_ij / total_time_spent_at_site)

        res_times = np.empty(shape = n_sites, dtype = np.float)
        for site in xrange(n_sites):
            times = avg_time_before_jump[site]
            noninf = times < np.inf
            if np.any(noninf):
                res_times[site] = np.mean(times[noninf])
            else:
                res_times[site] = np.inf
        st.site_network.add_site_attribute('residence_times', res_times)

        return st

    def jump_lag_by_type(self, sn):
        """Given a SiteNetwork with jump_lag info, compute it's jump_lag_by_type"""

        if sn.site_types is None:
            raise ValueError("SiteNetwork has no type information.")

        n_types = sn.n_types
        site_types = sn.site_types
        all_types = sn.types
        outmat = np.empty(shape = (n_types, n_types), dtype = sn.jump_lag.dtype)

        for stype_from, stype_to in itertools.product(xrange(len(all_types)), repeat = 2):
            lags = sn.jump_lag[site_types == all_types[stype_from]][:, site_types == all_types[stype_to]]
            # Only take things that aren't inf
            lags = lags[lags < np.inf]
            # If there aren't any, then avg is inf
            if len(lags) == 0:
                outmat[stype_from, stype_to] = np.inf
            else:
                outmat[stype_from, stype_to] = np.mean(lags)

        return outmat

    @plotter(is3D = False)
    def plot_jump_lag(self, sn, mode = 'site', ax = None, fig = None, **kwargs):
        if mode == 'site':
            mat = sn.jump_lag
        elif mode == 'type':
            mat = self.jump_lag_by_type(sn)
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
            lbls = [''] + ["Type %i" % t for t in sn.types]
            ax.set_xticklabels(lbls)
            ax.set_yticklabels(lbls)

        cbar = fig.colorbar(im, ax = ax, extend = 'max')
        cbar.set_label("Avg. time before A->B jump")

        ax.set_title("Jump Lag" if mode == 'site' else "Jump Lag by Type")
