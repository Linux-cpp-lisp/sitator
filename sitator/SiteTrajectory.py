from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *

import numpy as np

from sitator.util import PBCCalculator
from sitator.visualization import plotter, plot_atoms, plot_points, layers, DEFAULT_COLORS

import matplotlib
from matplotlib.collections import LineCollection

class SiteTrajectory(object):
    """A trajectory capturing the dynamics of particles through a SiteNetwork."""

    SITE_UNKNOWN = -1

    def __init__(self,
                 site_network,
                 particle_assignments,
                 confidences = None):
        """
        :param SiteNetwork site_network:
        :param ndarray (n_frames, n_mobile) particle_assignments:
        :param ndarray (n_frames, n_mobile) confidences (optional): the confidence
            with which each assignment was made.
        """
        if particle_assignments.ndim != 2:
            raise ValueError("particle_assignments must be 2D")
        if particle_assignments.shape[1] != site_network.n_mobile:
            raise ValueError("particle_assignments has wrong shape %s" % particle_assignments.shape)

        self._sn = site_network
        self._traj = particle_assignments.copy()

        if not confidences is None:
            if confidences.shape != particle_assignments.shape:
                raise ValueError("confidences has wrong shape %s; should be %s" % (confidences.shape, particle_assignments.shape))
            self._confs = confidences
        else:
            self._confs = None

        self._real_traj = None

    def __len__(self):
        return self.n_frames

    def __getitem__(self, key):
        st = type(self)(self._sn,
                        self._traj[key],
                        confidences = None if self._confs is None else self._confs[key])
        if not self._real_traj is None:
            st.set_real_traj(self._real_traj[key])

        return st

    @property
    def traj(self):
        """The underlying trajectory."""
        return self._traj

    @property
    def n_frames(self):
        return len(self._traj)

    @property
    def n_unassigned(self):
        return np.sum(self._traj < 0)

    @property
    def n_assigned(self):
        return self._sn.n_mobile * self.n_frames - self.n_unassigned

    @property
    def percent_unassigned(self):
        return float(self.n_unassigned) / (self._sn.n_mobile * self.n_frames)

    @property
    def site_network(self):
        return self._sn

    @site_network.setter
    def site_network(self, value):
        # Captures len, #, and dist.
        assert np.all(value.mobile_mask == self._sn.mobile_mask)
        assert np.all(value.static_mask == self._sn.static_mask)
        self._sn = value

    @property
    def real_trajectory(self):
        return self._real_traj

    def set_real_traj(self, real_traj):
        """Assocaite this SiteTrajectory with a trajectory of points in real space.

        The trajectory is not copied, and should have shape (n_frames, n_total)
        """
        expected_shape = (self.n_frames, self._sn.n_total, 3)
        if not real_traj.shape == expected_shape:
            raise ValueError("real_traj of shape %s does not have expected shape %s" % (real_traj.shape, expected_shape))
        self._real_traj = real_traj

    def remove_real_traj(self):
        """Forget associated real trajectory."""
        del self._real_traj
        self._real_traj = None

    def trajectory_for_particle(self, i, return_confidences = False):
        """Returns the array of sites particle i is assigned to over time."""
        if return_confidences and self._confs is None:
            raise ValueError("This SiteTrajectory has no confidences")
        if return_confidences:
            return self._traj[:, i], self._confs[:, i]
        else:
            return self._traj[:, i]

    def real_positions_for_site(self, site, return_confidences = False):
        if self._real_traj is None:
            raise ValueError("This SiteTrajectory has no real trajectory")
        if return_confidences and self._confs is None:
            raise ValueError("This SiteTrajectory has no confidences")

        assert site < self._sn.n_sites
        msk = self._traj == site
        pts = self._real_traj[:, self._sn.mobile_mask][msk]

        assert pts.shape[1] == 3

        if return_confidences:
            return pts, self._confs[msk].flatten()
        else:
            return pts

    def compute_site_occupancies(self):
        """Computes site occupancies and adds site attribute `occupancies` to site_network."""
        occ = np.true_divide(np.bincount(self._traj[self._traj >= 0]), self.n_frames)
        self.site_network.add_site_attribute('occupancies', occ)
        return occ

    def assign_to_last_known_site(self, frame_threshold = 1, verbose = True):
        """Assign unassigned mobile particles to their last known site within
            `frame_threshold` frames.

        :returns: information dictionary of debugging/diagnostic information.
        """
        total_unknown = self.n_unassigned

        if verbose:
            print("%i unassigned positions (%i%%); assigning unassigned mobile particles to last known positions within %i frames..." % (total_unknown, 100.0 * self.percent_unassigned, frame_threshold))

        last_known = np.empty(shape = self._sn.n_mobile, dtype = np.int)
        last_known.fill(-1)
        time_unknown = np.zeros(shape = self._sn.n_mobile, dtype = np.int)
        avg_time_unknown = 0
        avg_time_unknown_div = 0
        max_time_unknown = 0
        total_reassigned = 0

        for i in range(self.n_frames):
            # All those unknown this frame
            unknown = self._traj[i] == -1
            # Update last_known for assigned sites
            last_known[~unknown] = self._traj[i][~unknown]

            times = time_unknown[~unknown]
            times = times[times != 0]

            if len(times) > 0:
                maxtime = np.max(times)
                if maxtime > frame_threshold:
                    max_time_unknown = maxtime
                avg_time_unknown += np.sum(times)
                avg_time_unknown_div += len(times)

            time_unknown[~unknown] = 0

            to_correct = unknown & (time_unknown < frame_threshold)

            self._traj[i][to_correct] = last_known[to_correct]
            total_reassigned += np.sum(to_correct)
            time_unknown[unknown] += 1

        res = None
        if avg_time_unknown_div > 0: # We corrected some unknowns
            avg_time_unknown = float(avg_time_unknown) / avg_time_unknown_div

            if verbose:
                print("  Maximum # of frames any mobile particle spent unassigned: %i" % max_time_unknown)
                print("  Avg. # of frames spent unassigned: %f" % avg_time_unknown)
                print("  Assigned %i/%i unassigned positions, leaving %i (%i%%) unknown" % (total_reassigned, total_unknown, self.n_unassigned, self.percent_unassigned))

            res = {
                'max_time_unknown' : max_time_unknown,
                'avg_time_unknown' : avg_time_unknown,
                'total_reassigned' : total_reassigned
            }
        else:
            if self.verbose:
                print("  None to correct.")

            res = {
                'max_time_unknown' : 0,
                'avg_time_unknown' : 0,
                'total_reassigned' : 0
            }

        return res

    @plotter(is3D = True)
    def plot_frame(self, frame, **kwargs):
        sites_of_frame = np.unique(self._traj[frame])
        frame_sn = self._sn[sites_of_frame]

        frame_sn.plot(**kwargs)

        if not self._real_traj is None:
            mobile_atoms = self._sn.structure.copy()
            del mobile_atoms[~self._sn.mobile_mask]

            mobile_atoms.positions[:] = self._real_traj[frame, self._sn.mobile_mask]
            plot_atoms(atoms = mobile_atoms, **kwargs)

        kwargs['ax'].set_title("Frame %i/%i" % (frame, self.n_frames))

    @plotter(is3D = True)
    def plot_site(self, site, **kwargs):
        pbcc = PBCCalculator(self._sn.structure.cell)
        pts = self.real_positions_for_site(site).copy()
        offset = pbcc.cell_centroid - pts[3]
        pts += offset
        pbcc.wrap_points(pts)
        lattice_pos = self._sn.static_structure.positions.copy()
        lattice_pos += offset
        pbcc.wrap_points(lattice_pos)
        site_pos = self._sn.centers[site:site+1].copy()
        site_pos += offset
        pbcc.wrap_points(site_pos)
        # Plot point cloud
        plot_points(points = pts, alpha = 0.3, marker = '.', color = 'k', **kwargs)
        # Plot site
        plot_points(points = site_pos, color = 'cyan', **kwargs)
        # Plot everything else
        plot_atoms(self._sn.static_structure, positions = lattice_pos, **kwargs)

        title = "Site %i/%i" % (site, len(self._sn))

        if not self._sn.site_types is None:
            title += " (type %i)" % self._sn.site_types[site]

        kwargs['ax'].set_title(title)

    @plotter(is3D = False)
    def plot_particle_trajectory(self, particle, ax = None, fig = None, **kwargs):
        types = not self._sn.site_types is None
        if types:
            type_height_percent = 0.1
            axpos = ax.get_position()
            typeax_height = type_height_percent * axpos.height
            typeax = fig.add_axes([axpos.x0, axpos.y0, axpos.width, typeax_height], sharex = ax)
            ax.set_position([axpos.x0, axpos.y0 + typeax_height, axpos.width, axpos.height - typeax_height])
            type_height = 1
        # Draw trajectory
        segments = []
        linestyles = []
        colors = []

        traj = self._traj[:, particle]
        current_value = traj[0]
        last_value = traj[0]
        if types:
            last_type = None
        current_segment_start = 0
        puttext = False

        for i, f in enumerate(traj):
            if f != current_value or i == len(traj) - 1:
                val = last_value if current_value == -1 else current_value
                segments.append([[current_segment_start, last_value], [current_segment_start, val], [i, val]])
                linestyles.append(':' if current_value == -1 else '-')
                colors.append('lightgray' if current_value == -1 else 'k')

                if types:
                    rxy = (current_segment_start, 0)
                    this_type = self._sn.site_types[val]
                    typerect = matplotlib.patches.Rectangle(rxy, i - current_segment_start, type_height,
                                                            color = DEFAULT_COLORS[this_type], linewidth = 0)
                    typeax.add_patch(typerect)
                    if this_type != last_type:
                        typeax.annotate("T%i" % this_type,
                                    xy = (rxy[0], rxy[1] + 0.5 * type_height),
                                    xytext = (3, -1),
                                    textcoords = 'offset points',
                                    fontsize = 'xx-small',
                                    va = 'center',
                                    fontweight = 'bold')
                    last_type = this_type

                last_value = val
                current_segment_start = i
                current_value = f

        lc = LineCollection(segments, linestyles = linestyles, colors = colors, linewidth=1.5)
        ax.add_collection(lc)

        if types:
            typeax.set_xlabel("Frame")
            ax.tick_params(axis = 'x', which = 'both', bottom = False, top = False, labelbottom = False)
            typeax.tick_params(axis = 'y', which = 'both', left = False, right = False, labelleft = False)
            typeax.annotate("Type", xy = (0, 0.5), xytext = (-25, 0), xycoords = 'axes fraction', textcoords = 'offset points', va = 'center', fontsize = 'x-small')
        else:
            ax.set_xlabel("Frame")
        ax.set_ylabel("Atom %i's site" % particle)

        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax.grid()

        ax.set_xlim((0, self.n_frames - 1))
        margin_percent = 0.04
        ymargin = (margin_percent * self._sn.n_sites)
        ax.set_ylim((-ymargin, self._sn.n_sites - 1.0 + ymargin))

        if types:
            typeax.set_xlim((0, self.n_frames - 1))
            typeax.set_ylim((0, type_height))
