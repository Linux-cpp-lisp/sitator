
import matplotlib
from matplotlib.collections import LineCollection

from sitator.util import PBCCalculator
from sitator.visualization import plotter, plot_atoms, plot_points, layers, DEFAULT_COLORS


class SiteTrajectoryPlotter(object):
    """Produce various plots of a ``SiteTrajectory``."""

    @plotter(is3D = True)
    def plot_frame(self, st, frame, **kwargs):
        """Plot sites and instantaneous positions from a given frame.

        Args:
            st (SiteTrajectory)
            frame (int)
        """
        sites_of_frame = np.unique(st._traj[frame])
        frame_sn = st._sn[sites_of_frame]

        frame_sn.plot(**kwargs)

        if not st._real_traj is None:
            mobile_atoms = st._sn.structure.copy()
            del mobile_atoms[~st._sn.mobile_mask]

            mobile_atoms.positions[:] = st._real_traj[frame, st._sn.mobile_mask]
            plot_atoms(atoms = mobile_atoms, **kwargs)

        kwargs['ax'].set_title("Frame %i/%i" % (frame, st.n_frames))


    @plotter(is3D = True)
    def plot_site(self, st, site, **kwargs):
        """Plot all real space positions associated with a site.

        Args:
            st (SiteTrajectory)
            site (int)
        """
        pbcc = PBCCalculator(st._sn.structure.cell)
        pts = st.real_positions_for_site(site).copy()
        offset = pbcc.cell_centroid - pts[3]
        pts += offset
        pbcc.wrap_points(pts)
        lattice_pos = st._sn.static_structure.positions.copy()
        lattice_pos += offset
        pbcc.wrap_points(lattice_pos)
        site_pos = st._sn.centers[site:site+1].copy()
        site_pos += offset
        pbcc.wrap_points(site_pos)
        # Plot point cloud
        plot_points(points = pts, alpha = 0.3, marker = '.', color = 'k', **kwargs)
        # Plot site
        plot_points(points = site_pos, color = 'cyan', **kwargs)
        # Plot everything else
        plot_atoms(st._sn.static_structure, positions = lattice_pos, **kwargs)

        title = "Site %i/%i" % (site, len(st._sn))

        if not st._sn.site_types is None:
            title += " (type %i)" % st._sn.site_types[site]

        kwargs['ax'].set_title(title)


    @plotter(is3D = False)
    def plot_particle_trajectory(self, st, particle, ax = None, fig = None, **kwargs):
        """Plot the sites occupied by a mobile particle over time.

        Args:
            st (SiteTrajectory)
            particle (int)
        """
        types = not st._sn.site_types is None
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

        traj = st._traj[:, particle]
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
                if current_value == -1:
                    c = 'lightgray' # Unknown but reassigned
                elif val == -1:
                    c = 'red' # Uncorrected unknown
                else:
                    c = 'k' # Known
                colors.append(c)

                if types:
                    rxy = (current_segment_start, 0)
                    this_type = st._sn.site_types[val]
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

        ax.set_xlim((0, st.n_frames - 1))
        margin_percent = 0.04
        ymargin = (margin_percent * st._sn.n_sites)
        ax.set_ylim((-ymargin, st._sn.n_sites - 1.0 + ymargin))

        if types:
            typeax.set_xlim((0, st.n_frames - 1))
            typeax.set_ylim((0, type_height))
