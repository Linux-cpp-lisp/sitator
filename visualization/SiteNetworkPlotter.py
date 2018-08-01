from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *

import numpy as np

import itertools

import matplotlib
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from analysis.util import PBCCalculator
from analysis.visualization import plotter, plot_atoms, plot_points, layers, DEFAULT_COLORS

class SiteNetworkPlotter(object):
    """Plot a SiteNetwork.

    site_mappings defines how to show different properties. Each entry maps a
    visual aspect ('marker', 'color', 'size') to the name of a site attribute
    including 'site_type'.

    Likewise for edge_mappings, each key maps a visual property ('intensity', 'color',
    'width', 'linestyle') to an edge attribute in the SiteNetwork.

    Note that for edges, the average of the edge property for i -> j and j -> i
    is often used for visual clarity; if your edge properties are not almost symmetric,
    the visualization might not be useful.
    """

    DEFAULT_SITE_MAPPINGS = {
        'marker' : 'site_types',
    }

    DEFAULT_MARKERS = ['x', '+', 'v', '<', '^', '>']
    DEFAULT_LINESTYLES = ['--', ':', '-.', '-']

    def __init__(self,
                site_mappings = DEFAULT_SITE_MAPPINGS,
                edge_mappings = {},
                markers = DEFAULT_MARKERS,
                plot_points_params = {},
                max_linewidth = 4,
                max_edge_alpha = 0.75,
                min_color_threshold = 0.01,
                min_width_threshold = 0.01,
                title = ""):
        self.site_mappings = site_mappings
        self.edge_mappings = edge_mappings
        self.markers = markers
        self.plot_points_params = plot_points_params

        self.max_linewidth = max_linewidth
        self.max_edge_alpha = max_edge_alpha
        self.min_color_threshold = min_color_threshold
        self.min_width_threshold = min_width_threshold

        self.title = title

    @plotter(is3D = True, figsize = (10, 10))
    def __call__(self, sn, *args, **kwargs):
        l = [(plot_atoms,  {'atoms' : sn.static_structure})]
        l += self._site_layers(sn, self.plot_points_params)

        l += self._plot_edges(sn, *args, **kwargs)

        kwargs['ax'].set_title(self.title)

        layers(*l, **kwargs)

    def _site_layers(self, sn, plot_points_params):
        pts_arrays = {'points' : sn.centers}
        pts_params = {}

        # -- Apply mapping
        # - other mappings
        markers = None

        for key in self.site_mappings:
            val = getattr(sn, self.site_mappings[key])
            if key == 'marker':
                markers = val
            elif key == 'color':
                pts_arrays['c'] = val
            elif key == 'size':
                pts_arrays['s'] = val
            else:
                raise KeyError("Unknown mapping `%s`" % key)
        # - markers first
        marker_layers = {}

        if markers is None:
            # Just one layer with all points and one marker
            marker_layers[SiteNetworkPlotter.DEFAULT_MARKERS[0]] = np.ones(shape = sn.n_sites, dtype = np.bool)
        else:
            markers = self._make_discrete(markers)
            unique_markers = np.unique(markers)
            marker_i = 0
            for um in unique_markers:
                marker_layers[SiteNetworkPlotter.DEFAULT_MARKERS[marker_i]] = (markers == um)
                marker_i += 1

        # -- Do plot
        # If no color info provided, a fallback
        if not 'color' in pts_params and not 'c' in pts_arrays:
            pts_params['color'] = 'k'
        # Add user options for `plot_points`
        pts_params.update(plot_points_params)

        pts_layers = []

        for marker in marker_layers:
            d = {'marker' : marker}
            msk = marker_layers[marker]
            for arr in pts_arrays:
                d[arr] = pts_arrays[arr][msk]
            d.update(pts_params)
            pts_layers.append((plot_points, d))

        return pts_layers

    def _plot_edges(self, sn, ax = None, *args, **kwargs):
        if not 'intensity' in self.edge_mappings:
            return []

        pbcc = PBCCalculator(sn.structure.cell)

        n_sites = sn.n_sites
        centers = sn.centers

        # -- Edge attributes
        all_cs = None
        all_linewidths = None
        all_color = None
        # Get value arrays as they exist
        for edgekey in self.edge_mappings:
            edgeval = getattr(sn, self.edge_mappings[edgekey])
            if edgekey == 'intensity':
                all_cs = edgeval
            elif edgekey == 'width':
                all_linewidths = edgeval
            else:
                raise KeyError("Invalid edge mapping key `%s`" % edgekey)

        do_widths = not all_linewidths is None

        # - Normalize
        # Ignore values on the diagonal since we ignore them in the loop
        diag_mask = np.ones(shape = all_cs.shape, dtype = np.bool)
        np.fill_diagonal(diag_mask, False)

        all_cs += np.min(all_cs[diag_mask])
        all_cs /= np.max(all_cs[diag_mask])

        if do_widths:
            all_linewidths += np.min(all_linewidths[diag_mask])
            all_linewidths /= np.max(all_linewidths[diag_mask])

        # -- Construct Line3DCollection segments

        # Whether an edge has already been added
        done_already = np.zeros(shape = (n_sites, n_sites), dtype = np.bool)
        # For the Line3DCollection
        segments = []
        cs = []
        linewidths = []
        # To plot minimum images that are outside unit cell
        sites_to_plot = []
        sites_to_plot_positions = []

        for i in xrange(n_sites):
            for j in xrange(n_sites):
                # No self edges
                if i == j:
                    continue
                # If was already done
                if done_already[i, j]:
                    continue
                # Ignore anything below the threshold
                if all_cs[i, j] < self.min_color_threshold:
                    continue
                if do_widths and all_linewidths[i, j] < self.min_width_threshold:
                    continue

                segment = np.empty(shape = (2, 3), dtype = centers.dtype)
                segment[0] = centers[i]
                segment[1] = centers[j]

                # Modified segment[1] in place
                minimg = pbcc.min_image(segment[0], segment[1])
                was_already_min_img = minimg == 13

                segments.append(segment)

                # If they are eachother's minimum image, then don't bother plotting
                # j -> i
                if was_already_min_img:
                    done_already[j, i] = True
                else:
                    # We'll plot it
                    sites_to_plot.append(j)
                    sites_to_plot_positions.append(segment[1])

                # The mean
                cs.append(np.mean([all_cs[i, j], all_cs[j, i]]))

                if do_widths:
                    linewidths.append(np.mean([all_linewidths[i, j], all_linewidths[j, i]]))

                done_already[i, j] = True

        # -- Construct final Line3DCollection
        assert len(cs) == len(segments)
        print(np.max(cs))
        print(np.max(linewidths))
        lccolors = np.empty(shape = (len(cs), 4), dtype = np.float)
        lccolors[:] = [0.0, 0.1, 1.0, 0.0]
        lccolors[:,3] = np.array(cs) * self.max_edge_alpha

        if do_widths:
            linewidths = np.asarray(linewidths)
            linewidths *= self.max_linewidth
        else:
            linewidths = self.max_linewidth * 0.5

        lc = Line3DCollection(segments, linewidths = linewidths, colors = lccolors, zorder = -20)
        ax.add_collection(lc)

        # -- Plot new sites
        sn2 = sn[sites_to_plot]
        sn2._centers[:] = sites_to_plot_positions

        pts_params = dict(self.plot_points_params)
        pts_params['alpha'] = 0.2
        return self._site_layers(sn2, pts_params)

    def _make_discrete(self, arr):
        return np.round(arr).astype(np.int)
