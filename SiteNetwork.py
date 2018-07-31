from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *

import numpy as np

import re

import matplotlib
from analysis.visualization import plotter, plot_atoms, plot_points, layers, DEFAULT_COLORS

class SiteNetwork(object):
    """A network of sites for some diffusive/mobile particle in a static lattice.

    Stores the locations of sites, their defining static atoms, and their "types".

    Arbitrary data can also be associated with each site and with each edge.
    Site data is any area of length n_sites; edge data is any matrix of shape
    (n_sites, n_sites) where entry i, j is the value for the edge from site i to
    site j.
    """

    ATTR_NAME_REGEX = re.compile("^[a-zA-Z][a-zA-Z0-9_]*$")

    def __init__(self,
                 structure,
                 static_mask,
                 mobile_mask):
        """
        :param Atoms structure: an ASE/Quippy :class:Atoms object containing the structure simulated in the trajectory
          Should be a representative/ideal/thermal-average structure.
        :param ndarray(bool) static_mask: Boolean mask indicating which atoms to consider immobile
        :param ndarray(bool) mobile_mask: Boolean mask indicating which atoms to track
        """

        assert static_mask.ndim == mobile_mask.ndim == 1, "The masks must be one-dimensional"
        assert len(structure) == len(static_mask) == len(mobile_mask), "The masks must have the same length as the # of atoms in the strucutre."

        # No overlap
        assert not np.any(static_mask & mobile_mask), "static_mask and mobile_mask cannot overlap."

        self.structure = structure
        self.static_mask = static_mask
        self.n_static = np.sum(static_mask)
        self.mobile_mask = mobile_mask
        self.n_mobile = np.sum(mobile_mask)

        # Create static structure
        self.static_structure = structure.copy()
        del self.static_structure[(~static_mask) | mobile_mask]
        assert len(self.static_structure) == self.n_static

        # Set variables
        self._centers = None
        self._vertices = None
        self._types = None

        self._site_attrs = {}
        self._edge_attrs = {}

    def copy(self):
        """Returns a (shallowish) copy of self."""
        sn = type(self)(self.structure,
                        self.static_mask,
                        self.mobile_mask)

        if not self._centers is None:
            sn.centers = self._centers.copy()
            if not self._vertices is None:
                sn.vertices = list(self._vertices)
            if not self._types is None:
                sn.site_types = self._types.copy()

        return sn

    def __len__(self):
        return self.n_sites

    def __getitem__(self, key):
        if self._centers is None:
            raise ValueError("This SiteNetwork has no sites; can't slice.")

        mask = np.zeros(shape = len(self), dtype = np.bool)
        mask[key] = True # This will deal with wrong shapes and all kinds of fancy indexing

        sn = type(self)(self.structure,
                        self.static_mask,
                        self.mobile_mask)

        view = self._centers[mask]
        view.flags.writeable = False
        sn.centers = view

        if not self._vertices is None:
            sn.vertices = [v for i, v in enumerate(self._vertices) if mask[i]]

        if not self._types is None:
            view = self._types[mask]
            view.flags.writeable = False
            sn.site_types = view

        return sn

    def of_type(self, stype):
        """Returns a "view" to this SiteNetwork with only sites of a certain type."""
        if self._types is None:
            raise ValueError("This SiteNetwork has no type information.")

        if not stype in self._types:
            raise ValueError("This SiteNetwork has no sites of type %i" % stype)

        return self[self._types == stype]

    @property
    def n_sites(self):
        return len(self._centers)

    @property
    def n_total(self):
        return len(self.static_mask)

    @property
    def centers(self):
        view = self._centers.view()
        view.flags.writeable = False
        return view

    @centers.setter
    def centers(self, value):
        self._centers = value
        self._vertices = None
        self._types = None

    @property
    def vertices(self):
        return self._vertices

    @vertices.setter
    def vertices(self, value):
        if not len(value) == len(self._centers):
            raise ValueError("Wrong # of vertices %i; expected %i" % (len(value), len(self._centers)))
        self._vertices = value

    @property
    def site_types(self):
        if self._types is None:
            return None
        view = self._types.view()
        view.flags.writeable = False
        return view

    @site_types.setter
    def site_types(self, value):
        if not value.shape == (len(self._centers),):
            raise ValueError("Wrong # of types %i; expected %i" % (value.shape, len(self._centers)))
        self._types = value

    @property
    def n_types(self):
        return len(np.unique(self.site_types))

    @property
    def types(self):
        return np.unique(self.site_types)

    @property
    def site_attributes(self):
        return self._site_attrs.keys()

    @property
    def edge_attributes(self):
        return self._edge_attrs.keys()

    def __getattr__(self, attrkey):
        if attrkey in self._site_attrs:
            return self._site_attrs[attrkey]
        elif attrkey in self._edge_attrs:
            return self._edge_attrs[attrkey]
        else:
            raise KeyError("This SiteNetwork has no site or edge attribute `%s`" % attrkey)

    def get_site(self, site):
        """Get all info about a site.

        :param int site:

        :returns dict:
        """
        out = {
            'center' : self.centers[site]
        }
        if not self._vertices is None:
            out['vertices'] = self._vertices[site]
        if not self._types is None:
            out['type'] = self._types[site]

        for attrkey in self._site_attrs:
            out[attrkey] = self._site_attrs[attrkey][site]

        return out

    def get_edge(self, edge):
        """Get all info for the edge identified by (i, j)

        :param tuple edge: which edge to get data on.

        :returns dict:
        """
        if len(self._edge_attrs) == 0:
            raise ValueError("This SiteNetwork has no edge attributes")

    def add_site_attribute(self, name, attr):
        self._check_name(name)
        if not attr.shape[0] == self.n_sites:
            raise ValueError("Attribute array has only %i entries; need one for all %i sites." % (len(attr), self.n_sites))

        self._site_attrs[name] = attr

    def add_edge_attribute(self, name, attr):
        self._check_name(name)
        if not (attr.shape[0] == attr.shape[1] == self.n_sites):
            raise ValueError("Attribute matrix has shape; need first two dimensions to be %i" % (attr.shape, self.n_sites))

        self._edge_attrs[name] = attr

    def _check_name(self, name):
        if not SiteNetwork.ATTR_NAME_REGEX.match(name):
            raise ValueError("Attribute name `%s` invalid; must begin with a letter and contain only letters, numbers, and underscores." % name)
        if name in self._edge_attrs or name in self._site_attrs:
            raise KeyError("Attribute with name `%s` already exists" % name)
        if name in self.__dict__:
            raise ValueError("Attribute name `%s` reserved." % name)

    DEFAULT_MAPPINGS = {
        'marker' : 'site_types',
    }

    DEFAULT_MARKERS = ['x', 'D', '+', 'v', '<', '^', '>']

    @plotter(is3D = True)
    def plot(self, mappings = DEFAULT_MAPPINGS, plot_points_params = {}, **kwargs):
        """Plot the SiteNetwork.

        mappings defines how to show different properties. Each entry maps a
        visual aspect ('marker', 'color', 'size') to the name of a site attribute
        including 'site_type'.

        The type for marker must be integral; the others can be integer or float.
        """
        pts_arrays = {'points' : self.centers}
        pts_params = {}

        # -- Apply mapping
        # - other mappings
        markers = None

        for key in mappings:
            val = getattr(self, mappings[key])
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
            marker_layers[SiteNetwork.DEFAULT_MARKERS[0]] = np.ones(shape = self.n_sites, dtype = np.bool)
        else:
            unique_markers = np.unique(np.round(markers))
            marker_i = 0
            for um in unique_markers:
                marker_layers[SiteNetwork.DEFAULT_MARKERS[marker_i]] = (markers == um)
                marker_i += 1

        # -- Do plot

        if not 'color' in kwargs and not 'c' in pts_arrays:
            pts_params['color'] = 'k'

        pts_params.update(plot_points_params)

        pts_layers = []

        for marker in marker_layers:
            d = {'marker' : marker}
            msk = marker_layers[marker]
            for arr in pts_arrays:
                d[arr] = pts_arrays[arr][msk]
            d.update(pts_params)
            pts_layers.append((plot_points, d))

        layers((plot_atoms,  {'atoms' : self.static_structure}),
               *pts_layers, **kwargs)
