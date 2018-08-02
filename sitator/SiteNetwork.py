from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *

import numpy as np

import re

import matplotlib
from sitator.visualization import SiteNetworkPlotter

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
        :param SiteNetworkPlotter plotter: The plotter implementing this SiteNetwork's
            :func:plot method.
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
        # Use a mask to force a copy
        msk = np.ones(shape = self.n_sites, dtype = np.bool)
        return self[msk]

    def __len__(self):
        return self.n_sites

    def __getitem__(self, key):
        sn = type(self)(self.structure,
                        self.static_mask,
                        self.mobile_mask)

        if not self._centers is None:
            sn.centers = self._centers[key]

        if not self._vertices is None:
            sn.vertices = np.asarray(self._vertices)[key].tolist()

        if not self._types is None:
            view = self._types[key]
            sn.site_types = view

        for site_attr in self._site_attrs:
            view = self._site_attrs[site_attr][key]
            sn.add_site_attribute(site_attr, view)

        for edge_attr in self._edge_attrs:
            oldmat = self._edge_attrs[edge_attr]
            newmat = oldmat[key][:, key]
            sn.add_edge_attribute(edge_attr, newmat)

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
        if self._centers is None:
            return 0
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
        if value.shape[1] != 3:
            raise ValueError("`centers` must be a list of points")
        # We reset everything else too, since new centers imply everything changed
        self._vertices = None
        self._types = None
        self._site_attrs = {}
        self._edge_attrs = {}
        # Set centers
        self._centers = value

    def update_centers(self, newcenters):
        """Update the SiteNetwork's centers *without* reseting all other information."""
        if newcenters.shape != self._centers.shape:
            raise ValueError("New `centers` must have same shape as old; try using the setter `.centers = ...`")
        self._centers = newcenters

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

    def has_attribute(self, attr):
        return (attr in self._site_attrs) or (attr in self._edge_attrs)

    def __getattr__(self, attrkey):
        if attrkey in self._site_attrs:
            return self._site_attrs[attrkey]
        elif attrkey in self._edge_attrs:
            return self._edge_attrs[attrkey]
        else:
            raise AttributeError("This SiteNetwork has no site or edge attribute `%s`" % attrkey)

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

        out = {}

        for edgekey in self._edge_attrs:
            out[edgekey] = self._edge_attrs[edgekey][edge]

        return out

    def add_site_attribute(self, name, attr):
        self._check_name(name)
        if not attr.shape[0] == self.n_sites:
            raise ValueError("Attribute array has only %i entries; need one for all %i sites." % (len(attr), self.n_sites))

        self._site_attrs[name] = attr

    def add_edge_attribute(self, name, attr):
        self._check_name(name)
        if not (attr.shape[0] == attr.shape[1] == self.n_sites):
            raise ValueError("Attribute matrix has shape %s; need first two dimensions to be %i" % (attr.shape, self.n_sites))

        self._edge_attrs[name] = attr

    def _check_name(self, name):
        if not SiteNetwork.ATTR_NAME_REGEX.match(name):
            raise ValueError("Attribute name `%s` invalid; must begin with a letter and contain only letters, numbers, and underscores." % name)
        if name in self._edge_attrs or name in self._site_attrs:
            raise KeyError("Attribute with name `%s` already exists" % name)
        if name in self.__dict__:
            raise ValueError("Attribute name `%s` reserved." % name)

    def plot(self, *args, **kwargs):
        """Convenience method -- constructs a defualt SiteNetworkPlotter and calls it."""
        p = SiteNetworkPlotter()
        p(self, *args, **kwargs)
