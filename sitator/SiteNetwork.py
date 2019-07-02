import numpy as np

import re
import os
import tarfile
import tempfile

import ase.io

import matplotlib
from sitator.visualization import SiteNetworkPlotter

class SiteNetwork(object):
    """A network of mobile particle sites in a static lattice.

    Stores the locations of sites (`centers`), their defining static atoms (`vertices`),
    and their "types" (`site_types`).

    Arbitrary data can also be associated with each site and with each edge
    between sites. Site data can be any array of length n_sites; edge data can be
    any matrix of shape (n_sites, n_sites) where entry i, j is the value for the
    edge from site i to site j.

    Attributes:
        centers (ndarray): (n_sites, 3) coordinates of each site.
        vertices (list, optional): list of lists of indexes of static atoms defining each
            site.
        site_types (ndarray, optional): (n_sites,) values grouping sites into types.
    """

    ATTR_NAME_REGEX = re.compile("^[a-zA-Z][a-zA-Z0-9_]*$")

    def __init__(self,
                 structure,
                 static_mask,
                 mobile_mask):
        """
        Args:
            structure (Atoms): an ASE/Quippy ``Atoms`` containging whatever atoms exist
                in the MD trajectory.
            static_mask (ndarray): Boolean mask indicating which atoms make up the
                host lattice.
            mobile_mask (ndarray): Boolean mask indicating which atoms' movement we
                are interested in.
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
        return list(self._site_attrs.keys())

    @property
    def edge_attributes(self):
        return list(self._edge_attrs.keys())

    def has_attribute(self, attr):
        return (attr in self._site_attrs) or (attr in self._edge_attrs)

    def remove_attribute(self, attr):
        if attr in self._site_attrs:
            del self._site_attrs[attr]
        elif attr in self._edge_attrs:
            del self._edge_attrs[attr]
        else:
            raise AttributeError("This SiteNetwork has no site or edge attribute `%s`" % attr)

    def __getattr__(self, attrkey):
        v = vars(self)
        if '_site_attrs' in v and attrkey in self._site_attrs:
            return self._site_attrs[attrkey]
        elif '_edge_attrs' in v and attrkey in self._edge_attrs:
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
        attr = np.asarray(attr)
        if not attr.shape[0] == self.n_sites:
            raise ValueError("Attribute array has only %i entries; need one for all %i sites." % (len(attr), self.n_sites))

        self._site_attrs[name] = attr

    def add_edge_attribute(self, name, attr):
        self._check_name(name)
        attr = np.asarray(attr)
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
        p = SiteNetworkPlotter(title = "Sites")
        p(self, *args, **kwargs)
