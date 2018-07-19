import sys, os

import numpy as np

import itertools

from tqdm import tnrange, tqdm

import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D

import ase
from ase.visualize import view
import ase.neighborlist

from quippy import Atoms as qpAtoms

from SiteNetwork import SiteNetwork

from samos.trajectory import Trajectory
from samos.analysis.jumps.voronoi import collapse_into_unit_cell

class SOAPSiteAnalysis(object):
    def __init__(structure,
                ion_species,
                soap_settings,
                verbose = True):
        self._structure = structure
        self._single_ion_structure = None
        self._ion_species = ion_species
        self._ion_number = ase.data.chemical_symbols.index(self._ion_species)

        self._soap_settings = soap_settings

        self._verbose = verbose
        if verbose:
            self._log = lambda msg: print(msg)
            self._tnrange = tnrange
        else:
            self._log = lambda msg: pass
            self._tnrange = xrange

    def get_single_ion_structure():
        if not self._single_ion_structure is None:
            return self._single_ion_structure
        else:
            self._single_ion_structure = qpAtoms(self._structure)
            self._single_ion_structure.add_atoms((0.0, 0.0, 0.0),
                                                 ase.data.chemical_symbols.index(self._ion_species))
            return self._single_ion_structure

    def get_soap():
        if self._soaper is None:
            self._soaper = descriptors.Descriptor(
                ("soap cutoff={cutoff:0.3f} "
                "cutoff_transition_width={cutoff_transition_width:0.3f} "
                "l_max={l_max:d} n_max={n_max:d} "
                "atom_sigma={sigma:0.3f} n_Z=1 Z=Z{{{z:d}}").format(
                    z = self._ion_number,
                    **self._soap_settings
                ))
        return self._soaper

    def soaps_of_ion_positions(positions):
        struct = self.get_single_ion_structure()

        soaper = self.get_soap()

        soaps = np.empty(shape = (len(positions), soaper.n_dim))

        for i in self._tnrange(len(positions)):
            # Move the fake Li to the site
            struct[-1].position = positions[i]

            # SOAP requires connectivity data to be computed first
            struct.set_cutoff(soaper.cutoff())
            struct.calc_connect()

            #There should only be one descriptor, since there should only be one Li
            soap = soaper.calc(struct)['descriptor']
            assert(len(soap) == 1)

            soaps[i] = soap

        return soaps

    def calc_sites(voronoi_radial = False,
                    site_merge_threshold = None):
        # -- Do a Voronoi decomposition
        self._sn = SiteNetwork.from_lattice(lattice, radial = voronoi_radial)

        if not site_merge_threshold is None:
            sn.collapse_sites(site_merge_threshold, verbose = self._verbose)

        # Zeo++ only returns sites inside the unit cell, so checking for others
        # is unnecessary.

    def calc_site_clusters(pca_dimensions):
        

def get_clustered_ion_sites(structure,
                           ion_species,
                           ,
                           verbose = True):
    """
    :param Atoms structure:
    :param str ion_species:
    :param bool voronoi_radial:
    :param float site_merge_threshold:
    """




def soaps_of_ion_positions(structure,
                           ion_species)
