from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *

import numpy as np

from analysis.misc import GenerateAroundSites
from analysis import SiteNetwork, SiteTrajectory
from analysis.visualization import plotter, DEFAULT_COLORS
from analysis.util.elbow import index_of_elbow

from sklearn.decomposition import PCA

import itertools

import pydpc

class SiteTypeAnalysis(object):
    """Cluster sites into types using SOAP and DPCLUS.

    -- sampling_transform --
    Can take either a SiteNetwork or a SiteTrajectory. A "sampling method" must
    be provided -- something with a .run method that will take the input and return
    a SiteNetwork with a higher density of sites (for more SOAP data). Each site in
    this SiteNetwork should have its type set to the index of the site in the original
    input it was generated from.

    NAvgsPerSite and GenerateAroundSites are built for this, the first for SiteTrajectory's
    and the second for SiteNetwork's.

    -- descriptor --
    Some kind of object implementing:
         - n_dim: the number of components in a descriptor vector
         - get_descriptors(pts, out = None): fills and returns out (len(pts), n_dim) with the descriptor
            vectors for the points in pts.

    :param (density, delta) dpc_thresh: The minimum z-score for a point to be
        considered an outlier in the DPCLUS decision plot -- in other words to become
        a cluster. Default (-0.5, 20.0).
    """
    def __init__(self, sampling_transform, descriptor,
                min_pca_variance = 0.9, min_pca_dimensions = 2,
                verbose = True, n_site_types_max = 20):
        self.sampling_transform = sampling_transform
        self.descriptor = descriptor
        self.min_pca_variance = min_pca_variance
        self.min_pca_dimensions = min_pca_dimensions
        self.verbose = verbose
        self.n_site_types_max = n_site_types_max

        self._n_dvecs = None

    def run(self, input):
        if not self._n_dvecs is None:
            raise ValueError("Can't run SiteTypeAnalysis more than once!")

        # -- Sample enough points
        if self.verbose:
            print(" -- Running SiteTypeAnalysis --")
            print("  - Running Sampling Transform")
        sampling = self.sampling_transform.run(input)
        assert isinstance(sampling, SiteNetwork)

        if isinstance(input, SiteNetwork):
            sn = input.copy()
        elif isinstance(input, SiteTrajectory):
            sn = input.site_network.copy()

        # -- Compute descriptor vectors
        if self.verbose:
            print("  - Computing Descriptor Vectors")
        self.dvecs = self.descriptor.get_descriptors(sampling.centers)

        # -- Dimensionality Reduction
        if self.verbose:
            print("  - Clustering Descriptor Vectors")
        self.pca = PCA(self.min_pca_variance)
        pca_dvecs = self.pca.fit_transform(self.dvecs)

        if pca_dvecs.shape[1] < self.min_pca_dimensions:
            if self.verbose:
                print("     PCA accounted for %.0f%% variance in only %i dimensions; less than minimum of %.0f." % (100.0 * np.sum(self.pca.explained_variance_ratio_), pca_dvecs.shape[1], self.min_pca_dimensions))
                print("     Forcing PCA to use %i dimensions." % self.min_pca_dimensions)
                self.pca = PCA(n_components = self.min_pca_dimensions)
                pca_dvecs = self.pca.fit_transform(self.dvecs)

        self.dvecs = pca_dvecs

        if self.verbose:
            print("     Accounted for %.0f%% of variance in %i dimensions" % (100.0 * np.sum(self.pca.explained_variance_ratio_), self.dvecs.shape[1]))

        # -- Do clustering
        # pydpc requires a C-contiguous array
        self.dvecs = np.ascontiguousarray(self.dvecs)
        self.dpc = pydpc.Cluster(self.dvecs, autoplot = False)

        # Estimate the right assignment parameters using the sorted density*delta
        # product plot (Fig. 4 of the paper). The elbow in this plot gives us the
        # number of clusters -- more or less -- and is estimated using the simple
        # elbow finding algorithm of https://stackoverflow.com/questions/2018178/finding-the-best-trade-off-point-on-a-curve
        #
        # This will fail *badly* if there are more than n_site_types_max clusters,
        # but, hopefully, that should never happen?
        decision_curve = self.dpc.density * self.dpc.delta
        decision_curve_sort = np.argsort(decision_curve)[::-1]
        # 1.5 times seems to be enough to resolve the maxium. And the hope here is that we
        # never actually reach the maximum.
        decision_curve = decision_curve[decision_curve_sort[:int(self.n_site_types_max * 1.5)]]
        elbow = index_of_elbow(decision_curve)
        self._decision_curve = decision_curve
        self._decision_elbow = elbow

        if elbow == 0:
            raise ValueError("Something went wrong with the decision curve (elbow == 0); try more data.")

        # a little less than the minimum in both cases. If that lets any more
        # clusters in, they probably deserved to be anyway.
        density_threshold = np.min(self.dpc.density[decision_curve_sort[:elbow]]) - 0.001
        delta_threshold = np.min(self.dpc.delta[decision_curve_sort[:elbow]]) - 0.001

        self._real_dpc_thresh = (density_threshold, delta_threshold)

        self.dpc.assign(density_threshold, delta_threshold)
        assignments = self.dpc.membership
        unassigned = assignments < 0
        self._n_unassigned = np.sum(unassigned)
        self._n_dvecs = len(self.dvecs)
        site_type_counts = np.bincount(assignments[~unassigned])
        self.n_types = len(self.dpc.clusters)

        assert self.n_types == len(site_type_counts), "Got %i types from pydpc, but counted %i" % (self.n_types, len(site_type_counts))

        if self.verbose:
            print("     Found %i site type clusters" % self.n_types )
            print("     Failed to assign %i/%i descriptor vectors to clusters." % (self._n_unassigned, self._n_dvecs))

        # -- Voting
        types = np.empty(shape = sn.n_sites, dtype = np.int)
        self.winning_vote_percentages = np.empty(shape = sn.n_sites, dtype = np.float)
        for site in xrange(sn.n_sites):
            corresponding_samples = sampling.site_types == site
            votes = assignments[corresponding_samples]
            n_votes = len(votes)
            votes = np.bincount(votes[votes >= 0])

            if len(votes) == 0:
                raise ValueError("No votes for site %i; check clustering and descriptors." % site)

            winner = np.argmax(votes)
            self.winning_vote_percentages[site] = float(votes[winner]) / n_votes
            types[site] = winner

        n_sites_of_each_type = np.bincount(types, minlength = self.n_types)
        sn.site_types = types

        if self.verbose:
            print(("             " + "Type {:<2} " * self.n_types).format(*xrange(1, self.n_types + 1)))
            print(("# of sites   " + "{:<8}" * self.n_types).format(*n_sites_of_each_type))

            if np.any(n_sites_of_each_type == 0):
                print("WARNING: Had site types with no sites; check clustering settings/voting!")

        return sn

    @plotter(is3D = False)
    def plot_voting(self, fig = None, ax = None, **kwargs):
        ax.plot(self.winning_vote_percentages)
        ax.set_xlabel("Site")
        ax.set_ylabel("Winner's percentage of votes")
        ax.axhline(0.5, color = 'red', linestyle = '--')
        ax.set_title("Site Type Voting")

    @plotter(is3D = False)
    def plot_dvecs(self, fig = None, ax = None, **kwargs):
        ax.scatter(self.dvecs[:,0], self.dvecs[:,1], s = 3, c = self.dpc.density)
        ax.set_xlabel("(%i%% of variance)" % (100.0 * self.pca.explained_variance_ratio_[0]))
        ax.set_ylabel("(%i%% of variance)" % (100.0 * self.pca.explained_variance_ratio_[1]))
        ax.set_title("Descriptor Vectors")

    @plotter(is3D = False)
    def plot_clustering(self, fig = None, ax = None, **kwargs):
        ccycle = itertools.cycle(DEFAULT_COLORS)
        for cluster in xrange(self.n_types):
            mask = self.dpc.membership == cluster
            dvecs_core = self.dvecs[mask & ~self.dpc.border_member]
            dvecs_border = self.dvecs[mask & self.dpc.border_member]
            color = ccycle.next()
            ax.scatter(dvecs_core[:,0], dvecs_core[:,1], s = 3, color = color, label = "Type %i" % cluster)
            ax.scatter(dvecs_border[:,0], dvecs_border[:,1], s = 3, color = color, alpha = 0.3)

        ax.set_xlabel("(%i%% of variance)" % (100.0 * self.pca.explained_variance_ratio_[0]))
        ax.set_ylabel("(%i%% of variance)" % (100.0 * self.pca.explained_variance_ratio_[1]))
        ax.set_title("Descriptor Clustering")
        ax.legend()

    @plotter(is3D = False)
    def plot_dpc_decision_plot(self, ax = None, **kwargs):
        ax.plot(self._decision_curve, linestyle = '', marker = 'x', label='$\rho_i\delta_i$')
        ax.axvline(self._decision_elbow, color = 'darkgray', label = "Elbow")
        ax.set_title("DPCLUS Decision Curve")

    @plotter(is3D = False)
    def plot_dpc_delta_density(self, ax = None, **kwargs):
        ax.scatter(self.dpc.density, self.dpc.delta)
        ax.set_title("DPCLUS")
        ax.set_xlabel("Density")
        ax.set_ylabel("Delta / a.u.")

        thresh = self._real_dpc_thresh

        ax.plot([thresh[0], self.dpc.density.max()], [thresh[1], thresh[1]], linestyle='--', color = 'darkgray')
        ax.plot([thresh[0], thresh[0]], [thresh[1], self.dpc.delta.max()], linestyle='--', color = 'darkgray')
