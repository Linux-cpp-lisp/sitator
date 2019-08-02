import numpy as np

from sitator.util import PBCCalculator
from sitator.util.progress import tqdm

import sys

import importlib
import tempfile

from . import helpers
from sitator import SiteNetwork, SiteTrajectory
from sitator.errors import MultipleOccupancyError


import logging
logger = logging.getLogger(__name__)

from functools import wraps
def analysis_result(func):
    @property
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not self._has_run:
            raise ValueError("This LandmarkAnalysis hasn't been run yet.")
        return func(self, *args, **kwargs)
    return wrapper

class LandmarkAnalysis(object):
    """Site analysis of mobile atoms in a static lattice with landmark analysis.

    :param double cutoff_center: The midpoint for the logistic function used
        as the landmark cutoff function. (unitless)
    :param double cutoff_steepness: Steepness of the logistic cutoff function.
    :param double minimum_site_occupancy = 0.1: Minimum occupancy (% of time occupied)
        for a site to qualify as such.
    :param str clustering_algorithm: The landmark clustering algorithm. ``sitator``
        supplies two:
         - ``"dotprod"``: The method described in our "Unsupervised landmark
            analysis for jump detection in molecular dynamics simulations" paper.
         - ``"mcl"``: A newer method we are developing.
    :param dict clustering_params: Parameters for the chosen ``clustering_algorithm``.
    :param str site_centers_method: The method to use for computing the real
        space positions of the sites. Options:
         - ``SITE_CENTERS_REAL_UNWEIGHTED``: A spatial average of all real-space
            mobile atom positions assigned to the site is taken.
         - ``SITE_CENTERS_REAL_WEIGHTED``: A spatial average of all real-space
            mobile atom positions assigned to the site is taken, weighted
            by the confidences with which they assigned to the site.
         - ``SITE_CENTERS_REPRESENTATIVE_LANDMARK``: A spatial average over
            all landmarks' centers is taken, weighted by the representative
            or "typical" landmark vector at the site.
        The "real" methods will generally be more faithful to the simulation,
        but the representative landmark method can work better in cases with
        short trajectories, producing a more "ideal" site location.
    :param bool check_for_zero_landmarks: Whether to check for and raise exceptions
        when all-zero landmark vectors are computed.
    :param float static_movement_threshold: (Angstrom) the maximum allowed
        distance between an instantanous static atom position and it's ideal position.
    :param bool/callable dynamic_lattice_mapping: Whether to dynamically decide
        each frame which static atom represents each average lattice position;
        this allows the LandmarkAnalysis to deal with, say, a rare exchage of
        two static atoms that does not change the structure of the lattice.

        It does NOT allow LandmarkAnalysis to deal with lattices whose structures
        actually change over the course of the trajectory.

        In certain cases this is better delt with by ``MergeSitesByDynamics``.

        If ``False``, no mapping will occur. Otherwise, a callable taking a
        ``SiteNetwork`` should be provided. The callable should return a list
        of static atom indexes that can be validly assigned to each static lattice
        position. If ``True``, ``sitator.landmark.dynamic_mapping.within_species``
        is used.
    :param int max_mobile_per_site: The maximum number of mobile atoms that can
        be assigned to a single site without throwing an error. Regardless of the
        value, assignments of more than one mobile atom to a single site will
        be recorded and reported.

        Setting this to 2 can be necessary for very diffusive, liquid-like
        materials at high temperatures.

        Statistics related to this are reported in ``self.avg_mobile_per_site``
        and ``self.n_multiple_assignments``.
    :param bool force_no_memmap: if True, landmark vectors will be stored only in memory.
        Only useful if access to landmark vectors after the analysis has run is desired.
    :param bool verbose: Verbosity for the ``clustering_algorithm``. Other output
        controlled through ``logging``.
    """

    SITE_CENTERS_REAL_UNWEIGHTED = 'real-unweighted'
    SITE_CENTERS_REAL_WEIGHTED = 'real-weighted'
    SITE_CENTERS_REPRESENTATIVE_LANDMARK = 'representative-landmark'

    CLUSTERING_CLUSTER_SIZE = 'cluster-size'
    CLUSTERING_LABELS = 'cluster-labels'
    CLUSTERING_CONFIDENCES = 'cluster-confs'
    CLUSTERING_LANDMARK_GROUPINGS = 'cluster-landmark-groupings'
    CLUSTERING_REPRESENTATIVE_LANDMARKS = 'cluster-representative-lvecs'

    def __init__(self,
                 clustering_algorithm = 'dotprod',
                 clustering_params = {},
                 cutoff_midpoint = 1.5,
                 cutoff_steepness = 30,
                 minimum_site_occupancy = 0.01,
                 site_centers_method = SITE_CENTERS_REAL_WEIGHTED,
                 check_for_zero_landmarks = True,
                 static_movement_threshold = 1.0,
                 dynamic_lattice_mapping = False,
                 relaxed_lattice_checks = False,
                 max_mobile_per_site = 1,
                 force_no_memmap = False,
                 verbose = True):
        self._cutoff_midpoint = cutoff_midpoint
        self._cutoff_steepness = cutoff_steepness
        self._minimum_site_occupancy = minimum_site_occupancy

        self._cluster_algo = clustering_algorithm
        self._clustering_params = clustering_params

        self.verbose = verbose
        self.check_for_zero_landmarks = check_for_zero_landmarks
        self.site_centers_method = site_centers_method

        if dynamic_lattice_mapping is True:
            from sitator.landmark.dynamic_mapping import within_species
            dynamic_lattice_mapping = within_species
        self.dynamic_lattice_mapping = dynamic_lattice_mapping

        self.relaxed_lattice_checks = relaxed_lattice_checks

        self._landmark_vectors = None
        self._landmark_dimension = None

        self.static_movement_threshold = static_movement_threshold
        self.max_mobile_per_site = max_mobile_per_site

        self.force_no_memmap = force_no_memmap

        self._has_run = False

    @property
    def cutoff(self):
        return self._cutoff

    @analysis_result
    def landmark_vectors(self):
        """Landmark vectors from the last invocation of ``run()``"""
        view = self._landmark_vectors[:]
        view.flags.writeable = False
        return view

    @analysis_result
    def landmark_dimension(self):
        """Number of components in a single landmark vector."""
        return self._landmark_dimension

    def run(self, sn, frames):
        """Run the landmark analysis.

        The input ``SiteNetwork`` is a network of predicted sites; it's sites will
        be used as the "basis" for the landmark vectors.

        Wraps a copy of ``frames`` into the unit cell.

        Args:
            sn (SiteNetwork): The landmark basis. Each site is a landmark defined
                by its vertex static atoms, as indicated by `sn.vertices`.
                (Typically from ``VoronoiSiteGenerator``.)
            frames (ndarray n_frames x n_atoms x 3): A trajectory. Can be unwrapped;
                a copy will be wrapped before the analysis.
        """
        assert isinstance(sn, SiteNetwork)

        if self._has_run:
            raise ValueError("Cannot rerun LandmarkAnalysis!")

        if frames.shape[1:] != (sn.n_total, 3):
            raise ValueError("Wrong shape %s for frames." % (frames.shape,))

        if sn.vertices is None:
            raise ValueError("Input SiteNetwork must have vertices")

        n_frames = len(frames)

        logger.info("--- Running Landmark Analysis ---")

        # Create PBCCalculator
        self._pbcc = PBCCalculator(sn.structure.cell)

        # -- Step 0: Wrap to Unit Cell
        orig_frames = frames # Keep a reference around
        frames = frames.copy()
        # Flatten to list of points for wrapping
        orig_frame_shape = frames.shape
        frames.shape = (orig_frame_shape[0] * orig_frame_shape[1], 3)
        self._pbcc.wrap_points(frames)
        # Back to list of frames
        frames.shape = orig_frame_shape

        # -- Step 1: Compute site-to-vertex distances
        self._landmark_dimension = sn.n_sites

        longest_vert_set = np.max([len(v) for v in sn.vertices])
        verts_np = np.array([np.concatenate((v, [-1] * (longest_vert_set - len(v)))) for v in sn.vertices], dtype = np.int)
        site_vert_dists = np.empty(shape = verts_np.shape, dtype = np.float)
        site_vert_dists.fill(np.nan)

        for i, polyhedron in enumerate(sn.vertices):
            verts_poses = sn.static_structure.get_positions()[polyhedron]
            dists = self._pbcc.distances(sn.centers[i], verts_poses)
            site_vert_dists[i, :len(polyhedron)] = dists

        # -- Step 2: Compute landmark vectors
        logger.info("  - computing landmark vectors -")

        if self.dynamic_lattice_mapping:
            dynmap_compat = self.dynamic_lattice_mapping(sn)
        else:
            # If no dynamic mapping, each is only compatable with itself.
            dynmap_compat = np.arange(sn.n_static)[:, np.newaxis]
        assert len(dynmap_compat) == sn.n_static

        # The dimension of one landmark vector is the number of Voronoi regions
        shape = (n_frames * sn.n_mobile, self._landmark_dimension)

        with tempfile.NamedTemporaryFile() as mmap_backing:
            if self.force_no_memmap:
                self._landmark_vectors = np.empty(shape = shape, dtype = np.float)
            else:
                self._landmark_vectors = np.memmap(mmap_backing.name,
                                                   mode = 'w+',
                                                   dtype = np.float,
                                                   shape = shape)

            helpers._fill_landmark_vectors(self, sn, verts_np, site_vert_dists,
                                            frames,
                                            dynmap_compat = dynmap_compat,
                                            check_for_zeros = self.check_for_zero_landmarks,
                                            tqdm = tqdm, logger = logger)

            if not self.check_for_zero_landmarks and self.n_all_zero_lvecs > 0:
                logger.warning("     Had %i all-zero landmark vectors; no error because `check_for_zero_landmarks = False`." % self.n_all_zero_lvecs)
            elif self.check_for_zero_landmarks:
                assert self.n_all_zero_lvecs == 0

            # -- Step 3: Cluster landmark vectors
            logger.info("  - clustering landmark vectors -")

            #  - Cluster -
            # FIXME: remove reload after development done
            clustermod = importlib.import_module("..cluster." + self._cluster_algo, package = __name__)
            importlib.reload(clustermod)
            cluster_func = clustermod.do_landmark_clustering

            clustering = \
                cluster_func(self._landmark_vectors,
                             clustering_params = self._clustering_params,
                             min_samples = self._minimum_site_occupancy / float(sn.n_mobile),
                             verbose = self.verbose)

        cluster_counts = clustering[LandmarkAnalysis.CLUSTERING_CLUSTER_SIZE]
        lmk_lbls = clustering[LandmarkAnalysis.CLUSTERING_LABELS]
        lmk_confs = clustering[LandmarkAnalysis.CLUSTERING_CONFIDENCES]
        if LandmarkAnalysis.CLUSTERING_LANDMARK_GROUPINGS in clustering:
            landmark_clusters = clustering[LandmarkAnalysis.CLUSTERING_LANDMARK_GROUPINGS]
            assert len(cluster_counts) == len(landmark_clusters)
        else:
            landmark_clusters = None
        if LandmarkAnalysis.CLUSTERING_REPRESENTATIVE_LANDMARKS in clustering:
            rep_lvecs = np.asarray(clustering[LandmarkAnalysis.CLUSTERING_REPRESENTATIVE_LANDMARKS])
            assert rep_lvecs.shape == (len(cluster_counts), self._landmark_vectors.shape[1])
        else:
            rep_lvecs = None

        logging.info("    Failed to assign %i%% of mobile particle positions to sites." % (100.0 * np.sum(lmk_lbls < 0) / float(len(lmk_lbls))))

        # reshape lables and confidences
        lmk_lbls.shape = (n_frames, sn.n_mobile)
        lmk_confs.shape = (n_frames, sn.n_mobile)

        n_sites = len(cluster_counts)

        if n_sites < (sn.n_mobile / self.max_mobile_per_site):
            raise InsufficientSitesError(
                verb = "Landmark analysis",
                n_sites = n_sites,
                n_mobile = sn.n_mobile
            )

        logging.info("    Identified %i sites with assignment counts %s" % (n_sites, cluster_counts))

        # -- Do output
        out_sn = sn.copy()
        # - Compute site centers
        site_centers = np.empty(shape = (n_sites, 3), dtype = frames.dtype)
        if self.site_centers_method == LandmarkAnalysis.SITE_CENTERS_REAL_WEIGHTED or \
           self.site_centers_method == LandmarkAnalysis.SITE_CENTERS_REAL_UNWEIGHTED:
            for site in range(n_sites):
                mask = lmk_lbls == site
                pts = frames[:, sn.mobile_mask][mask]
                if self.site_centers_method == LandmarkAnalysis.SITE_CENTERS_REAL_WEIGHTED:
                    site_centers[site] = self._pbcc.average(pts, weights = lmk_confs[mask])
                else:
                    site_centers[site] = self._pbcc.average(pts)
        elif self.site_centers_method == LandmarkAnalysis.SITE_CENTERS_REPRESENTATIVE_LANDMARK:
            if rep_lvecs is None:
                raise ValueError("Chosen clustering method (with current parameters) didn't return representative landmark vectors; can't use SITE_CENTERS_REPRESENTATIVE_LANDMARK.")
            for site in range(n_sites):
                weights_nonzero = rep_lvecs[site] > 0
                site_centers[site] = self._pbcc.average(
                    sn.centers[weights_nonzero],
                    weights = rep_lvecs[site, weights_nonzero]
                )
        else:
            raise ValueError("Invalid site centers method '%s'" % self.site_centers_method)
        out_sn.centers = site_centers
        # - If clustering gave us that, compute site vertices
        if landmark_clusters is not None:
            vertices = []
            for lclust in landmark_clusters:
                vertices.append(set.union(*[set(sn.vertices[l]) for l in lclust]))
            out_sn.vertices = vertices

        out_st = SiteTrajectory(out_sn, lmk_lbls, lmk_confs)

        # Check that multiple particles are never assigned to one site at the
        # same time, cause that would be wrong.
        self.n_multiple_assignments, self.avg_mobile_per_site = out_st.check_multiple_occupancy(
            max_mobile_per_site = self.max_mobile_per_site
        )

        out_st.set_real_traj(orig_frames)
        self._has_run = True

        return out_st
