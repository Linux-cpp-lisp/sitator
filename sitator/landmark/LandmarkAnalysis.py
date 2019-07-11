import numpy as np

from sitator.util import PBCCalculator
from sitator.util.progress import tqdm

import sys

import importlib
import tempfile

from . import helpers
from sitator import SiteNetwork, SiteTrajectory
from .errors import MultipleOccupancyError

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
    """Track a mobile species through a fixed lattice using landmark vectors."""

    def __init__(self,
                 clustering_algorithm = 'dotprod',
                 clustering_params = {},
                 cutoff_midpoint = 1.5,
                 cutoff_steepness = 30,
                 minimum_site_occupancy = 0.01,
                 peak_evening = 'none',
                 weighted_site_positions = True,
                 check_for_zero_landmarks = True,
                 static_movement_threshold = 1.0,
                 dynamic_lattice_mapping = False,
                 relaxed_lattice_checks = False,
                 max_mobile_per_site = 1,
                 force_no_memmap = False,
                 verbose = True):
        """
        :param double cutoff_center: The midpoint for the logistic function used
            as the landmark cutoff function. (unitless)
        :param double cutoff_steepness: Steepness of the logistic cutoff function.
        :param double minimum_site_occupancy = 0.1: Minimum occupancy (% of time occupied)
            for a site to qualify as such.
        :param dict clustering_params: Parameters for the chosen clustering_algorithm
        :param str peak_evening: Whether and what kind of peak "evening" to apply;
            that is, processing that makes all large peaks in the landmark vector
            more similar in magnitude. This can help in site clustering.

            Valid options: 'none', 'clip'
        :param bool weighted_site_positions: When computing site positions, whether
            to weight the average by assignment confidence.
        :param bool check_for_zero_landmarks: Whether to check for and raise exceptions
            when all-zero landmark vectors are computed.
        :param float static_movement_threshold: (Angstrom) the maximum allowed
            distance between an instantanous static atom position and it's ideal position.
        :param bool dynamic_lattice_mapping: Whether to dynamically decide each
            frame which static atom represents each average lattice position;
            this allows the LandmarkAnalysis to deal with, say, a rare exchage of
            two static atoms that does not change the structure of the lattice.

            It does NOT allow LandmarkAnalysis to deal with lattices whose structures
            actually change over the course of the trajectory.

            In certain cases this is better delt with by MergeSitesByDynamics.
        :param int max_mobile_per_site: The maximum number of mobile atoms that can
            be assigned to a single site without throwing an error. Regardless of the
            value, assignments of more than one mobile atom to a single site will
            be recorded and reported.

            Setting this to 2 can be necessary for very diffusive, liquid-like
            materials at high temperatures.

            Statistics related to this are reported in self.avg_mobile_per_site
            and self.n_multiple_assignments.
        :param bool force_no_memmap: if True, landmark vectors will be stored only in memory.
            Only useful if access to landmark vectors after the analysis has run is desired.
        :param bool verbose: If `True`, progress bars will be printed to stdout.
            Other output is handled seperately through the `logging` module.
        """

        self._cutoff_midpoint = cutoff_midpoint
        self._cutoff_steepness = cutoff_steepness
        self._minimum_site_occupancy = minimum_site_occupancy

        self._cluster_algo = clustering_algorithm
        self._clustering_params = clustering_params

        if not peak_evening in ['none', 'clip']:
            raise ValueError("Invalid value `%s` for peak_evening" % peak_evening)
        self._peak_evening = peak_evening

        self.verbose = verbose
        self.check_for_zero_landmarks = check_for_zero_landmarks
        self.weighted_site_positions = weighted_site_positions
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
        view = self._landmark_vectors[:]
        view.flags.writeable = False
        return view

    @analysis_result
    def landmark_dimension(self):
        return self._landmark_dimension


    def run(self, sn, frames):
        """Run the landmark analysis.

        The input SiteNetwork is a network of predicted sites; it's sites will
        be used as the "basis" for the landmark vectors.

        Wraps a copy of `frames` into the unit cell; if you know `frames` is already
        wrapped, set `do_wrap = False` to avoid the copy.

        Takes a SiteNetwork and returns a SiteTrajectory.
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
        # Compute landmark vectors

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
                                            frames, check_for_zeros = self.check_for_zero_landmarks,
                                            tqdm = tqdm, logger = logger)

            if not self.check_for_zero_landmarks and self.n_all_zero_lvecs > 0:
                logger.warning("     Had %i all-zero landmark vectors; no error because `check_for_zero_landmarks = False`." % self.n_all_zero_lvecs)
            elif self.check_for_zero_landmarks:
                assert self.n_all_zero_lvecs == 0

            # -- Step 3: Cluster landmark vectors
            logger.info("  - clustering landmark vectors -")
            #  - Preprocess -
            self._do_peak_evening()

            #  - Cluster -
            clustermod = importlib.import_module("..cluster." + self._cluster_algo, package = __name__)
            importlib.reload(clustermod)
            cluster_func = clustermod.do_landmark_clustering

            cluster_counts, lmk_lbls, lmk_confs = \
                cluster_func(self._landmark_vectors,
                             clustering_params = self._clustering_params,
                             min_samples = self._minimum_site_occupancy / float(sn.n_mobile),
                             verbose = self.verbose)

        logging.info("    Failed to assign %i%% of mobile particle positions to sites." % (100.0 * np.sum(lmk_lbls < 0) / float(len(lmk_lbls))))

        # reshape lables and confidences
        lmk_lbls.shape = (n_frames, sn.n_mobile)
        lmk_confs.shape = (n_frames, sn.n_mobile)

        n_sites = len(cluster_counts)

        if n_sites < (sn.n_mobile / self.max_mobile_per_site):
            raise MultipleOccupancyError("There are %i mobile particles, but only identified %i sites. With %i max_mobile_per_site, this is an error. Check clustering_params." % (sn.n_mobile, n_sites, self.max_mobile_per_site))

        logging.info("    Identified %i sites with assignment counts %s" % (n_sites, cluster_counts))

        # -- Do output
        # - Compute site centers
        site_centers = np.empty(shape = (n_sites, 3), dtype = frames.dtype)

        for site in range(n_sites):
            mask = lmk_lbls == site
            pts = frames[:, sn.mobile_mask][mask]
            if self.weighted_site_positions:
                site_centers[site] = self._pbcc.average(pts, weights = lmk_confs[mask])
            else:
                site_centers[site] = self._pbcc.average(pts)

        # Build output obejcts
        out_sn = sn.copy()

        out_sn.centers = site_centers
        assert out_sn.vertices is None

        out_st = SiteTrajectory(out_sn, lmk_lbls, lmk_confs)

        # Check that multiple particles are never assigned to one site at the
        # same time, cause that would be wrong.
        self.n_multiple_assignments, self.avg_mobile_per_site = out_st.check_multiple_occupancy(
            max_mobile_per_site = self.max_mobile_per_site
        )

        out_st.set_real_traj(orig_frames)
        self._has_run = True

        return out_st

    # -------- "private" methods --------

    def _do_peak_evening(self):
        if self._peak_evening == 'none':
            return
        elif self._peak_evening == 'clip':
            lvec_peaks = np.max(self._landmark_vectors, axis = 1)
            # Clip all peaks to the lowest "normal" (stdev.) peak
            lvec_clip = np.mean(lvec_peaks) - np.std(lvec_peaks)
            # Do the clipping
            self._landmark_vectors[self._landmark_vectors > lvec_clip] = lvec_clip
