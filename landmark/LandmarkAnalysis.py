import numpy as np

from analysis.util import PBCCalculator

# From https://github.com/tqdm/tqdm/issues/506#issuecomment-373126698
import sys
try:
    ipy_str = str(type(get_ipython()))
    if 'zmqshell' in ipy_str:
        from tqdm import tqdm_notebook as tqdm
    if 'terminal' in ipy_str:
        from tqdm import tqdm
except:
    if sys.stderr.isatty():
        from tqdm import tqdm
    else:
        def tqdm(iterable, **kwargs):
            return iterable

import importlib

import helpers
from analysis import SiteNetwork, SiteTrajectory


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
                 cutoff = 2.0,
                 minimum_site_occupancy = 0.1,
                 peak_evening = 'none',
                 weighted_site_positions = True,
                 check_for_zero_landmarks = True,
                 verbose = True):
        """
        :param double cutoff: The distance cutoff for the landmark vectors. (unitless)
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
        :param bool verbose: If `True`, progress bars and messages will be printed to stdout.
        """

        self._cutoff = cutoff
        self._minimum_site_occupancy = minimum_site_occupancy

        self._cluster_algo = clustering_algorithm
        self._clustering_params = clustering_params

        if not peak_evening in ['none', 'clip']:
          raise ValueError("Invalid value `%s` for peak_evening" % peak_evening)
        self._peak_evening = peak_evening

        self.verbose = verbose
        self.check_for_zero_landmarks = check_for_zero_landmarks
        self.weighted_site_positions = weighted_site_positions

        self._landmark_vectors = None
        self._landmark_dimension = None

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

        Takes a SiteNetwork and returns a SiteTrajectory.
        """

        if self._has_run:
            raise ValueError("Cannot rerun LandmarkAnalysis!")

        if frames.shape[1:] != (sn.n_total, 3):
            raise ValueError("Wrong shape %s for frames." % frames.shape)

        if sn.vertices is None:
            raise ValueError("Input SiteNetwork must have vertices")

        n_frames = len(frames)

        if self.verbose:
            print "--- Running Landmark Analysis ---"

        # Create PBCCalculator
        self._pbcc = PBCCalculator(sn.structure.cell)

        # -- Step 1: Compute site-to-vertex distances
        self._landmark_dimension = sn.n_sites

        longest_vert_set = np.max([len(v) for v in sn.vertices])
        verts_np = np.array([v + [-1] * (longest_vert_set - len(v)) for v in sn.vertices])
        site_vert_dists = np.empty(shape = verts_np.shape, dtype = np.float)
        site_vert_dists.fill(np.nan)

        for i, polyhedron in enumerate(sn.vertices):
            verts_poses = sn.static_structure.get_positions()[polyhedron]
            dists = self._pbcc.distances(sn.centers[i], verts_poses)
            site_vert_dists[i, :len(polyhedron)] = dists

        # -- Step 2: Compute landmark vectors
        if self.verbose: print "  - computing landmark vectors -"
        # Compute landmark vectors
        helpers._fill_landmark_vectors(self, sn, verts_np, site_vert_dists,
                                        frames, check_for_zeros = self.check_for_zero_landmarks,
                                        tqdm = tqdm)

        # -- Step 3: Cluster landmark vectors
        if self.verbose: print "  - clustering landmark vectors -"
        #  - Preprocess -
        self._do_peak_evening()

        #  - Cluster -
        cluster_func = importlib.import_module("..cluster." + self._cluster_algo, package = __name__).do_landmark_clustering

        cluster_counts, lmk_lbls, lmk_confs = \
            cluster_func(self._landmark_vectors,
                         clustering_params = self._clustering_params,
                         min_samples = self._minimum_site_occupancy / float(sn.n_mobile),
                         verbose = self.verbose)

        # reshape lables and confidences
        lmk_lbls.shape = (n_frames, sn.n_mobile)
        lmk_confs.shape = (n_frames, sn.n_mobile)

        if self.verbose:
            print "    Identified %i sites with assignment counts %s" % (len(cluster_counts), cluster_counts)

        n_sites = len(cluster_counts)

        # -- Do output
        # - Compute site centers
        site_centers = np.empty(shape = (n_sites, 3), dtype = frames.dtype)

        for site in xrange(n_sites):
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
        out_st.set_real_traj(frames)
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
