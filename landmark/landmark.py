import numpy as np

from util import PBCCalculator

from util import Zeopy

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

import helpers

import importlib


from functools import wraps
def analysis_result(func):
    @property
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not self._has_run:
            raise ValueError("This LandmarkAnalysis hasn't been run yet.")
        return func(self, *args, **kwargs)
    return wrapper

DEFAULT_ZEOPP = "/home/amusaelian/Documents/Ionic Frustration/code/lib/zeo++/trunk/network"

class LandmarkAnalysis(object):
    """Track a mobile species through a fixed lattice using landmark vectors."""

    def __init__(self,
                structure,
                static_mask,
                mobile_mask,
                clustering_algorithm = 'dotprod',
                clustering_params = {},
                cutoff = 3.0,
                minimum_site_occupancy = 0.1,
                peak_evening = 'none',
                weighted_site_positions = True,
                verbose = True,
                zeopp_path = DEFAULT_ZEOPP):
        """
        :param Atoms structure: an ASE/Quippy :class:Atoms object containing the structure simulated in the trajectory
          Should be a representative/ideal/thermal-average structure.
        :param ndarray(bool) static_mask: Boolean mask indicating which atoms to consider immobile
        :param ndarray(bool) mobile_mask: Boolean mask indicating which atoms to track
        :param double cutoff: The distance cutoff for the landmark vectors.
        :param double minimum_site_occupancy = 0.1: Minimum occupancy (% of time occupied)
            for a site to qualify as such.
        :param double site_clustering_threshold = 0.45: Similarity threshold for creating a site;
            passed through to DotProdClassifier.
        :param double site_assignment_threshold = 0.8: Similarity threshold for assigning a vectors
            to a site; passed through to DotProdClassifier.
        :param str peak_evening: Whether and what kind of peak "evening" to apply;
            that is, processing that makes all large peaks in the landmark vector
            more similar in magnitude. This can help in site clustering.

            Valid options: 'none', 'clip'
        :param bool verbose: If `True`, progress bars and messages will be printed to stdout.
        """

        assert static_mask.ndim == mobile_mask.ndim == 1, "The masks must be one-dimensional"
        assert len(structure) == len(static_mask) == len(mobile_mask), "The masks must have the same length as the # of atoms in the strucutre."

        self._structure = structure
        self._static_mask = static_mask
        self.n_static = np.sum(static_mask)
        self._mobile_mask = mobile_mask
        self.n_mobile = np.sum(mobile_mask)

        self._cutoff = cutoff
        self._minimum_site_occupancy = minimum_site_occupancy

        self._cluster_algo = clustering_algorithm
        self._clustering_params = clustering_params

        if not peak_evening in ['none', 'clip']:
          raise ValueError("Invalid value `%s` for peak_evening" % peak_evening)
        self._peak_evening = peak_evening

        # Create static structure
        self._static_structure = structure.copy()
        del self._static_structure[(~static_mask) & mobile_mask]
        assert len(self._static_structure) == self.n_static

        # Create PBCCalculator
        self._pbcc = PBCCalculator(structure.cell)

        self.verbose = verbose
        self.weighted_site_positions = weighted_site_positions

        self._landmark_vectors = None

        self._zeopy = Zeopy(zeopp_path)
        self._voronoi_vertices = None
        self._landmark_dimension = None

        # Lazy-created result caches
        self._site_averages = None
        self._site_occupancies = None

        self._has_run = False
        self._stats = {}


    @analysis_result
    def landmark_vectors(self):
        view = self._landmark_vectors[:]
        view.flags.writeable = False
        return view


    @property
    def cutoff(self):
      return self._cutoff


    @analysis_result
    def landmark_dimension(self):
        return self._landmark_dimension


    @analysis_result
    def site_occupancies(self):
        return self._site_occupancies

    @analysis_result
    def n_assigned_positions_per_site(self):
        return np.round(self._site_occupancies * self._n_frames)

    @analysis_result
    def site_positions(self):
        """Returns the "position" of each site.
        Actually the mean of all positions assigned to it.
        """

        if self._site_averages is None:
            self._site_averages = np.empty(shape = (self.n_sites, 3), dtype = self._frames.dtype)

            if self.weighted_site_positions:
                for site in xrange(self.n_sites):
                    pos, confs = self.all_positions_for_site(site, return_confidences = True)
                    self._site_averages[site] = self._pbcc.average(pos, weights = confs)
            else:
                for site in xrange(self.n_sites):
                    self._site_averages[site] = self._pbcc.average(self.all_positions_for_site(site))

        return self._site_averages


    @analysis_result
    def n_sites(self):
        return len(self._site_occupancies)

    @analysis_result
    def n_frames(self):
        return self._n_frames

    @analysis_result
    def analysis_statistics(self):
        """Get a dictionary of various information on how well the analysis went."""
        return self._stats

    def assign_to_last_known_site(self, frame_threshold = 1):
        """Assign unassigned mobile particles to their last known site within
            `frame_threshold` frames.

        :returns: information dictionary of debugging/diagnostic information.
        """
        if not self._has_run:
            raise ValueError("This LandmarkAnalysis hasn't been run yet.")

        total_unknown = np.sum(self._lmk_lbls < 0)

        if self.verbose:
            print "%i unassigned positions (%i%%); assigning unassigned mobile particles to last known positions within %i frames..." % (total_unknown, 100.0 * (total_unknown / float(self._n_positions)), frame_threshold)

        last_known = np.empty(shape = self.n_mobile, dtype = np.int)
        last_known.fill(-1)
        time_unknown = np.zeros(shape = self.n_mobile, dtype = np.int)
        avg_time_unknown = 0
        avg_time_unknown_div = 0
        max_time_unknown = 0
        total_reassigned = 0

        for i in xrange(self._n_frames):
            # All those unknown this frame
            unknown = self._lmk_lbls[i] == -1
            # Update last_known for assigned sites
            last_known[~unknown] = self._lmk_lbls[i][~unknown]

            times = time_unknown[~unknown]
            times = times[times != 0]

            if len(times) > 0:
                maxtime = np.max(times)
                if maxtime > frame_threshold:
                    max_time_unknown = maxtime
                avg_time_unknown += np.sum(times)
                avg_time_unknown_div += len(times)

            time_unknown[~unknown] = 0

            to_correct = unknown & (time_unknown < frame_threshold)

            self._lmk_lbls[i][to_correct] = last_known[to_correct]
            total_reassigned += np.sum(to_correct)
            time_unknown[unknown] += 1

        res = None
        if avg_time_unknown_div > 0: # We corrected some unknowns
            avg_time_unknown = float(avg_time_unknown) / avg_time_unknown_div
            post_percent_unassign = 100.0 * ((total_unknown - total_reassigned) / float(self._n_positions))

            if self.verbose:
                print "  Maximum # of frames any mobile particle spent unassigned: %i" % max_time_unknown
                print "  Avg. # of frames spent unassigned: %f" % avg_time_unknown
                print "  Assigned %i/%i unassigned positions, leaving %i/%i (%i%%) unknown" % (total_reassigned, total_unknown, total_unknown - total_reassigned, self._n_positions, post_percent_unassign)

            res = {
                'max_time_unknown' : max_time_unknown,
                'avg_time_unknown' : avg_time_unknown,
                'total_unassigned' : total_unknown,
                'total_reassigned' : total_reassigned,
                'post_correction_total_unassigned' : post_percent_unassign
            }
        else:
            if self.verbose:
                print "  None to correct."

            res = {
                'max_time_unknown' : 0,
                'avg_time_unknown' : 0,
                'total_unassigned' : 0,
                'total_reassigned' : 0,
                'post_correction_total_unassigned' : total_unknown
            }

        self._stats.update(res)
        return res

    def site_trajectory_for_particle(self, i, return_confidences = False):
        """Returns the site trajectory of mobile particle(s) i.

        :param bool return_confidences: If True, also return an array of the
          0.0-1.0 "confidences" in that frame's site assignment.

        :returns: n_frames x len(i) ndarray of site ID's
        """
        if not self._has_run:
            raise ValueError("This LandmarkAnalysis hasn't been run yet.")
        if return_confidences:
            return self._lmk_lbls[:, i], self._lmk_confs[:, i]
        else:
            return self._lmk_lbls[:, i]


    def all_positions_for_site(self, site, return_confidences = False):
        """Return all positions assigned to `site`
        """
        assert 0 <= site < self.n_sites
        msk = self._lmk_lbls == site
        poses = self._frames[:, self._mobile_mask][msk]
        if return_confidences:
            return poses, self._lmk_confs[msk].flatten()
        else:
            return poses


    def run_analysis(self, frames,
                    n_jobs = 1,
                    check_for_zero_landmarks = True):
        """Run the landmark analysis. Returns nothing.

        :warning: The LandmarkAnalysis will keep a reference to `frames` in order
            to answer later queries about the analysis. To free up memory, be sure
            to delete the LandmarkAnalysis too!
        """

        if self._has_run:
            raise ValueError("Cannot rerun LandmarkAnalysis!")

        if frames.shape[1:] != (len(self._structure), 3):
          raise ValueError("Wrong shape %s for frames." % frames.shape)

        n_frames = len(frames)

        if self.verbose:
            print "--- Running Landmark Analysis ---"
            print "  - doing voronoi decomposition -"

        # -- Step 1: Voronoi analysis
        self._do_voronoi()
        self._landmark_dimension = len(self._voronoi_vertices)

        # -- Step 2: Compute landmark vectors
        if self.verbose: print "  - computing landmark vectors -"
        # Compute landmark vectors
        helpers._fill_landmark_vectors(self, frames, check_for_zeros = check_for_zero_landmarks, tqdm = tqdm)

        # -- Step 3: Cluster landmark vectors
        if self.verbose: print "  - clustering landmark vectors -"
        #  - Preprocess -
        self._do_peak_evening()

        #  - Cluster -

        cluster_func = importlib.import_module("..cluster." + self._cluster_algo, package = __name__).do_landmark_clustering

        cluster_counts, self._lmk_lbls, self._lmk_confs = \
            cluster_func(self._landmark_vectors,
                         clustering_params = self._clustering_params,
                         min_samples = self._minimum_site_occupancy / float(self.n_mobile),
                         verbose = self.verbose)

        # reshape lables and confidences
        self._lmk_lbls.shape = (n_frames, self.n_mobile)
        self._lmk_confs.shape = (n_frames, self.n_mobile)

        # Save some stats
        self._stats['failed_to_assign_percent'] = 100.0 * (np.sum(self._lmk_lbls == -1) / float(len(self._landmark_vectors)))

        if self.verbose:
            print "    Failed to assign %i%% mobile particle positions to individual sites." % int(self._stats['failed_to_assign_percent'])
            print "    Identified %i sites with assignment counts %s" % (len(cluster_counts), cluster_counts)

        self._site_occupancies = np.true_divide(cluster_counts, n_frames)

        # Save a weakref to frames for computing other analysis properties later
        self._frames = frames
        self._n_frames = n_frames
        self._n_positions = n_frames * self.n_mobile

        self._has_run = True

    # -------- "private" methods --------

    def _do_voronoi(self):
        # Run Zeo++
        assert not self._static_structure is None

        # from samos.analysis.jumps.voronoi import VoronoiNetwork
        #
        # vn = VoronoiNetwork()
        # vn.set_atoms(self._static_structure, self._static_structure.get_chemical_symbols())
        # vn.decompose_qhull()
        #
        # voronoi_nodes = np.asarray([node._center for node in vn.nodes])
        # self.voronoi_vertices = [node._vertices for node in vn.nodes]
        # self._voronoi_vertices = np.asarray(self.voronoi_vertices)

        nodes, verts, edges, _ = self._zeopy.voronoi(self._static_structure, radial = False, verbose = self.verbose)

        # -- Vertices

        # For the outside world, the pretty list of lists
        self.voronoi_vertices = verts

        # For the inside world, a numpy array for the sake of cython performance
        longest_vert_set = np.max([len(v) for v in self.voronoi_vertices])
        self._voronoi_vertices = np.array([v + [-1] * (longest_vert_set - len(v)) for v in self.voronoi_vertices])

        # ------ Compute centroid distances
        vvcd = np.empty(shape = len(self._voronoi_vertices), dtype = np.float)
        vvcd.fill(np.nan)

        for i, polyhedron in enumerate(self.voronoi_vertices):
            verts_poses = self._static_structure.get_positions()[polyhedron]
            dists = self._pbcc.distances(nodes[i], verts_poses)

            if len(dists) == 4:
                stdthresh = 0.0001
            else:
                # Be pretty generous with the std threshold, since Zeo's merging
                # *can* give nodes that aren't strictly equidistant.
                stdthresh = 0.05
            assert np.std(dists) < stdthresh, "Bad node distances %s (stdev. %f)" % (dists, np.std(dists))

            # if they're all the same, mean is same as dists[0]
            # if they're slightly not (Zeo++ merge), accounts for that.
            vvcd[i] = np.mean(dists)

        self._voronoi_vert_centroid_dists = vvcd

    def _do_peak_evening(self):
      if self._peak_evening == 'none':
          return
      elif self._peak_evening == 'clip':
          lvec_peaks = np.max(self._landmark_vectors, axis = 1)
          # Clip all peaks to the lowest "normal" (stdev.) peak
          lvec_clip = np.mean(lvec_peaks) - np.std(lvec_peaks)
          # Do the clipping
          self._landmark_vectors[self._landmark_vectors > lvec_clip] = lvec_clip
