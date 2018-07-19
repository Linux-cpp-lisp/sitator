
import numpy as np

import pyximport; pyximport.install()
from fast_util import fill_landmark_vec, wrap_coords

from vorozeo import Zeopy

DEFAULT_ZEOPP = "/home/amusaelian/Documents/Ionic Frustration/code/lib/zeo++/trunk/network"

class LandmarkAnalysis(object):
    """ Track a mobile species through a fixed lattice using landmark vectors."""

    def __init__(structure,
                static_mask,
                mobile_mask,
                verbose = True,
                zeopp_path = DEFAULT_ZEOPP):
        """
        :param Atoms structure: an ASE/Quippy :class:Atoms object containing the structure simulated in the trajectory
        :param ndarray(bool) static_mask: Boolean mask indicating which atoms to consider immobile
        :param ndarray(bool) mobile_mask: Boolean mask indicating which atoms to track
        :param bool verbose: If `True`, progress bars and messages will be printed to stdout.
        """

        assert static_mask.ndim == mobile_mask.ndim == 1, "The masks must be one-dimensional"
        assert len(structure) == len(static_mask) == len(mobile_mask), "The masks must have the same length as the # of atoms in the strucutre."

        self._structure = structure
        self._static_mask = static_mask
        self.n_static = np.sum(static_mask)
        self._mobile_mask = mobile_mask
        self.n_mobile = np.sum(mobile_mask)

        # Create static structure
        self._static_structure = structure.copy()
        del self._static_structure[(~static_mask) & mobile_mask]
        assert len(self._static_structure) == self.n_static

        self.verbose = verbose

        self._landmark_vectors = None

        self._zeopy = Zeopy(zeopp_path)
        self._voronoi_vertices = None
        self._landmark_dimension = None

    @property
    def landmark_vectors(self):
        # lazy eval
        if self._landmark_vectors is None:
            raise ValueError("This LandmarkAnalysis hasn't been run yet.")

        view = self._landmark_vectors[:]
        view.flags.writeable = False
        return view

    @property
    def landmark_dimension(self):
        if self._landmark_vectors is None:
            raise ValueError("This LandmarkAnalysis hasn't been run yet.")

        return self._landmark_dimension

    def run_analysis(self, frames, n_jobs = 1):

        if frames.shape[1:] != len(self.structure)

        # Compute landmark vectors
        self._fill_landmark_vectors()

        lvecs = self.landmark_vectors

    def _do_voronoi(self):
        # Run Zeo++
        assert not self._static_structure is None
        zeoverts, edges = self._zeopy.voronoi(self._static_structure, radial = False, verbose = self.verbose)

        # For the outside world, the pretty list of lists
        self.voronoi_vertices = [v['region-atom-indexes'] for v in zeoverts]

        # For the inside world, a numpy array for the sake of cython performance
        longest_vert_set = np.max([len(v) for v in verts])
        self._voronoi_vertices = np.array([v + [-1] * (verts_max - len(v)) for v in verts])

    def _fill_landmark_vectors(self, frames):
        if self._voronoi_vertices is None or self._landmark_dimension is None:
            raise ValueError("_fill_landmark_vectors called before Voronoi!")

        n_frames = len(frames)

        # The dimension of one landmark vector is the number of Voronoi regions
        self._landmark_vectors = np.empty(shape = (n_frames * self.n_mobile, self._landmark_dimension))


        cell = np.matrix(self.structure.cell).T
        cell_inverse = np.asarray(cell.I)
        cell_arr = np.asarray(cell)
        cell_centroid = np.sum(t.atoms.cell * 0.5, axis = 0)

        temp_distbuff = np.empty(shape = (n_lattice), dtype = np.float)
        # Iterate through time
        for i, frame in enumerate(tqdm(t.get_positions()[step_slice[0]:step_slice[1]:landmark_stride])):

            for j in xrange(n_li):
                li_pos = frame[li_indexes[j]]

                # Shift the Li in question to the center of the unit cell
                frame_shift = frame + (cell_centroid - li_pos)

                # Wrap all positions into the unit cell
        #         for pt in xrange(len(frame_shift)):
        #             # Get wrapped lattice coords
        #             frame_shift[pt] = np.dot(cell_inverse, frame_shift[pt]) % 1.0
        #             # Restore to real coords
        #             frame_shift[pt] = np.dot(cell, frame_shift[pt])
                wrap_coords(frame_shift, cell_arr, cell_inverse)

                lattice_positions = frame_shift[~li_mask]

                # The Li is now at the center of the cell...

                # Fill the landmark vector -- pure Python
        #         for k in xrange(landmark_dim):
        #             lvec = np.linalg.norm(lattice_positions[verts[k]] - cell_centroid, axis = 1)
        #             past_cutoff = lvec > cutoff

        #             # Short circut it, since the product then goes to zero too.
        #             if np.any(past_cutoff):
        #                 landmark_vectors[(i * n_li) + j, k] = 0
        #             else:
        #                 lvec = (np.cos((np.pi / cutoff) * lvec) + 1.0) / 2.0
        #                 landmark_vectors[(i * n_li) + j, k] = np.product(lvec)

                # Do the same, but Cython. At least 7 times faster.
                fill_landmark_vec(landmark_vectors, i, n_li, j,
                                  landmark_dim, lattice_positions, verts_np,
                                  cell_centroid, cutoff, temp_distbuff)
