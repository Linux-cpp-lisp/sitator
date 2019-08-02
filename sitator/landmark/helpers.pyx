cimport cython
from libc.math cimport sqrt, cos, M_PI, isnan, pow, exp, log

import numpy as np

from sitator.landmark import StaticLatticeError, ZeroLandmarkError

# -- Cython helpers --

ctypedef double precision

def _fill_landmark_vectors(self,
                           sn,
                           verts_np,
                           site_vert_dists,
                           frames,
                           dynmap_compat,
                           check_for_zeros = True,
                           tqdm = lambda i: i,
                           logger = None):
    if self._landmark_dimension is None:
        raise ValueError("_fill_landmark_vectors called before Voronoi!")

    # Setup state variables

    n_frames = len(frames)

    cdef pbcc = self._pbcc

    frame_shift = np.empty(shape = (sn.n_static, 3))
    temp_distbuff = np.empty(shape = sn.n_static, dtype = frames.dtype)

    mobile_idexes = np.where(sn.mobile_mask)[0]

    lattice_map = np.empty(shape = sn.n_static, dtype = np.int)

    lattice_pt = np.empty(shape = 3, dtype = sn.static_structure.positions.dtype)
    max_n_dynmat_compat = max(len(dm) for dm in dynmap_compat)
    lattice_pt_dists = np.empty(shape = max_n_dynmat_compat, dtype = np.float)
    static_pos_buffer = np.empty(shape = (max_n_dynmat_compat, 3), dtype = lattice_pt.dtype)
    static_positions_seen = np.empty(shape = sn.n_static, dtype = np.bool)
    static_positions = np.empty(shape = (sn.n_static, 3), dtype = frames.dtype)
    static_mask_idexes = sn.static_mask.nonzero()[0]

    # - Precompute cutoff function rounding point
    # TODO: Think about the 0.0001 value
    # Even at 0.0000001 and steepness 20 this still gives only ~1.9 (center 1.4),
    # so a stricter threshold is probably fine
    cutoff_round_to_zero = cutoff_round_to_zero_point(self._cutoff_midpoint,
                                                      self._cutoff_steepness,
                                                      0.0001)

    cdef Py_ssize_t n_all_zero_lvecs = 0

    cdef Py_ssize_t landmark_dim = self._landmark_dimension
    cdef Py_ssize_t current_landmark_i = 0

    cdef Py_ssize_t nearest_static_position
    cdef precision nearest_static_distance
    cdef Py_ssize_t n_dynmap_allowed
    # Iterate through time
    for i, frame in enumerate(tqdm(frames, desc = "Landmark Frame")):

        #static_positions = frame[sn.static_mask]
        np.take(frame,
                static_mask_idexes,
                out = static_positions,
                axis = 0,
                mode = 'clip')

        # Every frame, update the lattice map
        static_positions_seen.fill(False)

        for lattice_index in xrange(sn.n_static):
            dynmap_allowed = dynmap_compat[lattice_index]
            n_dynmap_allowed = len(dynmap_allowed)
            lattice_pt[:] = sn.static_structure.positions[lattice_index]
            np.take(static_positions,
                    dynmap_allowed,
                    out = static_pos_buffer[:n_dynmap_allowed],
                    axis = 0,
                    mode = 'clip')

            pbcc.distances(
                lattice_pt,
                static_pos_buffer[:n_dynmap_allowed],
                out = lattice_pt_dists[:n_dynmap_allowed]
            )
            nearest_static_position = np.argmin(lattice_pt_dists[:n_dynmap_allowed])
            nearest_static_distance = lattice_pt_dists[nearest_static_position]
            nearest_static_position = dynmap_allowed[nearest_static_position]

            if static_positions_seen[nearest_static_position]:
                # We've already seen this one... error
                logger.warning("Static atom %i is the closest to more than one static lattice position" % nearest_static_position)
                #raise ValueError("Static atom %i is the closest to more than one static lattice position" % nearest_static_position)

            static_positions_seen[nearest_static_position] = True

            if nearest_static_distance > self.static_movement_threshold:
                raise StaticLatticeError("No static atom position within %f A threshold of static lattice position %i" % (self.static_movement_threshold, lattice_index),
                                         lattice_atoms = [lattice_index],
                                         frame = i,
                                         try_recentering = True)

            lattice_map[lattice_index] = nearest_static_position

        # In normal circumstances, every current static position should be assigned.
        # Just a sanity check
        if (not self.relaxed_lattice_checks) and (not np.all(static_positions_seen)):
            not_assigned_atoms = np.where(~static_positions_seen)[0]
            raise StaticLatticeError("At frame %i, static positions of atoms %s not assigned to lattice positions" % (i, not_assigned_atoms),
                                     lattice_atoms = not_assigned_atoms,
                                     frame = i,
                                     try_recentering = True)


        for j in xrange(sn.n_mobile):
            mobile_pt = frame[mobile_idexes[j]]

            # Shift the Li in question to the center of the unit cell
            frame_shift[:] = static_positions
            frame_shift -= mobile_pt
            frame_shift += pbcc.cell_centroid

            # Wrap all positions into the unit cell
            pbcc.wrap_points(frame_shift)

            # The mobile ion is now at the center of the cell --
            # compute the landmark vector
            fill_landmark_vec(self._landmark_vectors, i, sn.n_mobile, j,
                              landmark_dim, frame_shift, lattice_map,
                              verts_np, site_vert_dists,
                              pbcc.cell_centroid,
                              self._cutoff_midpoint,
                              self._cutoff_steepness,
                              cutoff_round_to_zero,
                              temp_distbuff)

            if np.count_nonzero(self._landmark_vectors[current_landmark_i]) == 0:
                if check_for_zeros:
                    raise ZeroLandmarkError(mobile_index = j, frame = i)
                else:
                    n_all_zero_lvecs += 1

            current_landmark_i += 1

    self.n_all_zero_lvecs = n_all_zero_lvecs


cdef precision cutoff_round_to_zero_point(precision cutoff_midpoint,
                                          precision cutoff_steepness,
                                          precision threshold):
    # Computed by solving for x:
    return cutoff_midpoint + log((1/threshold) - 1.) / cutoff_steepness


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void fill_landmark_vec(precision [:,:] landmark_vectors,
                  Py_ssize_t i,
                  Py_ssize_t n_li,
                  Py_ssize_t j,
                  Py_ssize_t landmark_dim,
                  const precision [:,:] lattice_positions,
                  const Py_ssize_t [:] lattice_map,
                  const Py_ssize_t [:,:] verts_np,
                  const precision [:, :] verts_centroid_dists,
                  const precision [:] li_pos,
                  precision cutoff_midpoint,
                  precision cutoff_steepness,
                  precision cutoff_round_to_zero,
                  precision [:] distbuff) nogil:

    # Fill the landmark vector
    cdef int [:] vert
    cdef Py_ssize_t v
    cdef precision ci
    cdef precision temp
    cdef const precision [:] pt
    cdef int n_verts

    # precompute all distances
    # lattice_map maps an original static lattice index to an index in the current
    # frames lattice positions
    for idex in xrange(len(lattice_positions)):
        pt = lattice_positions[lattice_map[idex]]
        temp = sqrt((pt[0] - li_pos[0])**2 + (pt[1] - li_pos[1])**2 + (pt[2] - li_pos[2])**2)

        distbuff[idex] = temp

    # For each component
    for k in xrange(landmark_dim):
        ci = 1.0

        n_verts = 0
        for h in xrange(verts_np.shape[1]):
            v = verts_np[k, h]
            if v == -1:
                break
            n_verts += 1

            # normalize to centroid distance
            temp = distbuff[v] / verts_centroid_dists[k, h]

            if temp > cutoff_round_to_zero:
                temp = 0.0
                # Short circut
                ci = 0.0
                break
            else:
                temp = 1.0 / (1.0 + exp(cutoff_steepness * (temp - cutoff_midpoint)))

            # Multiply into accumulator
            ci *= temp

        # "Normalize" to number of vertices
        landmark_vectors[(i * n_li) + j, k] = pow(ci, 1.0 / n_verts)
