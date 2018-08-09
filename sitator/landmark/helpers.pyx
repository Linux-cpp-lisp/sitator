cimport cython
from libc.math cimport sqrt, cos, M_PI, isnan, pow

import numpy as np

# -- Cython helpers --

ctypedef double precision

def _fill_landmark_vectors(self, sn, verts_np, site_vert_dists, frames, check_for_zeros = True, tqdm = lambda i: i):
    if self._landmark_dimension is None:
        raise ValueError("_fill_landmark_vectors called before Voronoi!")

    n_frames = len(frames)

    cdef pbcc = self._pbcc

    frame_shift = np.empty(shape = (sn.n_static, 3))
    temp_distbuff = np.empty(shape = sn.n_static, dtype = frames.dtype)

    mobile_idexes = np.where(sn.mobile_mask)[0]

    if self.dynamic_lattice_mapping:
        lattice_map = np.empty(shape = sn.n_static, dtype = np.int)
    else:
        # Otherwise just map to themselves
        lattice_map = np.arange(sn.n_static, dtype = np.int)

    lattice_pt = np.empty(shape = 3, dtype = sn.static_structure.positions.dtype)
    lattice_pt_dists = np.empty(shape = sn.n_static, dtype = np.float)
    static_positions_seen = np.empty(shape = sn.n_static, dtype = np.bool)

    cdef Py_ssize_t landmark_dim = self._landmark_dimension
    cdef Py_ssize_t current_landmark_i = 0
    # Iterate through time
    for i, frame in enumerate(tqdm(frames, desc = "Frame")):

        static_positions = frame[sn.static_mask]

        # Every frame, update the lattice map
        static_positions_seen.fill(False)

        for lattice_index in xrange(sn.n_static):
            lattice_pt = sn.static_structure.positions[lattice_index]
            pbcc.distances(lattice_pt, static_positions, out = lattice_pt_dists)
            if self.dynamic_lattice_mapping:
                nearest_static_position = np.argmin(lattice_pt_dists)
            else:
                nearest_static_position = lattice_index

            if static_positions_seen[nearest_static_position]:
                # We've already seen this one... error
                print "Static atom %i is the closest to more than one static lattice position" % nearest_static_position
                #raise ValueError("Static atom %i is the closest to more than one static lattice position" % nearest_static_position)

            static_positions_seen[nearest_static_position] = True

            if lattice_pt_dists[nearest_static_position] > self.static_movement_threshold:
                raise ValueError("No static atom position within %f A threshold of static lattice position %i" % (self.static_movement_threshold, lattice_index))

            if self.dynamic_lattice_mapping:
                lattice_map[lattice_index] = nearest_static_position

        # In normal circumstances, every current static position should be assigned.
        # Just a sanity check
        if (not self.relaxed_lattice_checks) and (not np.all(static_positions_seen)):
            raise ValueError("At frame %i, static positions of atoms %s not assigned to lattice positions" % (i, np.where(~static_positions_seen)[0]))


        for j in xrange(sn.n_mobile):
            mobile_pt = frame[mobile_idexes[j]]

            # Shift the Li in question to the center of the unit cell
            np.copyto(frame_shift, frame[sn.static_mask])
            frame_shift += (pbcc.cell_centroid - mobile_pt)

            # Wrap all positions into the unit cell
            pbcc.wrap_points(frame_shift)

            # The mobile ion is now at the center of the cell --
            # compute the landmark vector
            fill_landmark_vec(self._landmark_vectors, i, sn.n_mobile, j,
                              landmark_dim, frame_shift, lattice_map,
                              verts_np, site_vert_dists,
                              pbcc.cell_centroid,
                              self._cutoff, temp_distbuff)

            if check_for_zeros and (np.count_nonzero(self._landmark_vectors[current_landmark_i]) == 0):
                raise ValueError("Encountered a zero landmark vector for mobile ion %i at frame %i." % (j, i))

            current_landmark_i += 1

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
                  precision cutoff,
                  precision [:] distbuff) nogil:

    # Pure Python equiv:
    #         for k in xrange(landmark_dim):
    #             lvec = np.linalg.norm(lattice_positions[verts[k]] - cell_centroid, axis = 1)
    #             past_cutoff = lvec > cutoff

    #             # Short circut it, since the product then goes to zero too.
    #             if np.any(past_cutoff):
    #                 landmark_vectors[(i * n_li) + j, k] = 0
    #             else:
    #                 lvec = (np.cos((np.pi / cutoff) * lvec) + 1.0) / 2.0
    #                 landmark_vectors[(i * n_li) + j, k] = np.product(lvec)

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

        # if temp > cutoff:
        #     distbuff[idex] = 0.0
        # else:
        #     distbuff[idex] = (cos((M_PI / cutoff) * temp) + 1.0) * 0.5

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

            if temp > cutoff:
                temp = 0.0
                # Short circut
                ci = 0.0
                break
            elif temp < 1.0:
                temp = 1.0
            else:
                temp = (cos((M_PI / (cutoff - 1.0)) * (temp - 1.0)) + 1.0) * 0.5

            # Multiply into accumulator
            #ci *= distbuff[v]
            ci *= temp

        # "Normalize" to number of vertices
        landmark_vectors[(i * n_li) + j, k] = pow(ci, 1.0 / n_verts)
