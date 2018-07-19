import numpy as np
from libc.math cimport sqrt, cos, M_PI, isnan

from cython import boundscheck, wraparound
from cython.parallel import prange



ctypedef fused precision:
    float
    double
    long double

# cpdef pbc_avg(frames,
#               cell,
#               cell_inverse):


@boundscheck(False)
@wraparound(False)
cpdef void fill_landmark_vec(precision [:,:] landmark_vectors,
                      Py_ssize_t i,
                      Py_ssize_t n_li,
                      Py_ssize_t j,
                      Py_ssize_t landmark_dim,
                      const precision [:,:] lattice_positions,
                      const Py_ssize_t [:,:] verts_np,
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

    # precompute all cutoff-ed distances
    for idex in xrange(len(lattice_positions)):
        pt = lattice_positions[idex]
        temp = sqrt((pt[0] - li_pos[0])**2 + (pt[1] - li_pos[1])**2 + (pt[2] - li_pos[2])**2)

        if temp > cutoff:
            distbuff[idex] = 0.0
        else:
            distbuff[idex] = (cos((M_PI / cutoff) * temp) + 1.0) * 0.5

    # For each component
    for k in xrange(landmark_dim):
        ci = 1.0

        for h in xrange(verts_np.shape[1]):
            v = verts_np[k, h]
            if v == -1:
                break

            # Multiply into accumulator
            ci *= distbuff[v]

        landmark_vectors[(i * n_li) + j, k] = ci
