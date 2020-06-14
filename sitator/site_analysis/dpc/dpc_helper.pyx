# cython: boundscheck=False, wraparound=False

cimport cython
from libc.math cimport sqrt, floor, ceil, INFINITY, NAN

import numpy as np
cimport numpy as npc

import sitator
from sitator.util.PBCCalculator cimport PBCCalculator, precision, cell_precision
from .density cimport gridded_density_periodic, to_linear_index_3D, from_linear_index_3D

CLUSTER_UNASSIGNED = sitator.SiteTrajectory.SITE_UNASSIGNED
cdef int _CLUSTER_UNASSIGNED_C = CLUSTER_UNASSIGNED


cpdef dpc_compute_vars(frames_np,
                       PBCCalculator pbcc,
                       int n_boxes_max,
                       int n_gamma,
                       precision d_cutoff = 0.0,
                       which_pts = None):
    """
    Compute delta (distance to higher density), and gamma (their product)

    Returns:
        density, point_box_assignments, deltas, delta_targets, gamma_buf, gamma_idex_buf
    """
    # == Set-up ==
    cdef Py_ssize_t i, j, k
    if frames_np.shape[2] != 3:
        raise ValueError("Data points must be 3D")
    cdef Py_ssize_t n_frames = frames_np.shape[0]
    cdef Py_ssize_t n_points = frames_np.shape[1]

    point_box_assignments = np.empty(shape = (n_frames, n_points), dtype = np.int)

    # == Compute Density ==
    density_np, box_dims = gridded_density_periodic(
        frames_np,
        n_boxes_max,
        pbcc,
        which_pts = which_pts,
        output_box_assignments = point_box_assignments,
        d_cutoff = d_cutoff
    )
    cdef precision [:, :, :] density = density_np
    cdef precision max_density = np.max(density_np)

    if d_cutoff < np.max(box_dims):
        raise ValueError(
            "d_cutoff = %f is too small for this n_boxes_max, which would require d_cutoff >= %f" % (d_cutoff, np.max(box_dims))
        )

    # == Compute Delta and Gamma ==
    cdef Py_ssize_t n_boxes[3]
    n_boxes[:] = density.shape
    cdef Py_ssize_t max_n_boxes = np.max(density.shape)
    # cdef Py_ssize_t box_order[3]
    # box_order[:] = np.argsort(box_dims) # Smallest first
    box_vecs_np =  pbcc.cell / n_boxes
    cdef cell_precision [:, :] box_vecs = box_vecs_np

    deltas_np = np.full(density_np.shape, np.inf, dtype = frames_np.dtype)
    cdef precision [:, :, :] deltas = deltas_np
    # As a linear index, the index of the nearest box with higher density
    delta_targets_np = np.empty(shape = density_np.shape, dtype = np.int)
    delta_targets_np.fill(-2)
    cdef Py_ssize_t [:, :, :] delta_targets = delta_targets_np
    # The linear indexes, ordered, of the boxes with the highest gamma
    gamma_idex_buf_np = np.full(n_gamma, -1, dtype = np.int)
    cdef Py_ssize_t [:] gamma_idex_buf = gamma_idex_buf_np
    cdef Py_ssize_t[3] gamma_idex_temp # A temporary buffer
    # Their gammas
    gamma_buf_np = np.full(n_gamma, -np.inf, dtype = frames_np.dtype)
    cdef precision [:] gamma_buf = gamma_buf_np
    #cdef Py_ssize_t offset[3]
    #cdef Py_ssize_t radius, component, temp1, temp2, side_i, side_j
    #cdef int side_sign
    #cdef Py_ssize_t offset_idex[3]
    #cdef precision offset_vec[3]
    cdef precision gamma #, curr_dist

    cdef precision curr_delta
    cdef Py_ssize_t curr_delta_target

    for i in range(n_boxes[0]):
        for j in range(n_boxes[1]):
            for k in range(n_boxes[2]):
                if density[i, j, k] >= max_density:
                    # In the non periodic case, Rodriguez and Laio
                    # customarily take this to be the maximum pairwise
                    # distance between data points. Since we're periodic,
                    # we take it as the distance to its minimum
                    # image of itself, which similarly gives a sense of the
                    # maximum scale of the data.
                    #
                    # We compute it here since this branch is only ever taken once.
                    deltas[i, j, k] = np.min(pbcc.cell_vector_lengths)
                    delta_targets[i, j, k] = -1
                elif density[i, j, k] == 0:
                    # We don't care about 0's, they're sparse:
                    deltas[i, j, k] = NAN
                    delta_targets[i, j, k] = -1
                    # Skip gamma computations
                    continue
                else:
                    # Compute delta for real
                    # Problem now find closest positive entry
                    # iterate through offsets ordered by distance increasing
                    # Check squares of each radius, i.e., radius 1 is:
                    #   # # #
                    #   # * #
                    #   # # #
                    # etc.
                    curr_delta, curr_delta_target = min_dist_for_relation_pbc(
                        i,j,k,
                        density,
                        n_boxes, box_vecs,
                        RELATION_GT,
                        max_n_boxes//2 + 1, # max radius,
                    )
                    assert curr_delta_target >= 0
                    # ^ There should always a valid delta, since we dealt
                    # with the max case seperately above.
                    deltas[i,j,k] = curr_delta
                    delta_targets[i,j,k] = curr_delta_target
                    # == end for density if-elif-else

                gamma = deltas[i, j, k] * density[i, j, k]
                # Go backwards since its much more likely that
                # our gamma doesn't make the outlier list at all
                temp1 = 0
                # ^ If the loop never trips, that means we
                # have the biggest gamma, so it should go first.
                for temp2 in range(n_gamma - 1, -1, -1):
                    if gamma <= gamma_buf[temp2]:
                        # We've found our place in the list: we're smaller than temp1,
                        # which means that we're smaller than everything further along.
                        temp1 = temp2 + 1
                        break

                if temp1 <= n_gamma - 1:
                    # Push back
                    gamma_buf[temp1 + 1:] = gamma_buf[temp1:n_gamma - 1]
                    gamma_idex_buf[temp1 + 1:] = gamma_idex_buf[temp1:n_gamma - 1]
                    # And insert
                    gamma_buf[temp1] = gamma
                    gamma_idex_temp[0] = i; gamma_idex_temp[1] = j; gamma_idex_temp[2] = k
                    gamma_idex_buf[temp1] = to_linear_index_3D(gamma_idex_temp, n_boxes)
                # == end for k
            # == end for j
        # == end for i

    assert np.min(delta_targets_np) >= -1 # -2's are unassigned, -1 is max pt.

    return box_vecs_np, density_np, point_box_assignments, deltas_np, delta_targets_np, gamma_buf_np, gamma_idex_buf_np


cpdef list dpc_assign(box_vecs_np,
                      density_np,
                      point_box_assignments,
                      deltas_np,
                      delta_targets_np,
                      precision density_threshold,
                      precision delta_threshold,
                      int n_halo_boxes = 0):
    """
    Returns:
        None. Fills point_box_assignments in-place with cluster assignments.
    """
    # === Assign boxes by DPC procedure
    cdef Py_ssize_t i,j,k, iteration, f_idex, pt_idex, component
    cdef Py_ssize_t idex_triple[3]
    # Nothing past this point is in space, so we proceed using linear
    # indexes and unraveled arrays:
    cdef Py_ssize_t n_boxes[3]
    n_boxes[:] = density_np.shape
    cdef Py_ssize_t n_total_boxes = n_boxes[0]*n_boxes[1]*n_boxes[2]

    cdef Py_ssize_t [:] delta_targets_linear = delta_targets_np.ravel()
    cdef precision [:, :, :] density = density_np
    cdef precision [:, :, :] deltas = deltas_np
    cdef precision [:] density_linear = density_np.ravel()

    # == Decide centers
    cdef list centers = []
    for i in range(n_boxes[0]):
        for j in range(n_boxes[1]):
            for k in range(n_boxes[2]):
                if density[i,j,k] >= density_threshold and \
                   deltas[i,j,k] >= delta_threshold:
                    idex_triple[0] = i; idex_triple[1] = j; idex_triple[2] = k
                    centers.append(to_linear_index_3D(idex_triple, n_boxes))

    # == Assign boxes to clusters
    box_cluster_assignments_np = np.full(n_total_boxes, CLUSTER_UNASSIGNED, dtype = np.intc)
    cdef int [:] box_cluster_assignments = box_cluster_assignments_np
    # Assign centers to themselves
    cdef Py_ssize_t center, center_id
    for center_id, center in enumerate(centers):
        box_cluster_assignments[center] = center_id

    # Assign all boxes to clusters
    cdef bint done = False
    for iteration in range(n_total_boxes):
        done = True
        for i in range(n_total_boxes):
            if box_cluster_assignments[i] == _CLUSTER_UNASSIGNED_C:
                # This one is unassigned, try to assign it
                box_cluster_assignments[i] = box_cluster_assignments[delta_targets_linear[i]]
            done = done & (not (box_cluster_assignments[i] == _CLUSTER_UNASSIGNED_C))

        if done:
            break

    assert done, "Somehow, box cluster assignment didn't finish. That shouldn't happen."

    # Make density zero boxes unassigned
    for i in range(n_total_boxes):
        if density_linear[i] == 0:
            box_cluster_assignments[i] = _CLUSTER_UNASSIGNED_C

    cdef precision [:] mean_halo_densities = np.zeros(shape = len(centers), dtype = density_np.dtype)
    cdef int [:] halos_count = np.zeros(shape = len(centers), dtype = np.intc)
    cdef precision dist_to_different
    cdef Py_ssize_t who_is_different
    cdef int [:, :, :] box_clusters_grid
    cdef int how_many_halos = 0
    if n_halo_boxes > 0:
        # == From the Rodriguez and Laio paper ==
        # Instead, we firstfind for each cluster a border region,
        # defined asthe set of points assigned to that cluster but
        # being within a distance dc from data points belonging to
        # other clusters. We then find, for each cluster, the
        # point of highest density within its border region.
        # We denote its density by rho. The points of the cluster
        # whose density is higher thanrbare consid-ered part of
        # the cluster core (robust assignation).The others are considered
        # part of the cluster halo (suitable to be considered as noise).
        #
        box_clusters_grid = box_cluster_assignments_np.reshape(density_np.shape)
        for i in range(n_boxes[0]):
            for j in range(n_boxes[1]):
                for k in range(n_boxes[2]):
                    if box_clusters_grid[i,j,k] == _CLUSTER_UNASSIGNED_C:
                        continue # We don't care, it isn't in a cluster

                    dist_to_different, who_is_different = min_dist_for_relation_pbc[int](
                        i,j,k,
                        box_clusters_grid, # as a grid
                        n_boxes, box_vecs_np,
                        RELATION_NE,
                        n_halo_boxes + 1,
                    )
                    if who_is_different >= 0:
                        how_many_halos += 1
                        #print(dist_to_different)
                        halos_count[box_clusters_grid[i,j,k]] += 1
                        mean_halo_densities[box_clusters_grid[i,j,k]] += density[i,j,k]
                        # We're in a halo
                        # if bigger, update the max
#                         if density[i,j,k] > max_halo_densities[box_clusters_grid[i,j,k]]:
#                             max_halo_densities[box_clusters_grid[i,j,k]] = density[i,j,k]

        # Do average
        for i in range(len(centers)):
            mean_halo_densities[i] /= halos_count[i]

        print("Found %i halos" % how_many_halos)
        # Mark boxes with lower densities as unassigned.
        for i in range(n_total_boxes):
            if density_linear[i] < mean_halo_densities[box_cluster_assignments[i]]:
                box_cluster_assignments[i] = _CLUSTER_UNASSIGNED_C

        # == end halos

    # == Assign points to clusters
    # (use the point_box_assignments array in place, since it's of the right
    # shape and because there could be a lot of points --- might be expensive
    # to allocate another such array.)
    cdef Py_ssize_t [:, :] point_box_assignments_c = point_box_assignments

    for f_idex in range(point_box_assignments_c.shape[0]):
        for pt_idex in range(point_box_assignments_c.shape[1]):
            point_box_assignments_c[f_idex, pt_idex] = box_cluster_assignments[point_box_assignments_c[f_idex, pt_idex]]

    return centers



cdef enum RELATION:
    RELATION_GT = 0
    RELATION_NE = 1

#TODO: file bug on const with fused type

ctypedef fused gridtype:
    precision
    int

cdef inline (precision, Py_ssize_t) min_dist_for_relation_pbc(const Py_ssize_t i,
                                                              const Py_ssize_t j,
                                                              const Py_ssize_t k,
                                                              gridtype [:, :, :] grid,
                                                              const Py_ssize_t* n_boxes,
                                                              const cell_precision [:, :] box_vecs,
                                                              const RELATION relation,
                                                              const Py_ssize_t max_radius,):
    """
    In a periodic grid, find the minimum distance from box (i,j,k) to another
    box whose value satisifies the given relation with box (i,j,k)'s value.
    """
    # Result
    cdef precision delta = INFINITY
    cdef Py_ssize_t delta_target = -1
    # Offsets
    cdef Py_ssize_t offset[3]
    cdef Py_ssize_t offset_idex[3]
    cdef precision offset_vec[3]
    # Loop variables:
    cdef Py_ssize_t radius, component, side_sign, side_i, side_j, temp1, temp2
    cdef precision curr_dist
    cdef bint done = False
    # Comparison values
    cdef gridtype value_main, value_offset
    value_main = grid[i, j, k]
    cdef bint curr_relation[2]
    # Misc
    cdef Py_ssize_t lookup_diff_component[3]
    lookup_diff_component[0] = 1; lookup_diff_component[1] = 2; lookup_diff_component[2] = 0

    for radius in range(1, max_radius): # For each box size
        for component in range(3): # For each box dimension
            for side_sign in range(-1, 2, 2): # For each box side --- -1 and 1
                for side_i in range(-radius, radius + 1): # TODO: change these bounds to not double count
                    for side_j in range(-radius, radius + 1):
                        # Build an offset
                        offset[0] = side_i; offset[1] = side_i; offset[2] = side_i
                        offset[component] = side_sign * radius
                        offset[lookup_diff_component[component]] = side_j
                        # % deals with periodicity
                        offset_idex[0] = (i + offset[0]) % n_boxes[0]
                        offset_idex[1] = (j + offset[1]) % n_boxes[1]
                        offset_idex[2] = (k + offset[2]) % n_boxes[2]
                        # Compute the offset vector
                        for temp1 in range(3):
                            offset_vec[temp1] = 0
                            for temp2 in range(3):
                                offset_vec[temp1] += offset[temp2] * \
                                                     box_vecs[temp2, temp1]
                        curr_dist = sqrt(
                            offset_vec[0]*offset_vec[0] + \
                            offset_vec[1]*offset_vec[1] + \
                            offset_vec[2]*offset_vec[2]
                        )

                        value_offset = grid[offset_idex[0], offset_idex[1], offset_idex[2]]
                        curr_relation[<int>RELATION_GT] = value_offset > value_main
                        curr_relation[<int>RELATION_NE] = value_offset != value_main

                        if curr_relation[<int>relation] and curr_dist < delta: # Must be closer to be the new delta
                            # we found a better delta
                            delta = curr_dist # Set the new delta
                            delta_target = to_linear_index_3D(offset_idex, n_boxes)
                            # We've found the "up gradient" for this box.
                            # Break out and move to the next:
                            done = True

                        # == end for side_j
                    # == end for side_i
                # == end for side_sign
            # == end for component
        # We check the whole cube first, but if we've found a
        # delta at this radius, is is the delta: no larger
        # radius is going to be closer.
        if done:
            break
        # == end for radius
    # We don't check done, since there could be no correct answer.
    # Especially if we have a max_radius.
    return delta, delta_target
