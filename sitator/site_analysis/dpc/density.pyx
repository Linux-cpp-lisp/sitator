# cython: boundscheck=False, wraparound=False

cimport cython
from libc.math cimport sqrt, floor, ceil

import numpy as np

cimport sitator.util.PBCCalculator
from sitator.util.PBCCalculator cimport PBCCalculator, precision, cell_precision


@cython.cdivision
cdef inline Py_ssize_t to_linear_index_3D(const Py_ssize_t* idex, const Py_ssize_t* shape) nogil:
    """Convert a three-compoent (i, j, k) index into a row-major linear index"""
    return idex[0]*shape[1]*shape[2] + idex[1]*shape[2] + idex[2]


@cython.cdivision
cdef inline void from_linear_index_3D(Py_ssize_t lin_idex, const Py_ssize_t* shape, Py_ssize_t* out) nogil:
    """Convert a row-major linear index into a three-compoent (i, j, k) index"""
    out[0] = lin_idex / (shape[1]*shape[2]) # This is C-style floor division
    lin_idex -= out[0]*shape[1]*shape[2]
    out[1] = lin_idex / shape[2]
    lin_idex -= out[1]*shape[2]
    out[2] = lin_idex


cpdef gridded_density_periodic(frames,
                               int n_boxes_max,
                               PBCCalculator pbcc,
                               precision d_cutoff = 0.0,
                               which_pts = None,
                               output_box_assignments = None,
                               ):
    """Compute the number of points in a periodic grid of triclinic subcells.

    The longest crystal direction will have `n_boxes_max` gridboxes, and
    the other dimensions will have the number of gridboxes that make the
    grid boxes as regular (same side lengths) as possible.

    Params:
     - frames (ndarray): (n_frames, n_points, 3) input data.
     - n_boxes_max (int): How many boxes, at most, to place along each
        dimension. The size of the output grid is bounded by
        n_boxes_max^3.
     - pbcc (PBCCalculator): The periodic conditions under which to compute.
     - d_cutoff (precision): The cutoff radius for a point contributing to
        another box's density. If a box's centroid is within d_cutoff of
        the centroid of the box into which a point is placed, that point contributes
        +1 to the density of both boxes.
     - which_pts (ndarray): list of indexes of the points in each frame
        that should be considered. If None, all `n_points` points are considered.
     - output_box_assignments (ndarray): If not None (default None), an
        integer ndarray into which the box each input point is found to be in
        will be placed as a linear index. (See `to_linear_index_3D`.) Must be
        of shape `(n_frames, len(which_pts))`.
    Returns:
        (out, box_dims): out is the three-dimensional grid of densities;
            box_dims is a 3-element array giving the length of the dimension
            of the boxes along each cell vector.
    """
    # == Some variable set-ups
    cdef Py_ssize_t i, j
    cdef int x, y, z

    # == Set-up input data
    # TODO: extend to arbitrary dimensional case.
    if not (len(frames.shape) == 3 and frames.shape[2] == 3):
        raise ValueError("Frames must be of shape (n_frames, n_points, 3) --- only 3D data is supported.")

    # == Set up the cell
    cdef cell_precision [:] cell_vec_lengths = pbcc._cell_vec_lengths

    # == Determine number of boxes
    cdef Py_ssize_t n_boxes[3]
    cdef precision box_dims[3]

    cdef Py_ssize_t longest_dim = np.argmax(cell_vec_lengths)
    cdef precision box_side_length = cell_vec_lengths[longest_dim] / n_boxes_max
    for i in range(3):
        # Round n_boxes based on how close it puts the dimension of the box
        # on that side to box_side_length --- maximize box regularity (in the
        # regular polygon sense)
        #
        # Round down to start:
        n_boxes[i] = <Py_ssize_t>floor(cell_vec_lengths[i] / box_side_length)
        # If the error from rounding up is smaller than that of rounding down:
        if abs((cell_vec_lengths[i] / (n_boxes[i] + 1)) - box_side_length) \
            < abs((cell_vec_lengths[i] / n_boxes[i]) - box_side_length):
            # Then round up
            n_boxes[i] += 1
        box_dims[i] = cell_vec_lengths[i] / n_boxes[i]
    assert n_boxes[longest_dim] == n_boxes_max

    # == Build output array
    out = np.empty(shape = (n_boxes[0], n_boxes[1], n_boxes[2]), dtype = frames.dtype)
    cdef precision [:, :, :] out_c  = out
    out_c[:, :, :] = 0

    # == Make the weight mapping
    cdef int weight_width = <int> max([floor(d_cutoff / box_dims[i]) for i in range(3)])
    # One octant. +1 larger because there's always the center point
    weight_grid = np.zeros(shape = (weight_width + 1,) * 3, dtype = frames.dtype)
    cdef precision [:, :, :] weight_grid_c = weight_grid
    cdef precision temp_dist
    for x in range(weight_width + 1):
        for y in range(weight_width + 1):
            for z in range(weight_width + 1):
                temp_dist = sqrt(
                    (x * box_dims[0])*(x * box_dims[0]) +
                    (y * box_dims[0])*(y * box_dims[1]) +
                    (z * box_dims[0])*(z * box_dims[2])
                )
                if temp_dist <= d_cutoff:
                    weight_grid_c[x, y, z] = 1.0

    # == Frame-by-frame
    cdef Py_ssize_t [:] which_pts_c
    if which_pts is None:
        which_pts_c = np.arange(frames.shape[1])
    else:
        assert np.min(which_pts) >= 0
        assert np.max(which_pts) < frames.shape[1]
        which_pts_c = which_pts
    cdef Py_ssize_t n_which_pts = len(which_pts_c)

    cdef bint do_output_assign = 0
    cdef Py_ssize_t [:, :] output_box_assignments_c
    if output_box_assignments is not None:
        assert output_box_assignments.shape == (frames.shape[0], len(which_pts_c))
        output_box_assignments_c = output_box_assignments
        do_output_assign = 1

    cdef precision [:, :, :] frames_c = frames
    cdef precision [:, :] buf = np.empty(shape = (len(which_pts_c), 3), dtype = frames.dtype)
    cdef Py_ssize_t [3] idex
    cdef Py_ssize_t buf_idex
    for i in range(len(frames)):
        # Copy inputs into the buffer
        buf_idex = 0
        for j in range(n_which_pts):
            buf[buf_idex] = frames_c[i, which_pts_c[j]]
            buf_idex += 1

        # Bring points into cell coordinates
        pbcc.to_cell_coords_wrapped(buf)

        # Assign points to boxes an accumulate density
        for j in range(len(buf)):
            for k in range(3):
                idex[k] = <Py_ssize_t>floor(buf[j, k] * n_boxes[k])
            for x in range(-weight_width, weight_width + 1):
                for y in range(-weight_width, weight_width + 1):
                    for z in range(-weight_width, weight_width + 1):
                        out_c[
                            (idex[0] + x) % n_boxes[0],
                            (idex[1] + y) % n_boxes[1],
                            (idex[2] + z) % n_boxes[2],
                        ] += weight_grid_c[abs(x), abs(y), abs(z)]
            if do_output_assign:
                output_box_assignments_c[i, j] = to_linear_index_3D(idex, n_boxes)

    return out, box_dims
