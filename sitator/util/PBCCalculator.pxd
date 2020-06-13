ctypedef double precision
ctypedef double cell_precision

cdef class PBCCalculator:

    cdef object _cell_np
    cdef cell_precision [:, :] _cell_mat_array
    cdef cell_precision [:, :] _cell_mat_inverse_array
    cdef cell_precision [:] _cell_centroid
    cdef cell_precision [:, :] _cell
    cdef cell_precision [:] _cell_vec_lengths

    cpdef pairwise_distances(self, pts, out = ?)

    cpdef distances(self, pt1, pts2, in_place = ?, out = ?)

    cpdef average(self, points, weights = ?)

    cpdef time_average(self, frames)

    cpdef void wrap_point(self, precision [:] pt)

    cpdef bint is_in_unit_cell(self, const precision [:] pt)

    cpdef bint all_in_unit_cell(self, const precision [:, :] pts)

    cpdef bint is_in_image_of_cell(self, const precision [:] pt, image)

    cpdef void to_cell_coords(self, precision [:, :] points)

    cpdef void to_cell_coords_wrapped(self, precision [:, :] points)

    cpdef int min_image(self, const precision [:] ref, precision [:] pt)

    cpdef void to_real_coords(self, precision [:, :] points)

    cpdef void wrap_points(self, precision [:, :] points)
