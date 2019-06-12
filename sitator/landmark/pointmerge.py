
import numpy as np

# From https://github.com/tqdm/tqdm/issues/506#issuecomment-373126698
import sys
try:
    from tqdm.autonotebook import tqdm
except:
    def tqdm(iterable, **kwargs):
        return iterable

def merge_points_soap_paths(tsoap,
                            pbcc,
                            points,
                            connectivity_dict,
                            threshold,
                            n_steps = 5,
                            sanity_check_cutoff = np.inf):
    """Merge points using SOAP paths method.

    :param SOAP tsoap: to compute SOAPs with.
    :param dict connectivity_dict: Maps a point index to a set of point indexes
        it is connected to, however defined.
    :param threshold: Similarity threshold, 0 < threshold <= 1
    """

    merge_sets = set()

    points_along = np.empty(shape = (n_steps, 3), dtype = np.float)
    step_vec_mult = np.linspace(0.0, 1.0, num = n_steps)[:, np.newaxis]

    for pt_idex in tqdm(connectivity_dict.keys()):
        merge_set = set()
        current_pts = [pt_idex]
        from_soap = None
        keep_going = True
        while keep_going:
            added_this_iter = set()
            for edge_from in current_pts:
                offset = pbcc.cell_centroid - points[edge_from]
                edge_from_pt = pbcc.cell_centroid

                for edge_to in connectivity_dict[edge_from] - merge_set:
                    edge_to_pt = points[edge_to].copy()
                    edge_to_pt += offset
                    pbcc.wrap_point(edge_to_pt)

                    step_vec = edge_to_pt - edge_from_pt
                    edge_length = np.linalg.norm(step_vec)

                    assert edge_length <= sanity_check_cutoff, "edge_length %s" % edge_length

                    # Points along the line
                    for i in range(n_steps):
                        points_along[i] = step_vec
                    points_along *= step_vec_mult
                    points_along += edge_from_pt
                    # Re-center back to original center
                    points_along -= offset
                    # Wrap back into original unit cell - the one frame_atoms has
                    pbcc.wrap_points(points_along)

                    merge = tsoap.soaps_similar_for_points(points_along, threshold = threshold)

                    if merge:
                        added_this_iter.add(edge_from)
                        added_this_iter.add(edge_to)

            if len(added_this_iter) == 0:
                keep_going = False
            else:
                current_pts = added_this_iter - merge_set
                merge_set.update(added_this_iter)

        if len(merge_set) > 0:
            merge_sets.add(frozenset(merge_set))

    return merge_sets
