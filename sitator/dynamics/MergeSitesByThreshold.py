import numpy as np

import operator

from scipy.sparse.csgraph import connected_components

from sitator.network.merging import MergeSites


class MergeSitesByThreshold(MergeSites):
    """Merge sites using a strict threshold on any edge property.

    Takes the edge property matrix given by `attrname`, applys `relation` to it
    with `threshold`, and merges all connected components in the graph represented
    by the resulting boolean adjacency matrix.

    Threshold is given by a keyword argument to `run()`.

    Args:
        - attrname (str): Name of the edge attribute to merge on.
        - relation (func, default: operator.ge): The relation to use for the
            thresholding.
        - directed, connection (bool, str): Parameters for scipy.sparse.csgraph's
            `connected_components`.
        - **kwargs: Passed to `MergeSites`.
    """
    def __init__(self,
                 attrname,
                 relation = operator.ge,
                 directed = True,
                 connection = 'strong',
                 **kwargs):
        self.attrname = attrname
        self.relation = relation
        self.directed = directed
        self.connection = connection
        super().__init__(**kwargs)


    def _get_sites_to_merge(self, st, threshold = 0):
        sn = st.site_network

        attrmat = getattr(sn, self.attrname)
        assert attrmat.shape == (sn.n_sites, sn.n_sites), "`attrname` doesn't seem to indicate an edge property."

        # Get mergable groups
        n_merged_sites, labels = connected_components(
            self.relation(attrmat, threshold),
            directed = self.directed,
            connection = self.connection
        )
        # MergeSites will check pairwise distances; we just need to make it the
        # right format.
        merge_groups = []
        for lbl in range(n_merged_sites):
            merge_groups.append(np.where(labels == lbl)[0])

        return merge_groups
