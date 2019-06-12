
import numpy as np

import numbers
from sklearn.cluster import DBSCAN

import logging
logger = logging.getLogger(__name__)

DEFAULT_PARAMS = {
    'eps' : 0.05,
    'min_samples' : 5,
    'n_jobs' : -1
}

def do_landmark_clustering(landmark_vectors,
                           clustering_params,
                           min_samples,
                           verbose = False):
    # `verbose` ignored.

    tmp = DEFAULT_PARAMS.copy()
    tmp.update(clustering_params)
    clustering_params = tmp

    landmark_classifier = \
        DBSCAN(eps = clustering_params['eps'],
               min_samples = clustering_params['min_samples'],
               n_jobs = clustering_params['n_jobs'],
               metric = 'cosine')

    lmk_lbls = \
        landmark_classifier.fit_predict(landmark_vectors)

    # - Filter low occupancy sites
    cluster_counts = np.bincount(lmk_lbls[lmk_lbls >= 0])
    n_assigned = np.sum(cluster_counts)

    min_n_samples_cluster = None
    if isinstance(min_samples, numbers.Integral):
        min_n_samples_cluster = min_samples
    elif isinstance(min_samples, numbers.Real):
        min_n_samples_cluster = int(np.floor(min_samples * n_assigned))
    else:
        raise ValueError("Invalid value `%s` for min_samples; must be integral or float." % self._min_samples)

    to_remove_mask = cluster_counts < min_n_samples_cluster
    to_remove = np.where(to_remove_mask)[0]

    trans_table = np.empty(shape = len(cluster_counts) + 1, dtype = np.int)
    # Map unknown to unknown
    trans_table[-1] = -1
    # Map removed to unknwon
    trans_table[:-1][to_remove_mask] = -1
    # Map known to rescaled known
    trans_table[:-1][~to_remove_mask] = np.arange(len(cluster_counts) - len(to_remove))
    # Do the remapping
    lmk_lbls = trans_table[lmk_lbls]

    logging.info("DBSCAN landmark: %i/%i assignment counts below threshold %f (%i); %i clusters remain." % \
            (len(to_remove), len(cluster_counts), min_samples, min_n_samples_cluster, len(cluster_counts) - len(to_remove)))

    # Remove counts
    cluster_counts = cluster_counts[~to_remove_mask]

    # There are no confidences with DBSCAN, so just give everything confidence 1
    # so as not to screw up later weighting.
    confs = np.ones(shape = lmk_lbls.shape, dtype = np.float)
    confs[lmk_lbls == -1] = 0.0

    return cluster_counts, lmk_lbls, confs
