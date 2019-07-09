import numpy as np

from sitator.util.progress import tqdm
from sitator.util.mcl import markov_clustering
from sitator.util import DotProdClassifier
from ..helpers import _cross_correlation_matrix

import logging
logger = logging.getLogger(__name__)

DEFAULT_PARAMS = {
    'assignment_threshold' : 0.9
}

def do_landmark_clustering(landmark_vectors,
                           clustering_params,
                           min_samples,
                           verbose = False):
    tmp = DEFAULT_PARAMS.copy()
    tmp.update(clustering_params)
    clustering_params = tmp

    graph = _cross_correlation_matrix(landmark_vectors)

    # -- Cluster Landmarks
    clusters = markov_clustering(graph) # **clustering_params
    n_clusters = len(clusters)
    centers = np.zeros(shape = (n_clusters, landmark_vectors.shape[1]))
    for i, cluster in enumerate(clusters):
        centers[i, list(cluster)] = 1.0 # Set the peaks

    landmark_classifier = \
        DotProdClassifier(threshold = np.nan, # We're not fitting
                          min_samples = min_samples)

    landmark_classifier.set_cluster_centers(centers)

    lmk_lbls, lmk_confs = \
        landmark_classifier.fit_predict(landmark_vectors,
                                        predict_threshold = clustering_params['assignment_threshold'],
                                        verbose = verbose)

    return landmark_classifier.cluster_counts, lmk_lbls, lmk_confs
