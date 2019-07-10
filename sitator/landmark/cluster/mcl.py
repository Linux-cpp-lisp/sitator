import numpy as np

from sitator.util.progress import tqdm
from sitator.util.mcl import markov_clustering
from sitator.util import DotProdClassifier
from ..helpers import _cross_correlation_matrix

from sklearn.covariance import empirical_covariance

import logging
logger = logging.getLogger(__name__)

DEFAULT_PARAMS = {
    'assignment_threshold' : 0.9
}

def cov2corr( A ):
    """
    covariance matrix to correlation matrix.
    """
    d = np.sqrt(A.diagonal())
    A = ((A.T/d).T)/d
    return A

def do_landmark_clustering(landmark_vectors,
                           clustering_params,
                           min_samples,
                           verbose):
    tmp = DEFAULT_PARAMS.copy()
    tmp.update(clustering_params)
    clustering_params = tmp

    cor = empirical_covariance(landmark_vectors)
    cor = cov2corr(cor)
    graph = np.clip(cor, 0, None)

    predict_threshold = clustering_params.pop('assignment_threshold')

    # -- Cluster Landmarks
    clusters = markov_clustering(graph, **clustering_params)
    n_clusters = len(clusters)
    centers = np.zeros(shape = (n_clusters, landmark_vectors.shape[1]))
    for i, cluster in enumerate(clusters):
        centers[i, list(cluster)] = 1 / len(cluster) # Set the peaks

    landmark_classifier = \
        DotProdClassifier(threshold = np.nan, # We're not fitting
                          min_samples = min_samples)

    landmark_classifier.set_cluster_centers(centers)

    lmk_lbls, lmk_confs = \
        landmark_classifier.fit_predict(landmark_vectors,
                                        predict_threshold = predict_threshold,
                                        predict_normed = False,
                                        verbose = verbose)

    return landmark_classifier.cluster_counts, lmk_lbls, lmk_confs
