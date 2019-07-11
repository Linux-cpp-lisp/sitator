import numpy as np

from sitator.util.progress import tqdm
from sitator.util.mcl import markov_clustering
from sitator.util import DotProdClassifier

from sklearn.covariance import empirical_covariance

from scipy.sparse.linalg import eigsh

import logging
logger = logging.getLogger(__name__)

DEFAULT_PARAMS = {
    'inflation' : 4,
    'assignment_threshold' : 0.7,
}

def cov2corr( A ):
    """
    covariance matrix to correlation matrix.
    """
    d = np.sqrt(A.diagonal())
    d[d == 0] = np.inf # Forces correlations to zero where variance is 0
    A = ((A.T/d).T)/d
    return A

def do_landmark_clustering(landmark_vectors,
                           clustering_params,
                           min_samples,
                           verbose):
    tmp = DEFAULT_PARAMS.copy()
    tmp.update(clustering_params)
    clustering_params = tmp

    n_lmk = landmark_vectors.shape[1]

    cov = empirical_covariance(landmark_vectors)
    corr = cov2corr(cov)
    graph = np.clip(corr, 0, None)
    for i in range(n_lmk):
        if graph[i, i] == 0: # i.e. no self correlation = 0 variance = landmark never seen
            graph[i, i] = 1 # Needs a self loop for Markov clustering not to degenerate. Arbitrary value, shouldn't affect anyone else.

    predict_threshold = clustering_params.pop('assignment_threshold')

    # -- Cluster Landmarks
    clusters = markov_clustering(graph, **clustering_params)
    clusters = [list(c) for c in clusters]
    n_clusters = len(clusters)
    centers = np.zeros(shape = (n_clusters, n_lmk))
    for i, cluster in enumerate(clusters):
        if len(cluster) == 1:
            centers[i, cluster] = 1.0 # Eigenvec is trivial case; scale doesn't matter either.
        else:
            # PCA inspired:
            eigenval, eigenvec = eigsh(cov[cluster][:, cluster], k = 1)
            centers[i, cluster] = eigenvec.T


    landmark_classifier = \
        DotProdClassifier(threshold = np.nan, # We're not fitting
                          min_samples = min_samples)

    landmark_classifier.set_cluster_centers(centers)

    lmk_lbls, lmk_confs, info = \
        landmark_classifier.fit_predict(landmark_vectors,
                                        predict_threshold = predict_threshold,
                                        predict_normed = True,
                                        verbose = verbose,
                                        return_info = True)

    msk = info['kept_clusters_mask']
    clusters = [c for i, c in enumerate(clusters) if msk[i]] # Only need the ones above the threshold

    return (
        landmark_classifier.cluster_counts,
        lmk_lbls,
        lmk_confs,
        clusters
    )
