"""Cluster landmarks into sites using Markov Clustering and then assign each landmark vector.

Valid clustering params include:
 - ``"assignment_threshold"`` (float between 0 and 1): The similarity threshold
    below which a landmark vector will be marked unassigned.
 - ``"good_site_normed_threshold"`` (float between 0 and 1): The minimum for
    the cosine similarity between a good site's representative unit vector and
    its best match landmark vector.
 - ``"good_site_projected_threshold"`` (positive float): The minimum inner product
    between a good site's representative unit vector and its best match
    landmark vector.
 - All other params are passed along to `sitator.util.mcl.markov_clustering`.
"""

import numpy as np

from sitator.util.progress import tqdm
from sitator.util.mcl import markov_clustering
from sitator.util import DotProdClassifier
from sitator.landmark import LandmarkAnalysis

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
    # Center landmark vectors
    seen_ntimes = np.count_nonzero(landmark_vectors, axis = 0)
    cov = np.dot(landmark_vectors.T, landmark_vectors) / landmark_vectors.shape[0]
    corr = cov2corr(cov)
    graph = np.clip(corr, 0, None)
    for i in range(n_lmk):
        if graph[i, i] == 0: # i.e. no self correlation = 0 variance = landmark never seen
            graph[i, i] = 1 # Needs a self loop for Markov clustering not to degenerate. Arbitrary value, shouldn't affect anyone else.

    predict_threshold = clustering_params.pop('assignment_threshold')
    good_site_normed_threshold = clustering_params.pop('good_site_normed_threshold', predict_threshold)
    good_site_project_thresh = clustering_params.pop('good_site_projected_threshold', predict_threshold)

    # -- Cluster Landmarks
    clusters = markov_clustering(graph, **clustering_params)
    # Filter out single element clusters of landmarks that never appear.
    clusters = [list(c) for c in clusters if seen_ntimes[c[0]] > 0]
    n_clusters = len(clusters)
    centers = np.zeros(shape = (n_clusters, n_lmk))
    maxbuf = np.empty(shape = len(landmark_vectors))
    good_clusters = np.zeros(shape = n_clusters, dtype = np.bool)
    for i, cluster in enumerate(clusters):
        if len(cluster) == 1:
            eigenvec = [1.0] # Eigenvec is trivial
        else:
            # PCA inspired:
            _, eigenvec = eigsh(cov[cluster][:, cluster], k = 1)
            eigenvec = eigenvec.T
        centers[i, cluster] = eigenvec
        np.dot(landmark_vectors, centers[i], out = maxbuf)
        np.abs(maxbuf, out = maxbuf)
        best_match = np.argmax(maxbuf)
        best_match_lvec = landmark_vectors[best_match]
        best_match_dot = np.abs(np.dot(best_match_lvec, centers[i]))
        best_match_dot_norm = best_match_dot / np.linalg.norm(best_match_lvec)
        good_clusters[i] = best_match_dot_norm >= good_site_normed_threshold
        good_clusters[i] &= best_match_dot >= good_site_project_thresh
        centers[i] /= best_match_dot

    logger.debug("Kept %i/%i landmark clusters as good sites" % (np.sum(good_clusters), len(good_clusters)))

    # Filter out "bad" sites
    clusters = [c for i, c in enumerate(clusters) if good_clusters[i]]
    centers = centers[good_clusters]
    n_clusters = len(clusters)

    landmark_classifier = \
        DotProdClassifier(threshold = np.nan, # We're not fitting
                          min_samples = min_samples)

    landmark_classifier.set_cluster_centers(centers)

    lmk_lbls, lmk_confs, info = \
        landmark_classifier.fit_predict(landmark_vectors,
                                        predict_threshold = predict_threshold,
                                        predict_normed = False,
                                        verbose = verbose,
                                        return_info = True)

    msk = info['kept_clusters_mask']
    clusters = [c for i, c in enumerate(clusters) if msk[i]] # Only need the ones above the threshold

    # Find the average landmark vector at each site
    weighted_reps = clustering_params.get('weighted_representative_landmarks', True)
    centers = np.zeros(shape = (len(clusters), n_lmk))
    weights = np.empty(shape = lmk_lbls.shape)
    for site in range(len(clusters)):
        np.equal(lmk_lbls, site, out = weights)
        if weighted_reps:
            weights *= lmk_confs
        centers[site] = np.average(landmark_vectors, weights = weights, axis = 0)


    return {
        LandmarkAnalysis.CLUSTERING_CLUSTER_SIZE : landmark_classifier.cluster_counts,
        LandmarkAnalysis.CLUSTERING_LABELS : lmk_lbls,
        LandmarkAnalysis.CLUSTERING_CONFIDENCES: lmk_confs,
        LandmarkAnalysis.CLUSTERING_LANDMARK_GROUPINGS : clusters,
        LandmarkAnalysis.CLUSTERING_REPRESENTATIVE_LANDMARKS : centers
    }
