"""Cluster landmark vectors using the custom online algorithm from the original paper."""

from sitator.util import DotProdClassifier
from sitator.landmark import LandmarkAnalysis

DEFAULT_PARAMS = {
    'clustering_threshold' : 0.45,
    'assignment_threshold' : 0.8
}

def do_landmark_clustering(landmark_vectors,
                           clustering_params,
                           min_samples,
                           verbose):

    tmp = DEFAULT_PARAMS.copy()
    tmp.update(clustering_params)
    clustering_params = tmp

    landmark_classifier = \
        DotProdClassifier(threshold = clustering_params['clustering_threshold'],
                          min_samples = min_samples)
    lmk_lbls, lmk_confs = \
        landmark_classifier.fit_predict(landmark_vectors,
                                        predict_threshold = clustering_params['assignment_threshold'],
                                        verbose = verbose)

    return {
        LandmarkAnalysis.CLUSTERING_CLUSTER_SIZE : landmark_classifier.cluster_counts,
        LandmarkAnalysis.CLUSTERING_LABELS: lmk_lbls,
        LandmarkAnalysis.CLUSTERING_CONFIDENCES : lmk_confs,
        LandmarkAnalysis.CLUSTERING_REPRESENTATIVE_LANDMARKS : landmark_classifier.cluster_centers
    }
