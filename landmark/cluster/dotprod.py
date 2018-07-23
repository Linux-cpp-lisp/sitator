
from util import DotProdClassifier

DEFAULT_PARAMS = {
    'clustering_threshold' : 0.45,
    'assignment_threshold' : 0.8
}

def do_landmark_clustering(landmark_vectors,
                           clustering_params,
                           min_samples,
                           verbose):

    clustering_params.update(DEFAULT_PARAMS)

    landmark_classifier = \
        DotProdClassifier(threshold = clustering_params['clustering_threshold'],
                          min_samples = min_samples)
    lmk_lbls, lmk_confs = \
        landmark_classifier.fit_predict(landmark_vectors,
                                        predict_threshold = clustering_params['assignment_threshold'],
                                        verbose = verbose)

    return landmark_classifier.cluster_counts, lmk_lbls, lmk_confs
