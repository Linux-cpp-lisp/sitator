import numpy as np

from sitator.util.progress import tqdm
from sitator.util import DotProdClassifier

from sklearn.decomposition import IncrementalPCA

import logging
logger = logging.getLogger(__name__)

DEFAULT_PARAMS = {
    'clustering_threshold' : 0.9
    'assignment_threshold' : 0.9
}

def do_landmark_clustering(landmark_vectors,
                           clustering_params,
                           min_samples,
                           verbose = False):
    tmp = DEFAULT_PARAMS.copy()
    tmp.update(clustering_params)
    clustering_params = tmp

    pca = IncrementalPCA()
    pca.fit(landmark_vectors)
    keep_n_clusters = np.where(np.cumsum(pca.explained_variance_ratio_) >= clustering_params['clustering_threshold'])[0][0]

    landmark_classifier = \
        DotProdClassifier(threshold = np.nan, # We're not fitting
                          min_samples = min_samples)

    landmark_classifier.set_cluster_centers(pca.components_[:keep_n_clusters])

    lmk_lbls, lmk_confs = \
        landmark_classifier.fit_predict(landmark_vectors,
                                        predict_threshold = clustering_params['assignment_threshold'],
                                        verbose = verbose)

    return landmark_classifier.cluster_counts, lmk_lbls, lmk_confs
