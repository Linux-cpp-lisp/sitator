import warnings

import numpy as np

from tqdm import tqdm

class DotProdClassifier(object):
    def __init__(self,
                 threshold = 0.9,
                 max_converge_iters = 10):
        """
        :param float threshold: Similarity threshold for joining a cluster.
            In cos-of-angle-between-vectors (i.e. 1 is exactly the same, 0 is orthogonal)
        :param int max_converge_iters: Maximum number of iterations. If the algorithm hasn't converged
            by then, it will exit with a warning.
        """
        self._threshold = threshold
        self._max_iters = max_converge_iters
        self._cluster_centers = None
        self._cluster_counts = None

    @property
    def cluster_centers(self):
        return self._cluster_centers

    @property
    def cluster_counts(self):
        return self._cluster_counts

    def fit_predict(self, X, verbose = True):
        """ Fit the data vectors X and return their cluster labels.
        """
        # Essentially hierarchical clustering that stops when no cluster *centers*
        #  are more similar than the threshold.

        labels = np.empty(shape = len(X), dtype = np.int)

        # Start with each sample as a cluster
        old_centers = X
        old_n_assigned = [1] * len(X)

        # -- Classification loop

        # Maximum number of iterations
        last_n_sites = -1
        did_converge = False

        for _ in xrange(self._max_iters):
            # This iterations centers
            cluster_centers = list()
            n_assigned_to = list()

            for i, vec in enumerate(tqdm(old_centers) if verbose else old_centers):

                assigned_to = -1
                assigned_cosang = self._threshold

                for j, sitevec in enumerate(cluster_centers):
                    # OLD: only allowing merging into a larger/similar sized cluster
    #                 if not old_n_assigned[j] + 10 >= old_n_assigned[i]:
    #                     continue

                    cosang = np.dot(vec, sitevec)
                    cosang /= np.linalg.norm(vec) * np.linalg.norm(sitevec)

                    # Assign
                    if cosang > assigned_cosang:
                        assigned_to = j
                        assigned_cosang = cosang

                # If couldn't assign, start a new cluster
                if assigned_to == -1:
                    # New cluster!
                    cluster_centers.append(vec)
                    n_assigned_to.append(old_n_assigned[i])
                    assigned_to = len(cluster_centers) - 1
                else:
                    # Update average center vector of assigned cluster
                    cluster_centers[assigned_to] *= n_assigned_to[assigned_to]
                    cluster_centers[assigned_to] += vec
                    n_assigned_to[assigned_to] += old_n_assigned[i]
                    cluster_centers[assigned_to] /= n_assigned_to[assigned_to]

                # Add to label list
                labels[i] = assigned_to

            old_centers = cluster_centers
            old_n_assigned = n_assigned_to
            n_sites = len(n_assigned_to)

            # Check converged
            if last_n_sites == n_sites:
                did_converge = True
                break

            last_n_sites = n_sites

        if not did_converge:
            warnings.warn("Clustering for site type %i did NOT converge after %i iterations" % (site_type, max_converge_iters))

        self._cluster_centers = cluster_centers
        self._cluster_counts = n_assigned_to

        return labels
