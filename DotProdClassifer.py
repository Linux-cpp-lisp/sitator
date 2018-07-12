import warnings

import numpy as np

from tqdm import tqdm

class DotProdClassifier(object):
    def __init__(self,
                 threshold = 0.9,
                 max_converge_iters = 10,
                 min_samples = 1):
        """
        :param float threshold: Similarity threshold for joining a cluster.
            In cos-of-angle-between-vectors (i.e. 1 is exactly the same, 0 is orthogonal)
        :param int max_converge_iters: Maximum number of iterations. If the algorithm hasn't converged
            by then, it will exit with a warning.
        :param int min_samples: any cluster with fewer samples will be filtered out before being
            returned.
        """
        self._threshold = threshold
        self._max_iters = max_converge_iters
        self._min_samples = min_samples
        self._cluster_centers = None
        self._cluster_counts = None
        self._featuredim = None

    @property
    def cluster_centers(self):
        return self._cluster_centers

    @property
    def cluster_counts(self):
        return self._cluster_counts

    def fit_predict(self, X, verbose = True):
        """ Fit the data vectors X and return their cluster labels.
        """

        assert len(X.shape) == 2, "Training data must be 2D."

        if self._featuredim is None:
            self._featuredim = X.shape[1]
        else:
            raise RuntimeError("DotProdClassifier cannot be fitted twice!")

        # Essentially hierarchical clustering that stops when no cluster *centers*
        #  are more similar than the threshold.

        labels = np.empty(shape = len(X), dtype = np.int)
        labels.fill(-1)

        # Start with each sample as a cluster
        old_centers = X
        old_n_assigned = [1] * len(X)
        old_members = [[i] for i in xrange(len(X))]

        # -- Classification loop

        # Maximum number of iterations
        last_n_sites = -1
        did_converge = False

        for _ in xrange(self._max_iters):
            # This iterations centers
            cluster_centers = list()
            n_assigned_to = list()
            members = list()

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
                    members.append(list())
                    members[-1].extend(old_members[i])
                    assigned_to = len(cluster_centers) - 1
                else:
                    # Update average center vector of assigned cluster
                    cluster_centers[assigned_to] *= n_assigned_to[assigned_to]
                    cluster_centers[assigned_to] += vec
                    #cluster_centers[assigned_to] = np.maximum(cluster_centers[assigned_to], vec)
                    n_assigned_to[assigned_to] += old_n_assigned[i]
                    members[assigned_to].extend(old_members[i])
                    cluster_centers[assigned_to] /= n_assigned_to[assigned_to]


            old_centers = cluster_centers
            old_n_assigned = n_assigned_to
            old_members = members

            assert [len(m) for m in members] == n_assigned_to, "%s\n%s" % (members, n_assigned_to)

            n_sites = len(n_assigned_to)

            # Check converged
            if last_n_sites == n_sites:
                did_converge = True
                break

            last_n_sites = n_sites

        if not did_converge:
            warnings.warn("Clustering for site type %i did NOT converge after %i iterations" % (site_type, max_converge_iters))

        self._cluster_centers = np.asarray(cluster_centers)
        self._cluster_counts = np.asarray(n_assigned_to)

        # filter out low counts
        assert len(self._cluster_counts) == len(self._cluster_centers)

        count_mask = self._cluster_counts > self._min_samples
        self._cluster_centers = self._cluster_centers[count_mask]
        self._cluster_counts = self._cluster_counts[count_mask]

        if verbose:
            print "DotProdClassifier: %i/%i assignment counts below threshold %i; %i clusters remain." % (np.sum(~count_mask), len(count_mask), self._min_samples, len(self._cluster_counts))

        # construct label list

        next_label = 0
        for clust in np.where(count_mask)[0]:
            labels[members[clust]] = next_label
            next_label += 1

        return labels

    def predict(self, X, return_confidences = False, threshold = None, verbose = True):
        """Return a predicted cluster label for vectors X.

        :param float threshold: alternate threshold. Defaults to None, when self.threshold
            is used.

        :returns: an array of labels. -1 indicates no assignment.
        :returns: an array of confidences in assignments. Normalzied
            values from 0 (no confidence, no label) to 1 (identical to cluster center).
        """

        assert len(X.shape) == 2, "Data must be 2D."

        if not X.shape[1] == (self._featuredim):
            raise TypeError("x has wrong dimension %s; should be (%i)" % (x.shape, self._featuredim))

        labels = np.empty(shape = len(X), dtype = np.int)

        if threshold is None:
            threshold = self._threshold

        confidences = None
        if return_confidences:
            confidences = np.empty(shape = len(X), dtype = np.float)

        for i, x in enumerate(tqdm(X) if verbose else X):

            assigned_to = -1
            assigned_cosang = threshold

            for j, sitevec in enumerate(self._cluster_centers):

                cosang = np.dot(x, sitevec)
                cosang /= np.linalg.norm(x) * np.linalg.norm(sitevec)

                # Assign
                if cosang > assigned_cosang:
                    assigned_to = j
                    assigned_cosang = cosang

            labels[i] = assigned_to

            if return_confidences:
                confidences[i] = assigned_cosang

        if return_confidences:
            return labels, confidences
        else:
            return labels
