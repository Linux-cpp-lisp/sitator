import warnings

import numpy as np

import numbers

import sys
try:
    ipy_str = str(type(get_ipython()))
    if 'zmqshell' in ipy_str:
        from tqdm import tqdm_notebook as tqdm
    if 'terminal' in ipy_str:
        from tqdm import tqdm
except:
    if sys.stderr.isatty():
        from tqdm import tqdm
    else:
        def tqdm(iterable, **kwargs):
            return iterable

N_SITES_ALLOC_INCREMENT = 100

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
        :param int|float min_samples: filter out clusters with low sample counts.
            If an int, filters out clusters with fewer samples than this.
            If a float, filters out clusters with fewer than floor(min_samples * n_assigned_samples)
                samples assigned to them.
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

    @property
    def n_clusters(self):
        return len(self._cluster_counts)

    def fit_predict(self, X, verbose = True, predict_threshold = None, return_info = False):
        """ Fit the data vectors X and return their cluster labels.
        """

        assert len(X.shape) == 2, "Training data must be 2D."

        if self._featuredim is None:
            self._featuredim = X.shape[1]
        else:
            raise RuntimeError("DotProdClassifier cannot be fitted twice!")

        if predict_threshold is None:
            predict_threshold = self._threshold

        # Essentially hierarchical clustering that stops when no cluster *centers*
        #  are more similar than the threshold.

        labels = np.empty(shape = len(X), dtype = np.int)
        labels.fill(-1)

        # Start with each sample as a cluster
        old_centers = np.copy(X)
        old_n_assigned = np.ones(shape = len(X), dtype = np.int)
        old_n_clusters = len(X)

        # -- Classification loop

        # Maximum number of iterations
        last_n_sites = -1
        did_converge = False

        # preallocate buffers
        assert N_SITES_ALLOC_INCREMENT > 10
        diffs = np.empty(shape = N_SITES_ALLOC_INCREMENT, dtype = np.float)
        cluster_center_norms = np.empty(shape = N_SITES_ALLOC_INCREMENT, dtype = np.float)
        cluster_centers = np.empty(shape = (N_SITES_ALLOC_INCREMENT, X.shape[1]), dtype = X.dtype)
        n_assigned_to = np.empty(shape = N_SITES_ALLOC_INCREMENT, dtype = np.int)

        for iteration in xrange(self._max_iters):
            # This iteration's centers
            # The first sample is always its own cluster
            cluster_centers[0] = old_centers[0]
            cluster_center_norms[0] = np.linalg.norm(cluster_centers[0])
            n_assigned_to[0] = old_n_assigned[0]
            n_clusters = 1
            # skip the first sample which has already been accounted for
            for i, vec in tqdm(zip(xrange(1, old_n_clusters), old_centers[1:old_n_clusters]), desc = "Iteration %i" % iteration):

                assigned_to = -1
                assigned_cosang = 0.0

                np.dot(cluster_centers[:n_clusters], vec, out = diffs[:n_clusters])
                diffs[:n_clusters] /= cluster_center_norms[:n_clusters]
                diffs[:n_clusters] /= np.linalg.norm(vec)

                assigned_to = np.argmax(diffs[:n_clusters])
                assigned_cosang = diffs[:n_clusters][assigned_to]

                if assigned_cosang < self._threshold:
                    assigned_cosang = 0.0
                    assigned_to = -1

                # If couldn't assign, start a new cluster
                if assigned_to == -1:
                    # New cluster!
                    cluster_centers[n_clusters] = vec
                    n_assigned_to[n_clusters] = old_n_assigned[i]
                    assigned_to = n_clusters
                    # By definition, confidence is 1.0
                    assigned_cosang = 1.0
                    # Maintainance of norms
                    cluster_center_norms[n_clusters] = np.linalg.norm(vec)

                    n_clusters += 1

                    # Handle expansion
                    if n_clusters == len(diffs):
                        assert len(diffs) == len(cluster_center_norms)

                        diffs = np.empty(shape = len(diffs) + N_SITES_ALLOC_INCREMENT, dtype = diffs.dtype)

                        tmp_norms = np.empty(shape = len(cluster_center_norms) + N_SITES_ALLOC_INCREMENT, dtype = cluster_center_norms.dtype)
                        tmp_norms[:len(cluster_center_norms)] = cluster_center_norms
                        cluster_center_norms = tmp_norms

                        tmp_centers = np.empty(shape = (len(cluster_centers) + N_SITES_ALLOC_INCREMENT, cluster_centers.shape[1]), dtype = cluster_centers.dtype)
                        tmp_centers[:len(cluster_centers)] = cluster_centers
                        cluster_centers = tmp_centers

                        tmp_n_assigned = np.empty(shape = len(n_assigned_to) + N_SITES_ALLOC_INCREMENT, dtype = n_assigned_to.dtype)
                        tmp_n_assigned[:len(n_assigned_to)] = n_assigned_to
                        n_assigned_to = tmp_n_assigned

                else:
                    # Update average center vector of assigned cluster
                    assert assigned_to < n_clusters
                    cluster_centers[assigned_to] *= n_assigned_to[assigned_to]
                    cluster_centers[assigned_to] += vec
                    n_assigned_to[assigned_to] += old_n_assigned[i]
                    cluster_centers[assigned_to] /= n_assigned_to[assigned_to]
                    # Update center norm
                    cluster_center_norms[assigned_to] = np.linalg.norm(cluster_centers[assigned_to])

            old_centers[:n_clusters] = cluster_centers[:n_clusters]
            old_n_assigned[:n_clusters] = n_assigned_to[:n_clusters]
            old_n_clusters = n_clusters

            n_sites = n_clusters

            # Check converged
            if last_n_sites == n_sites:
                did_converge = True
                break

            last_n_sites = n_sites

        if not did_converge:
            raise ValueError("Clustering did not converge after %i iterations" % (self._max_iters))

        self._cluster_centers = np.asarray(cluster_centers[:n_clusters])

        # Run a predict now:
        labels, confs = self.predict(X, return_confidences = True, verbose = verbose, threshold = predict_threshold)

        total_n_assigned = np.sum(labels >= 0)

        # -- filter out low counts
        if not self._min_samples is None:
            self._cluster_counts = np.bincount(labels[labels >= 0])

            assert len(self._cluster_counts) == len(self._cluster_centers)

            min_samples = None
            if isinstance(self._min_samples, numbers.Integral):
                min_samples = self._min_samples
            elif isinstance(self._min_samples, numbers.Real):
                min_samples = int(np.floor(self._min_samples * total_n_assigned))
            else:
                raise ValueError("Invalid value `%s` for min_samples; must be integral or float." % self._min_samples)

            count_mask = self._cluster_counts >= min_samples

            self._cluster_centers = self._cluster_centers[count_mask]
            self._cluster_counts = self._cluster_counts[count_mask]

            if len(self._cluster_centers) == 0:
                # Then we removed everything...
                raise ValueError("`min_samples` too large; all %i clusters under threshold." % len(count_mask))

            if verbose:
                print "DotProdClassifier: %i/%i assignment counts below threshold %s (%s); %i clusters remain." % \
                    (np.sum(~count_mask), len(count_mask), self._min_samples, min_samples, len(self._cluster_counts))

            # Do another predict -- this could be more efficient, but who cares?
            labels, confs = self.predict(X, return_confidences = True, verbose = verbose, threshold = predict_threshold)

        if return_info:
            info = {
                'clusters_below_min_samples' : np.sum(~count_mask)
            }
            return labels, confs, info
        else:
            return labels, confs

    def predict(self, X, return_confidences = False, threshold = None, verbose = True, ignore_zeros = True):
        """Return a predicted cluster label for vectors X.

        :param float threshold: alternate threshold. Defaults to None, when self.threshold
            is used.

        :returns: an array of labels. -1 indicates no assignment.
        :returns: an array of confidences in assignments. Normalzied
            values from 0 (no confidence, no label) to 1 (identical to cluster center).
        """

        assert len(X.shape) == 2, "Data must be 2D."

        if not X.shape[1] == (self._featuredim):
            raise TypeError("X has wrong dimension %s; should be (%i)" % (X.shape, self._featuredim))

        labels = np.empty(shape = len(X), dtype = np.int)

        if threshold is None:
            threshold = self._threshold

        confidences = None
        if return_confidences:
            confidences = np.empty(shape = len(X), dtype = np.float)

        zeros_count = 0

        center_norms = np.linalg.norm(self._cluster_centers, axis = 1)

        # preallocate buffers
        diffs = np.empty(shape = len(center_norms), dtype = np.float)

        for i, x in enumerate(tqdm(X, desc = "Sample")):

            if np.all(x == 0):
                if ignore_zeros:
                    labels[i] = -1
                    zeros_count += 1
                    continue
                else:
                    raise ValueError("Data %i is all zeros!" % i)

            # diffs = np.sum(x * self._cluster_centers, axis = 1)
            # diffs /= np.linalg.norm(x) * center_norms
            np.dot(self._cluster_centers, x, out = diffs)
            diffs /= np.linalg.norm(x)
            diffs /= center_norms

            assigned_to = np.argmax(diffs)
            assignment_confidence = diffs[assigned_to]

            if assignment_confidence < threshold:
                assigned_to = -1
                assignment_confidence = 0.0

            labels[i] = assigned_to
            confidences[i] = assignment_confidence

        if verbose and zeros_count > 0:
            print "Encountered %i zero vectors during prediction" % zeros_count

        if return_confidences:
            return labels, confidences
        else:
            return labels
