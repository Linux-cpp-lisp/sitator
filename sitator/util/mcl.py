import numpy as np

def markov_clustering(transition_matrix,
                      expansion = 2,
                      inflation = 2,
                      pruning_threshold = 0.00001,
                      iterlimit = 100):
    """Compute the Markov Clustering of a graph.
    See https://micans.org/mcl/.

    Because we're dealing with matrixes that are stochastic already,
    there's no need to add artificial loop values.

    Implementation inspired by https://github.com/GuyAllard/markov_clustering
    """

    assert transition_matrix.shape[0] == transition_matrix.shape[1]

    # Check for nonzero diagonal -- self loops needed to avoid div by zero and NaNs
    assert np.count_nonzero(transition_matrix.diagonal()) == len(transition_matrix)

    m1 = transition_matrix.copy()

    # Normalize (though it should be close already)
    m1 /= np.sum(m1, axis = 0)

    allcols = np.arange(m1.shape[1])

    converged = False
    for i in range(iterlimit):
        # -- Expansion
        m2 = np.linalg.matrix_power(m1, expansion)
        # -- Inflation
        np.power(m2, inflation, out = m2)
        m2 /= np.sum(m2, axis = 0)
        # -- Prune
        to_prune = m2 < pruning_threshold
        # Exclude the max of every column
        to_prune[np.argmax(m2, axis = 0), allcols] = False
        m2[to_prune] = 0.0
        # -- Check converged
        if np.allclose(m1, m2):
            converged = True
            break

        m1[:] = m2

    if not converged:
        raise ValueError("Markov Clustering couldn't converge in %i iterations" % iterlimit)

    # -- Get clusters
    attractors = m2.diagonal().nonzero()[0]

    clusters = set()

    for a in attractors:
        cluster = tuple(m2[a].nonzero()[0])
        clusters.add(cluster)

    return list(clusters)
