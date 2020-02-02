import numpy as np

# See discussion around this question: https://stackoverflow.com/questions/2018178/finding-the-best-trade-off-point-on-a-curve/2022348#2022348
def index_of_elbow(points):
    """Returns the index of the "elbow" in ``points``.

    Decently fast and pretty approximate. Performs worse with disproportionately
    long "flat" tails. For example, in a dataset with a nearly right-angle elbow,
    it overestimates the elbow by 1 starting at a before/after ratio of 1/4.
    """

    # Sort in descending order
    points = np.sort(points)[::-1]
    points = np.column_stack([np.arange(len(points)), points])

    line_vector = points[-1] - points[0]
    line_vector /= np.linalg.norm(line_vector)

    from_first = points - points[0]
    component_along_line = np.sum(from_first * line_vector, axis = 1)
    along_line = np.outer(component_along_line, line_vector)

    dist_to_line = np.linalg.norm(from_first - along_line, axis = 1)

    return np.argmax(dist_to_line)
