import pytest

import numpy as np

from sitator.site_analysis.dpc.dpc_helpers import dpc_compute_vars, dpc_assign

@pytest.fixture(params = [
    (np.full(3, 0.4), np.full(3, 0.1)),
    ()
])
def gaussian_data_3d(mean, sigmas):
    return np.random.multivariate_normal(mean, np.diag(sigmas), size = 200)


@pytest.fixture
def dpc_var_values(gaussian_data_3d):
    return dpc_compute_vars(
        frames, pbcc,
        30, # n_boxes_max
        50, # n_gamma
        d_cutoff = 0.1
    )


def test_gammas(gaussian_data_3d)
