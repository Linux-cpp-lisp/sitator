import numpy as np

import pytest
from pytest import approx

import itertools

from sitator.util import PBCCalculator
from sitator.site_analysis.dpc.density import gridded_density_periodic


def test_2x2_counts():
    """ Confirm that points in boxes are counted correctly """
    cell = np.diag([1, 1, 1.])
    # Points in three frames.
    points = np.array([
        [
            [0.25, 0.2, 0.1]
        ],
        [
            [0.12, 0.27, 0.1],
        ],
        [
            [0.78, 0.78, 0.1]
        ]
    ])
    # Transpose because this is slices along z-y-x rather than x-y-z
    should_be = np.transpose(np.array([
        [[2, 0],
         [0, 1.]],
        [[0, 0],
         [0, 0.]]
    ]))

    density, box_dims_out = gridded_density_periodic(
        points,
        n_boxes_max = 2,
        pbcc = PBCCalculator(cell),
        d_cutoff  = 0.,
    )
    assert np.all(density == should_be)


def test_3x3_counts():
    cell = np.diag([1, 1, 1.])
    points = np.array([[
        [0.25, 0.2, 0.1],
        [0.12, 0.37, 0.1],
        [0.78, 0.78, 0.5]
    ]])
    # Transpose because this is slices along z-y-x rather than x-y-z
    should_be = np.transpose(np.array([
        [[1, 0, 0],
         [1., 0., 0.],
         [0, 0, 0]],
        [[0, 0, 0],
         [0., 0., 0.],
         [0, 0, 1.]],
        [[0, 0, 0],
         [0., 0., 0.],
         [0, 0, 0]],
    ]))

    density, box_dims_out = gridded_density_periodic(
        points,
        n_boxes_max = 3,
        pbcc = PBCCalculator(cell),
        d_cutoff  = 0.,
    )
    assert np.all(density == should_be)


def test_1point_smeared():
    cell = np.diag([1, 1, 1.])
    points = np.array([[
        [0.25, 0.2, 0.1],
    ]])
    # Transpose because this is slices along z-y-x rather than x-y-z
    should_be = np.transpose(np.array([
        [[1, 1, 1],
         [1., 0., 0.],
         [1, 0, 0]],
        [[1, 0, 0],
         [0., 0., 0.],
         [0, 0, 0.]],
        [[1, 0, 0],
         [0., 0., 0.],
         [0, 0, 0]],
    ]))

    density, box_dims_out = gridded_density_periodic(
        points,
        n_boxes_max = 3,
        pbcc = PBCCalculator(cell),
        d_cutoff  = 0.4,
    )
    assert np.all(density == should_be)


def test_1point():
    cell = np.diag([1, 1, 1.])
    points = np.array([[
        [0.25, 0.2, 0.1],
    ]])
    # Transpose because this is slices along z-y-x rather than x-y-z
    should_be = np.transpose(np.array([
        [[1, 0, 0],
         [0, 0., 0.],
         [0, 0, 0]],
        [[0, 0, 0],
         [0., 0., 0.],
         [0, 0, 0.]],
        [[0, 0, 0],
         [0., 0., 0.],
         [0, 0, 0]],
    ]))

    box_indexes = np.empty(shape = (1, len(points)), dtype = np.int)

    density, box_dims_out = gridded_density_periodic(
        points,
        n_boxes_max = 3,
        pbcc = PBCCalculator(cell),
        output_box_assignments = box_indexes,
        d_cutoff  = 0.,
    )
    assert np.all(density == should_be)
    assert np.all(box_indexes == [0])


def test_1point_circle():
    cell = np.diag([1, 1, 1.])
    points = np.array([[
        [0.25, 0.2, 0.1],
    ]])
    n_box = 5
    density, box_dims_out = gridded_density_periodic(
        points,
        n_boxes_max = n_box,
        pbcc = PBCCalculator(cell),
        d_cutoff  = 1.1 * (1 / n_box), #radius length in one dim
    )
    assert np.sum(density) == 7 #Only straight line offsets get it

    density, box_dims_out = gridded_density_periodic(
        points,
        n_boxes_max = n_box,
        pbcc = PBCCalculator(cell),
        d_cutoff  = 1.1 * np.sqrt(2 * (1 / n_box)**2), #radius length in 2 dims
    )
    assert np.sum(density) == 27 - 8 # 3x3 cube minus corners


def test_bad_cell():
    cell = np.diag([1, 0, 1.])
    points = np.array([[
        [0.25, 0.2, 0.1],
    ]])
    n_box = 5
    with pytest.raises(ValueError):
        density, box_dims_out = gridded_density_periodic(
            points,
            n_boxes_max = n_box,
            pbcc = PBCCalculator(cell),
            d_cutoff  = 1.1 * (1 / n_box),
        )


def test_random_uniform():
    for trial in range(10):
        n_boxes = np.random.randint(2, 25, 3)
        should_be = np.empty(shape = tuple(n_boxes), dtype = np.int)
        points = []
        for idex in itertools.product(*[range(n) for n in n_boxes]):
            n_points = np.random.randint(0, 100)
            should_be[idex] = n_points
            if n_points == 0:
                continue
            # Put a random number of points in
            points.append(
                np.random.random_sample((n_points, 3)) + np.array(idex)
            )

        density, box_dims_out = gridded_density_periodic(
            np.concatenate(points)[np.newaxis, :],
            n_boxes_max = np.max(n_boxes),
            pbcc = PBCCalculator(np.diag(n_boxes).astype(np.float)),
            d_cutoff = 0,
        )
        assert np.asarray(box_dims_out) == approx(1.)
        assert np.sum(density) == np.sum(should_be)
        assert np.all(density == should_be)
