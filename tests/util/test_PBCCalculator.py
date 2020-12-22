import numpy as np

import pytest
from pytest import approx

from sitator.util import PBCCalculator

def test_cell_coords():
    """Confirm that `to_cell_coords` is equivalent to matrix inverse multiplication"""
    cells = [
        np.eye(3),
        np.array([
            [1, 0, 0],
            [0, 1, 0],
            [2, 2, 3.]
        ])
    ]
    for cell in cells:
        cell_points = np.random.random_sample((100, 3))
        points = np.matmul(cell.T, cell_points.T).T
        pointbuf = points.copy()
        pbcc = PBCCalculator(cell)
        pbcc.to_cell_coords_wrapped(pointbuf)
        assert pointbuf == approx(cell_points)
        pointbuf[:] = points
        pbcc.to_cell_coords_wrapped(pointbuf)
        assert pointbuf == approx(cell_points) # Nothing changed since these coefficients are already within-cell
