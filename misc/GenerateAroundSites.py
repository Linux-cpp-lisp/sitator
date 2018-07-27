import numpy as np

from analysis import SiteNetwork
from analysis.util import PBCCalculator

class GenerateAroundSites(object):
    """Generate n normally distributed sites around each input site"""
    def __init__(self, n, sigma):
        self.n = n
        self.sigma = sigma

    def run(self, sn):
        assert isinstance(sn, SiteNetwork)
        out = sn.copy()
        pbcc = PBCCalculator(sn.structure.cell)

        print out.centers.shape
        newcenters = out.centers.repeat(self.n, axis = 0)
        print newcenters.shape
        newcenters += self.sigma * np.random.standard_normal(size = newcenters.shape)

        pbcc.wrap_points(newcenters)

        out.centers = newcenters

        return out
