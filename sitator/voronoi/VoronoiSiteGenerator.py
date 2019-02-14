
import numpy as np

from sitator import SiteNetwork
from sitator.util import Zeopy

class VoronoiSiteGenerator(object):
    """Given an empty SiteNetwork, use the Voronoi decomposition to predict/generate sites.

    :param str zeopp_path: Path to the Zeo++ `network` executable
    :param bool radial: Whether to use the radial Voronoi transform. Defaults to,
        and should typically be, False.
    :param bool verbose:
    """

    def __init__(self, zeopp_path = "network", radial = False, verbose = True):
        self._radial = radial
        self._verbose = verbose
        self._zeopy = Zeopy(zeopp_path)

    def run(self, sn):
        """SiteNetwork -> SiteNetwork"""
        assert isinstance(sn, SiteNetwork)

        with self._zeopy:
            nodes, verts, edges, _ = self._zeopy.voronoi(sn.static_structure,
                                                        radial = self._radial,
                                                        verbose = self._verbose)

        out = sn.copy()
        out.centers = nodes
        out.vertices = verts

        return out
