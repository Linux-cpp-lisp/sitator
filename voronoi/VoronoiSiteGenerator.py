
import numpy as np

from analysis.util import Zeopy

DEFAULT_ZEOPP = "/home/amusaelian/Documents/Ionic Frustration/code/lib/zeo++/trunk/network"

class VoronoiSiteGenerator(object):
    """Given an empty SiteNetwork, use the Voronoi decomposition to predict/generate sites."""

    def __init__(self, radial = False, verbose = True, zeopp_path = DEFAULT_ZEOPP):
        self._radial = radial
        self._verbose = verbose
        self._zeopy = Zeopy(zeopp_path)

    def run(self, sn):
        """SiteNetwork -> SiteNetwork"""

        nodes, verts, edges, _ = self._zeopy.voronoi(sn.static_structure,
                                                    radial = self._radial,
                                                    verbose = self._verbose)

        out = sn.copy()
        out.centers = nodes
        out.vertices = verts

        return out
