
import numpy as np

import os

from sitator import SiteNetwork
from sitator.util import Zeopy

class VoronoiSiteGenerator(object):
    """Given an empty SiteNetwork, use the Voronoi decomposition to predict/generate sites.

    :param str zeopp_path: Path to the Zeo++ ``network`` executable
    :param bool radial: Whether to use the radial Voronoi transform. Defaults to,
        and should typically be, ``False``.
    """

    def __init__(self,
                 zeopp_path = os.getenv("SITATOR_ZEO_PATH", default = "network"),
                 radial = False):
        self._radial = radial
        self._zeopy = Zeopy(zeopp_path)

    def run(self, sn):
        """
        Args:
            sn (SiteNetwork): Any sites will be ignored; needed for structure
                and static mask.
        Returns:
            A ``SiteNetwork``.
        """
        assert isinstance(sn, SiteNetwork)

        with self._zeopy:
            nodes, verts, edges, _ = self._zeopy.voronoi(sn.static_structure,
                                                        radial = self._radial)

        out = sn.copy()
        out.centers = nodes
        out.vertices = verts

        return out
