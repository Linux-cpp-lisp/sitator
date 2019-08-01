
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

    def run(self, sn, seed_mask = None):
        """
        Args:
            sn (SiteNetwork): Any sites will be ignored; needed for structure
                and static mask.
        Returns:
            A ``SiteNetwork``.
        """
        assert isinstance(sn, SiteNetwork)

        if seed_mask is None:
            seed_mask = sn.static_mask
        assert not np.any(seed_mask & sn.mobile_mask), "Seed mask must not overlap with mobile mask"
        assert not np.any(seed_mask & ~sn.static_mask), "All seed atoms must be static."
        voro_struct = sn.structure[seed_mask]
        translation = np.zeros(shape = len(sn.static_mask), dtype = np.int)
        translation[sn.static_mask] = np.arange(sn.n_static)
        translation = translation[seed_mask]

        with self._zeopy:
            nodes, verts, edges, _ = self._zeopy.voronoi(voro_struct,
                                                        radial = self._radial)

        out = sn.copy()
        out.centers = nodes
        out.vertices = [translation[v] for v in verts]

        return out
