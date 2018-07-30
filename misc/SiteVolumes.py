import numpy as np

from scipy.spatial import ConvexHull
from scipy.spatial.qhull import QhullError

from analysis import SiteTrajectory
from analysis.util import PBCCalculator

class SiteVolumes(object):
    """Computes the volumes of convex hulls around all positions associated with a site."""
    def __init__(self):
        pass

    def compute(self, st):
        vols = np.empty(shape = st.site_network.n_sites, dtype = np.float)
        areas = np.empty(shape = st.site_network.n_sites, dtype = np.float)

        pbcc = PBCCalculator(st.site_network.structure.cell)

        for site in xrange(st.site_network.n_sites):
            pos = st.real_positions_for_site(site)

            assert pos.flags['OWNDATA']

            # Recenter
            offset = pbcc.cell_centroid - pos[int(len(pos)/2)]
            pos += offset
            pbcc.wrap_points(pos)

            try:
                hull = ConvexHull(pos)
            except QhullError as qhe:
                print "For site %i: %s" % (site, qhe)
                vols[site] = np.nan
                areas[site] = np.nan
                continue
            vols[site] = hull.volume
            areas[site] = hull.area

        return vols, areas
