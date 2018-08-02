import numpy as np

from scipy.spatial import ConvexHull
from scipy.spatial.qhull import QhullError

from analysis import SiteTrajectory
from analysis.util import PBCCalculator

class SiteVolumes(object):
    """Computes the volumes of convex hulls around all positions associated with a site.

    Adds the `site_volumes` and `site_surface_areas` attributes to the SiteNetwork.
    """
    def __init__(self, n_recenterings = 8):
        self.n_recenterings = n_recenterings

    def run(self, st):
        vols = np.empty(shape = st.site_network.n_sites, dtype = np.float)
        areas = np.empty(shape = st.site_network.n_sites, dtype = np.float)

        pbcc = PBCCalculator(st.site_network.structure.cell)

        for site in xrange(st.site_network.n_sites):
            pos = st.real_positions_for_site(site)

            assert pos.flags['OWNDATA']

            vol = np.inf
            area = None
            for i in xrange(self.n_recenterings):
                # Recenter
                offset = pbcc.cell_centroid - pos[int(i * (len(pos)/self.n_recenterings))]
                pos += offset
                pbcc.wrap_points(pos)

                try:
                    hull = ConvexHull(pos)
                except QhullError as qhe:
                    print "For site %i, iter %i: %s" % (site, i, qhe)
                    vols[site] = np.nan
                    areas[site] = np.nan
                    continue

                if hull.volume < vol:
                    vol = hull.volume
                    area = hull.area

            vols[site] = vol
            areas[site] = area

        st.site_network.add_site_attribute('site_volumes', vols)
        st.site_network.add_site_attribute('site_surface_areas', areas)
