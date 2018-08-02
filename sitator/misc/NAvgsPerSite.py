
import numpy as np

from sitator import SiteNetwork, SiteTrajectory
from sitator.util import PBCCalculator

class NAvgsPerSite(object):
    """Given a SiteTrajectory, return a SiteNetwork containing n avg. positions per site.

    The `types` of sites in the output are the index of the site in the input that generated
    that average.

    :param int n: How many averages to take
    :param bool error_on_insufficient: Whether to throw an error if n points cannot
        be provided for a site, or just take all that are available.
    :param bool weighted: Use SiteTrajectory confidences to weight the averages.
    """

    def __init__(self, n,
                 error_on_insufficient = True,
                 weighted = True):
        assert n % 2 == 0
        self.n = n
        self.error_on_insufficient = error_on_insufficient
        self.weighted = weighted

    def run(self, st):
        assert isinstance(st, SiteTrajectory)
        if st.real_trajectory is None:
            raise ValueError("SiteTrajectory must have associated real trajectory.")

        pbcc = PBCCalculator(st.site_network.structure.cell)
        # Maximum length
        centers = np.empty(shape = (self.n * st.site_network.n_sites, 3), dtype = st.real_trajectory.dtype)
        types = np.empty(shape = centers.shape[0], dtype = np.int)

        current_idex = 0
        for site in xrange(st.site_network.n_sites):
            if self.weighted:
                pts, confs = st.real_positions_for_site(site, return_confidences = True)
            else:
                pts = st.real_positions_for_site(site)
                confs = np.ones(shape = len(pts), dtype = np.int)

            old_idex = current_idex

            if len(pts) > self.n:
                sanity = 0
                for i in xrange(self.n):
                    ps = pts[i::self.n]
                    sanity += len(ps)
                    c = confs[i::self.n]
                    centers[current_idex] = pbcc.average(ps, weights = c)
                    current_idex += 1

                assert sanity == len(pts)
                assert current_idex - old_idex == self.n
            else:
                if self.error_on_insufficient:
                    raise ValueError("Insufficient points assigned to site %i (%i) to take %i averages." % (site, len(pts), self.n))
                centers[current_idex:current_idex + len(pts)] = pts
                current_idex += len(pts)

            types[old_idex:current_idex] = site

        sn = st.site_network.copy()
        sn.centers = centers
        sn.site_types = types

        return sn
