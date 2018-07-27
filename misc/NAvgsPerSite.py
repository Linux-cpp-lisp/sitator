
import numpy as np

from analysis.util import PBCCalculator

class NAvgsPerSite(object):
    """Given a SiteTrajectory, return a SiteNetwork containing n avg. positions per site.

    The `types` of sites in the output are the index of the site in the input that generated
    that average.
    """

    def __init__(self, n,
                 error_on_insufficient = False,
                 weighted = True):
        """
        :params int n: How many avgs. to take
        :
        """
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

            old_idex = current_idex

            if len(pts) < self.n:
                if self.error_on_insufficient:
                    raise ValueError("Insufficient points assigned to site %i (%i) to take %i averages." % (site, len(pts), self.n))
                centers[current_idex:current_idex + len(pts)] = pts
                current_idex += len(pts)
            else:
                # Actually do averages
                avg_len = int(np.floor(len(pts) / self.n))
                if avg_len % 2 != 0: avg_len -= 1
                n_avg, n_take = np.divmod(len(pts), avg_len)
                half = avg_len / 2

                buf = np.empty(shape = (avg_len, 3), dtype = st.real_trajectory.dtype)
                buf.fill(np.nan)

                confbuf = np.ones(shape = avg_len, dtype = np.float)
                for i in xrange(n_avg):
                    buf[:half] = pts[i * avg_len:i * avg_len + half]
                    half2end = -(i * avg_len)
                    if half2end == 0: half2end = None
                    buf[half:] = pts[-(i * avg_len + half):half2end]
                    if self.weighted:
                        confbuf[:half] = confs[i * avg_len:i * avg_len + half]
                        confbuf[half:] = confs[-(i * avg_len + half):half2end]
                    centers[current_idex] = pbcc.average(buf, weights = confbuf)
                    current_idex += 1

                for pt in pts[half * n_avg:-(half * avg_len)]:
                    centers[current_idex] = pt
                    current_idex += 1

                assert current_idex - old_idex == self.n

            types[old_idex:current_idex] = site

        sn = st.site_network.copy()
        sn.centers = centers
        sn.site_types = types

        return sn
