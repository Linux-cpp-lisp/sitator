import numpy as np

from analysis import SiteTrajectory

class ConfigurationalEntropy(object):
    """Compute the S~ configurational entropy descriptor.

    If the SiteTrajectory lacks type information, the summation is taken over
    the sites rather than the site types.

    Ref:
        Structural, Chemical, and Dynamical Frustration: Origins of Superionic Conductivity in closo-Borate Solid Electrolytes
        Kyoung E. Kweon, Joel B. Varley, Patrick Shea, Nicole Adelstein, Prateek Mehta, Tae Wook Heo, Terrence J. Udovic, Vitalie Stavila, and Brandon C. Wood
        Chemistry of Materials 2017 29 (21), 9142-9153
        DOI: 10.1021/acs.chemmater.7b02902
    """
    def __init__(self, acceptable_overshoot = 0.0, verbose = True):
        self.acceptable_overshoot = acceptable_overshoot
        self.verbose = verbose

    def compute(self, st):
        assert isinstance(st, SiteTrajectory)

        traj_re = np.reshape(st._traj, (st.site_network.n_mobile * st.n_frames,))
        traj_re = traj_re[traj_re >= 0]

        if not st.site_network.site_types is None:
            # By site type
            traj_re = st.site_network.site_types[traj_re]
            _, N_i = np.unique(st.site_network.site_types, return_counts = True)
        else:
            # By site
            N_i = np.ones(shape = st.site_network.n_sites)

        _, counts = np.unique(traj_re, return_counts = True)
        n_i = np.true_divide(counts, st.n_frames)
        p2 = np.true_divide(n_i, N_i)
        n = np.sum(n_i)

        # Correct overshoots
        problems = p2 > 1.0
        size_of_problems = p2 - 1.0
        forgivable = problems & (size_of_problems < self.acceptable_overshoot)

        if self.verbose:
            print "n_i      " + ("{:5.3} " * len(n_i)).format(*n_i)
            print "N_i      " + ("{:>5} " * len(N_i)).format(*N_i)
            print "         " + ("------" * len(n_i))
            print "P_2      " + ("{:5.3} " * len(p2)).format(*p2)

        if not np.all(problems == forgivable):
            raise ValueError("P_2 values for site types %s larger than 1.0 + acceptable_overshoot (%f)" % (np.where(problems)[0], self.acceptable_overshoot))
        elif np.any(problems) and self.verbose:
            print ""

        # Correct forgivable problems
        p2[forgivable] = 1.0

        return self.s_tilde(p2, n, N_i)

    def s_tilde(self, p2, n_occupied_sites, n_sites_of_type):
        # The interior of the sum goes to zero when P_2 = 1 or 0, but numpy doesn't know that
        nonzero = ~((p2 == 0.0) | (p2 == 1.0))
        p2 = p2[nonzero] # Only compute the summation for terms with nonzero contributions

        # Compute the elements of the summation
        inner_sum = n_sites_of_type[nonzero] * (p2 * np.log(p2) + (1.0 - p2) * np.log((1.0 - p2)))

        # k_B is 1 (?) in atomic units
        return -1.0  * np.sum(inner_sum) / n_occupied_sites
