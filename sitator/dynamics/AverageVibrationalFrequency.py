import numpy as np

class AverageVibrationalFrequency(object):
    """Compute the average vibrational frequency of indicated atoms in a trajectory.

    Uses the method described in section 2.2 of this paper:

        Klerk, Niek J.J. de, Eveline van der Maas, and Marnix Wagemaker.
        “Analysis of Diffusion in Solid-State Electrolytes through MD Simulations,
            Improvement of the Li-Ion Conductivity in β-Li3PS4 as an Example.”
        ACS Applied Energy Materials 1, no. 7 (July 23, 2018): 3230–42.
        https://doi.org/10.1021/acsaem.8b00457.

    """
    def __init__(self,
                 min_frequency = 0,
                 max_frequency = np.inf):
        # Always want to exclude DC frequency
        assert min_frequency >= 0
        self.min_frequency = min_frequency
        self.max_frequency = max_frequency

    def compute_avg_vibrational_freq(self, traj, mask, return_stdev = False):
        """Compute the average vibrational frequency.

        Args:
            - traj (ndarray n_frames x n_atoms x 3)
            - mask (ndarray n_atoms bool): which atoms to average over.
        Returns:
            A frequency in units of (timestep)^-1
        """
        speeds = traj[1:, mask]
        speeds -= traj[:-1, mask]
        speeds = np.linalg.norm(speeds, axis = 2)

        freqs = np.fft.rfftfreq(speeds.shape[0])
        fmask = (freqs > self.min_frequency) & (freqs < self.max_frequency)
        assert np.any(fmask), "Trajectory too short?"
        freqs = freqs[fmask]

        # de Klerk et. al. do an average over the atom-by-atom averages
        n_mob = speeds.shape[1]
        avg_freqs = np.empty(shape = n_mob)

        for mob in range(n_mob):
            ps = np.abs(np.fft.rfft(speeds[:, mob])) ** 2
            avg_freqs[mob] = np.average(freqs, weights = ps[fmask])

        avg_freq = np.mean(avg_freqs)

        if return_stdev:
            return avg_freq, np.std(avg_freqs)
        else:
            return avg_freq
