import numpy as np

from sitator import SiteTrajectory
from sitator.util import PBCCalculator
from sitator.util.progress import tqdm


class GenerateClampedTrajectory(object):
    """Create a real-space trajectory with the fixed site/static structure positions.

    Generate a real-space trajectory where the atoms are clamped to the fixed
    positions of the current site/their fixed static position.

    Args:
        - wrap (bool, default: False): If True, all clamped positions will be in
            the unit cell; if False, the clamped position will be the minimum
            image of the clamped position with respect to the real-space position.
            (This can generate a clamped, unwrapped real-space trajectory
            from an unwrapped real space trajectory.)
        - pass_through_unassigned (bool, default: False): If True, when a
            mobile atom is supposed to be clamped but is unassigned at some
            frame, its real-space position will be passed through from the
            real trajectory. If False, an error will be raised.
    """
    def __init__(self, wrap = False, pass_through_unassigned = False):
        self.wrap = wrap
        self.pass_through_unassigned = pass_through_unassigned


    def run(self, st, clamp_mask = None):
        """Create a real-space trajectory with the fixed site/static structure positions.

        Generate a real-space trajectory where the atoms indicated in `clamp_mask` --
        any mixture of static and mobile -- are clamped to: (1) the fixed position of
        their current site, if mobile, or (2) the corresponding fixed position in
        the `SiteNetwork`'s static structure, if static.

        Atoms not indicated in `clamp_mask` will have their positions from `real_traj`
        passed through.

        Args:
            - clamp_mask (ndarray, len(sn.structure))
        Returns:
            ndarray (n_frames x n_atoms x 3)
        """
        wrap = self.wrap
        pass_through_unassigned = self.pass_through_unassigned
        cell = st._sn.structure.cell
        pbcc = PBCCalculator(cell)

        n_atoms = len(st._sn.structure)
        if clamp_mask is None:
            clamp_mask = np.ones(shape = n_atoms, dtype = np.bool)
        if st._real_traj is None and not np.all(clamp_mask):
            raise RuntimeError("This `SiteTrajectory` has no real-space trajectory, but the given clamp mask leaves some atoms unclamped.")

        clamptrj = np.empty(shape = (st.n_frames, n_atoms, 3))
        # Pass through unclamped positions
        if not np.all(clamp_mask):
            clamptrj[:, ~clamp_mask, :] = st._real_traj[:, ~clamp_mask, :]
        # Clamp static atoms
        static_clamp = clamp_mask & st._sn.static_mask
        clamptrj[:, static_clamp, :] = st._sn.structure.get_positions()[static_clamp]
        # Clamp mobile atoms
        mobile_clamp = clamp_mask & st._sn.mobile_mask
        selected_sitetraj = st._traj[:, mobile_clamp]
        mobile_clamp_indexes = np.where(mobile_clamp)[0]
        if not pass_through_unassigned and np.min(selected_sitetraj) < 0:
            raise RuntimeError("The mobile atoms indicated for clamping are unassigned at some point during the trajectory and `pass_through_unassigned` is set to False. Try `assign_to_last_known_site()`?")

        if wrap:
            for frame_i in tqdm(range(len(clamptrj))):
                for mobile_i in mobile_clamp_indexes:
                    at_site = st._traj[frame_i, mobile_i]
                    if at_site == SiteTrajectory.SITE_UNKNOWN: # we already know that this means pass_through_unassigned = True
                        clamptrj[frame_i, mobile_i] = st._real_traj[frame_i, mobile_i]
                        continue
                    clamptrj[frame_i, mobile_i] = st._sn.centers[at_site]
        else:
            buf = np.empty(shape = (1, 3))
            site_pt = np.empty(shape = 3)
            for frame_i in tqdm(range(len(clamptrj))):
                for mobile_i in mobile_clamp_indexes:
                    buf[:, :] = st._real_traj[frame_i, mobile_i]
                    at_site = st._traj[frame_i, mobile_i]
                    if at_site == SiteTrajectory.SITE_UNKNOWN: # we already know that this means pass_through_unassigned = True
                        clamptrj[frame_i, mobile_i] = st._real_traj[frame_i, mobile_i]
                        continue
                    site_pt[:] = st._sn.centers[at_site]
                    pbcc.wrap_point(site_pt)
                    pbcc.wrap_points(buf)
                    site_mic = pbcc.min_image(buf[0], site_pt)
                    site_mic = [(site_mic // 10**(2 - i) % 10) - 1 for i in range(3)]
                    buf[:, :] = st._real_traj[frame_i, mobile_i]
                    pbcc.to_cell_coords(buf)
                    pt_in_image = np.floor(buf[0])
                    pt_in_image += site_mic
                    clamptrj[frame_i, mobile_i] =  np.dot(pt_in_image, cell) + st._sn.centers[at_site]

        return clamptrj
