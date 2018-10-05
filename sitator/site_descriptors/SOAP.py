
import numpy as np
from abc import ABCMeta, abstractmethod

from sitator.SiteNetwork import SiteNetwork
from sitator.SiteTrajectory import SiteTrajectory
try:
    import quippy as qp
    from quippy import descriptors
except ImportError:
    raise ImportError("Quippy with GAP is required for using SOAP descriptors.")

from ase.data import atomic_numbers

DEFAULT_SOAP_PARAMS = {
    'cutoff' : 3.0,
    'cutoff_transition_width' : 1.0,
    'l_max' : 6, 'n_max' : 6,
    'atom_sigma' : 0.4
}

# From https://github.com/tqdm/tqdm/issues/506#issuecomment-373126698
import sys
try:
    ipy_str = str(type(get_ipython()))
    if 'zmqshell' in ipy_str:
        from tqdm import tqdm_notebook as tqdm
    if 'terminal' in ipy_str:
        from tqdm import tqdm
except:
    if sys.stderr.isatty():
        from tqdm import tqdm
    else:
        def tqdm(iterable, **kwargs):
            return iterable

class SOAP(object):
    """Abstract base class for computing SOAP vectors in a SiteNetwork.

    SOAP computations are *not* thread-safe; use one SOAP object per thread.

    :param int tracer_atomic_number = None: The type of tracer atom to add. If None,
        defaults to the type of the first mobile atom in the SiteNetwork.
    :param soap_mask: Which atoms in the SiteNetwork's structure
        to use in SOAP calculations.
        Can be either a boolean mask ndarray or a tuple of species.
        If `None`, the static_structure of the SiteNetwork will be used.
        Mobile atoms cannot be used for the SOAP host structure.
    :param dict soap_params = {}: Any custom SOAP params.
    """
    __metaclass__ = ABCMeta
    def __init__(self, tracer_atomic_number=None, soap_mask=None,
            soap_params={}, verbose =True):

        self._soap_mask = soap_mask
        self.tracer_atomic_number = tracer_atomic_number

        # -- Create the descriptor object
        soap_opts = dict(DEFAULT_SOAP_PARAMS)
        soap_opts.update(soap_params)
        soap_cmd_line = ["soap"]
        # User options
        for opt in soap_opts:
            soap_cmd_line.append("{}={}".format(opt, soap_opts[opt]))
        # Stuff that's the same no matter what
        # There's always only one tracer:
        soap_cmd_line.append("n_Z=1")
        # The species of that tracer is tracer_atomic_number
        soap_cmd_line.append("Z={{{}}}".format(self.tracer_atomic_number))

        self._soaper = descriptors.Descriptor(" ".join(soap_cmd_line))

        self.verbose = verbose

    @property
    def n_dim(self):
        return self._soaper.n_dim

    def get_descriptors(self, stn):
        """
        Get the descriptors.
        :param stn: A valid instance of SiteTrajectory or SiteNetwork
        :returns: an array of descriptor vectors and an equal length array of
            labels indicating which descriptors correspond to which sites.
        """
        # Build SOAP host structure
        if isinstance(stn, SiteTrajectory):
            structure, tracer_index, soap_mask = self._make_structure(stn.site_network)
        elif isinstance(stn, SiteNetwork):
            structure, tracer_index, soap_mask = self._make_structure(stn)
        else:
            raise TypeError("`stn` must be SiteNetwork or SiteTrajectory")

        # Compute descriptors
        return self._get_descriptors(stn, structure, tracer_index, soap_mask)

    # ----

    def _make_structure(self, sn):

        if self._soap_mask is None:
            # Make a copy of the static structure
            structure = qp.Atoms(sn.static_structure)
            soap_mask = sn.static_mask # soap mask is the 
        else:
            if isinstance(self._soap_mask, tuple):
                soap_mask = np.in1d(sn.structure.get_chemical_species(), self._soap_mask)
            else:
                soap_mask = self._soap_mask

            assert not np.any(soap_mask & sn.mobile_mask), "Error for atoms %s; No atom can be both static and mobile" % np.where(soap_mask & sn.mobile_mask)[0]
            structure = qp.Atoms(sn.structure[soap_mask])

        # Add a tracer
        if self.tracer_atomic_number is None:
            tracer_atomic_number = sn.structure.get_atomic_numbers()[sn.mobile_mask][0]
        else:
            tracer_atomic_number = self.tracer_atomic_number

        structure.add_atoms((0.0, 0.0, 0.0), tracer_atomic_number)
        structure.set_pbc([True, True, True])
        tracer_index = len(structure) - 1

        return structure, tracer_index, soap_mask

    @abstractmethod
    def _get_descriptors(self, stn, structure, tracer_index):
        pass




class SOAPCenters(SOAP):
    """Compute the SOAPs of the site centers in the fixed host structure.

    Requires a SiteNetwork as input.
    """
    def _get_descriptors(self, sn, structure, tracer_index, soap_mask):
        assert isinstance(sn, SiteNetwork), "SOAPCenters requires a SiteNetwork, not `%s`" % sn

        pts = sn.centers

        out = np.empty(shape = (len(pts), self.n_dim), dtype = np.float)

        structure.set_cutoff(self._soaper.cutoff())

        for i, pt in enumerate(tqdm(pts, desc="SOAP") if self.verbose else pts):
            # Move tracer
            structure.positions[tracer_index] = pt

            # SOAP requires connectivity data to be computed first
            structure.calc_connect()

            #There should only be one descriptor, since there should only be one Li
            out[i] = self._soaper.calc(structure)['descriptor'][0]

        return out, np.arange(sn.n_sites)


class SOAPSampledCenters(SOAPCenters):
    """Compute the SOAPs of representative points for each site, as determined by `sampling_transform`.

    Takes either a SiteNetwork or SiteTrajectory as input; requires that
    `sampling_transform` produce a SiteNetwork where `site_types` indicates
    which site in the original SiteNetwork/SiteTrajectory it was sampled from.

    Typical sampling transforms are `sitator.misc.NAvgsPerSite` (for a SiteTrajectory)
    and `sitator.misc.GenerateAroundSites` (for a SiteNetwork).
    """
    def __init__(self, *args, **kwargs):
        self.sampling_transform = kwargs.pop('sampling_transform', 1)
        super(SOAPSampledCenters, self).__init__(*args, **kwargs)

    def get_descriptors(self, stn):

        # Do sampling
        sampled = sampling_transform.run(stn)
        assert isinstance(sampling, SiteNetwork), "Sampling transform returned `%s`, not a SiteNetwork" % sampling

        # Compute actual dvecs
        dvecs, _ = super(SOAPSampledCenters, self).get_descriptors(sampled)

        # Return right corersponding sites
        return dvecs, sampled.site_types



class SOAPDescriptorAverages(SOAP):
    """Compute many instantaneous SOAPs for each site, and then average them in SOAP space.

    Computes the SOAP descriptors for mobile particles assigned to each site,
    in the host structure *as it was at that moment*. Those descriptor vectors are
    then averaged in SOAP space to give the final SOAP vectors for each site.

    This method often performs better than SOAPSampledCenters on more dynamic
    systems.

    :param int stepsize: Stride (in frames) when computing SOAPs
    :param int averaging: Number of average SOAP vectors to compute for each site.

    """
    def __init__(self, *args, **kwargs):
        self.stepsize = kwargs.pop('stepsize', 1)
        self.averaging = kwargs.pop('averaging', 1)
        super(SOAPDescriptorAverages, self).__init__(*args, **kwargs)


    def _get_descriptors(self, site_trajectory, structure, tracer_index, soap_mask):

        # the number of sites in the network
        nsit = site_trajectory.site_network.n_sites
        # I load the indices of the mobiles species into mob_indices:
        mob_indices = np.where(site_trajectory.site_network.mobile_mask)[0]

        # real_traj is the real space positions, site_traj the site trajectory
        # (i.e. for every mobile species the site index)
        # I load into new variable, only the steps I need (memory???)
        real_traj = site_trajectory._real_traj[::self.stepsize]
        site_traj = site_trajectory.traj[::self.stepsize]

        # Now, I need to allocate the output
        # so for each site, I count how much data there is!
        counts = np.array([np.count_nonzero(site_traj==site_idx) for site_idx in range(nsit)], dtype=int)
        nr_of_descs = counts // self.averaging
        if np.any(nr_of_descs == 0):
            raise ValueError("You are asking too much, averaging with {} gives a problem".format(self.averaging))
        # This is where I load the descriptor:
        descs = np.zeros((np.sum(nr_of_descs), self.n_dim))
        # An array that tells  me the index I'm at for each site type
        desc_index = [np.sum(nr_of_descs[:i]) for i in range(len(nr_of_descs))]
        max_index = [np.sum(nr_of_descs[:i+1]) for i in range(len(nr_of_descs))]

        count_of_site = np.zeros(len(nr_of_descs), dtype=int)
        blocked = np.empty(nsit, dtype=bool)
        blocked[:] = False
        structure.set_cutoff(self._soaper.cutoff())
        for site_traj_t, pos in zip(site_traj, real_traj):
            # I update the host lattice positions here, once for every timestep
            structure.positions[:tracer_index] = pos[soap_mask]
            for mob_idx, site_idx in enumerate(site_traj_t):
                if site_idx >= 0 and not blocked[site_idx]:
                    # Now, for every lithium that has been associated to a site of index site_idx,
                    # I take my structure and load the position of this mobile atom:
                    structure.positions[tracer_index] = pos[mob_indices[mob_idx]]
                    # calc_connect to calculated distance
                    structure.calc_connect()
                    #There should only be one descriptor, since there should only be one mobile
                    # I also divide  by averaging, to avoid getting into large numbers.
                    soapv =  self._soaper.calc(structure)['descriptor'][0] / self.averaging
                    # So, now I need to figure out where to load the soapv into desc
                    idx_to_add_desc = desc_index[site_idx]
                    descs[idx_to_add_desc,  :] += soapv
                    count_of_site[site_idx] += 1
                    # Now, if the count reaches the averaging I want, I augment
                    if count_of_site[site_idx] == self.averaging:
                        desc_index[site_idx] += 1
                        count_of_site[site_idx] = 0
                        # Now I check whether I have to block this site from accumulating more descriptors
                        if max_index[site_idx] == desc_index[site_idx]:
                            blocked[site_idx] = True

        desc_to_site = np.repeat(range(nsit), nr_of_descs)
        return descs, desc_to_site
