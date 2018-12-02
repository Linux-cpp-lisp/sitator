
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

    :param int tracer_atomic_number: The atomic number of the tracer.
    :param list environment: The atomic numbers or atomic symbols
        of the environment to consider. I.e. for Li2CO3, can be set to ['O']  or [8]
        for oxygen only, or ['C', 'O'] / ['C', 8] / [6,8] if carbon and oxygen
        are considered an environment.
        Defaults to `None`, in which case all non-mobile atoms are considered
        regardless of species.
    :param soap_mask: Which atoms in the SiteNetwork's structure
        to use in SOAP calculations.
        Can be either a boolean mask ndarray or a tuple of species.
        If `None`, the entire static_structure of the SiteNetwork will be used.
        Mobile atoms cannot be used for the SOAP host structure.
        Even not masked, species not considered in environment will be not accounted for.
        For ideal performance: Specify environment and soap_mask correctly!
    :param dict soap_params = {}: Any custom SOAP params.
    """
    __metaclass__ = ABCMeta
    def __init__(self, tracer_atomic_number, environment = None,
            soap_mask=None, soap_params={}, verbose =True):
        from ase.data import atomic_numbers

        # Creating a dictionary for convenience, to check the types and values:
        self.tracer_atomic_number = 3
        centers_list = [self.tracer_atomic_number]
        self._soap_mask = soap_mask

        # -- Create the descriptor object
        soap_opts = dict(DEFAULT_SOAP_PARAMS)
        soap_opts.update(soap_params)
        soap_cmd_line = ["soap"]

        # User options
        for opt in soap_opts:
            soap_cmd_line.append("{}={}".format(opt, soap_opts[opt]))

        #
        soap_cmd_line.append('n_Z={} Z={{{}}}'.format(len(centers_list), ' '.join(map(str, centers_list))))

        # - Add environment species controls if given
        self._environment = None
        if not environment is None:
            if not isinstance(environment, (list, tuple)):
                raise TypeError('environment has to be a list or tuple of species (atomic number'
                    ' or symbol of the environment to consider')

            environment_list = []
            for e in environment:
                if isinstance(e, int):
                    assert 0 < e <= max(atomic_numbers.values())
                    environment_list.append(e)
                elif isinstance(e, str):
                    try:
                        environment_list.append(atomic_numbers[e])
                    except KeyError:
                        raise KeyError("You provided a string that is not a valid atomic symbol")
                else:
                    raise TypeError("Environment has to be a list of atomic numbers or atomic symbols")

            self._environment = environment_list
            soap_cmd_line.append('n_species={} species_Z={{{}}}'.format(len(environment_list), ' '.join(map(str, environment_list))))

        soap_cmd_line = " ".join(soap_cmd_line)

        if verbose:
            print("SOAP command line: %s" % soap_cmd_line)

        self._soaper = descriptors.Descriptor(soap_cmd_line)
        self._verbose = verbose
        self._cutoff = soap_opts['cutoff']



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

        assert np.any(soap_mask), "Given `soap_mask` excluded all host atoms."
        if not self._environment is None:
            assert np.any(np.isin(sn.structure.get_atomic_numbers()[soap_mask], self._environment)), "Combination of given `soap_mask` with the given `environment` excludes all host atoms."

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

        for i, pt in enumerate(tqdm(pts, desc="SOAP") if self._verbose else pts):
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
        sampled = self.sampling_transform.run(stn)
        assert isinstance(sampled, SiteNetwork), "Sampling transform returned `%s`, not a SiteNetwork" % sampled

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
    systems, but requires significantly more computation. 

    :param int stepsize: Stride (in frames) when computing SOAPs. Default 1.
    :param int averaging: Number of SOAP vectors to average for each output vector.
    :param int avg_descriptors_per_site: Can be specified instead of `averaging`.
        Specifies the _average_ number of average SOAP vectors to compute for each
        site. This does not guerantee that number of SOAP vectors for any site,
        rather, it allows a trajectory-size agnostic way to specify approximately
        how many descriptors are desired.

    """
    def __init__(self, *args, **kwargs):

        averaging_key = 'averaging'
        stepsize_key = 'stepsize'
        avg_desc_per_key = 'avg_descriptors_per_site'

        assert not ((averaging_key in kwargs) and (avg_desc_per_key in kwargs)), "`averaging` and `avg_descriptors_per_site` cannot be specified at the same time."

        self._stepsize = kwargs.pop(stepsize_key, 1)

        d = {stepsize_key : self._stepsize}

        if averaging_key in kwargs:
            self._averaging = kwargs.pop(averaging_key)
            d[averaging_key] = self._averaging
            self._avg_desc_per_site = None
        elif avg_desc_per_key in kwargs:
            self._avg_desc_per_site = kwargs.pop(avg_desc_per_key)
            d[avg_desc_per_key] = self._avg_desc_per_site
            self._averaging = None
        else:
            raise RuntimeError("Either the `averaging` or `avg_descriptors_per_site` option must be provided.")

        for k,v in d.items():
            if not isinstance(v, int):
                raise TypeError('{} has to be an integer'.format(k))
            if not ( v > 0):
                raise ValueError('{} has to be an positive'.format(k))
        del d # not needed anymore!

        super(SOAPDescriptorAverages, self).__init__(*args, **kwargs)


    def _get_descriptors(self, site_trajectory, structure, tracer_index, soap_mask):
        """
        calculate descriptors
        """
        # the number of sites in the network
        nsit = site_trajectory.site_network.n_sites
        # I load the indices of the mobiles species into mob_indices:
        mob_indices = np.where(site_trajectory.site_network.mobile_mask)[0]
        # real_traj is the real space positions, site_traj the site trajectory
        # (i.e. for every mobile species the site index)
        # I load into new variable, only the steps I need (memory???)
        real_traj = site_trajectory._real_traj[::self._stepsize]
        site_traj = site_trajectory.traj[::self._stepsize]

        # Now, I need to allocate the output
        # so for each site, I count how much data there is!
        counts = np.array([np.count_nonzero(site_traj==site_idx) for site_idx in xrange(nsit)], dtype=int)

        if self._averaging is not None:
            averaging = self._averaging
        else:
            averaging = int(np.floor(np.mean(counts) / self._avg_desc_per_site))

        nr_of_descs = counts // averaging

        if np.any(nr_of_descs == 0):
            raise ValueError("You are asking too much, averaging with {} gives a problem".format(averaging))
        # This is where I load the descriptor:
        descs = np.zeros((np.sum(nr_of_descs), self.n_dim))

        # An array that tells  me the index I'm at for each site type
        desc_index = [np.sum(nr_of_descs[:i]) for i in range(len(nr_of_descs))]
        max_index = [np.sum(nr_of_descs[:i+1]) for i in range(len(nr_of_descs))]

        count_of_site = np.zeros(len(nr_of_descs), dtype=int)
        blocked = np.empty(nsit, dtype=bool)
        blocked[:] = False
        structure.set_cutoff(self._soaper.cutoff())
        for site_traj_t, pos in tqdm(zip(site_traj, real_traj), desc="SOAP"):
            # I update the host lattice positions here, once for every timestep
            structure.positions[:tracer_index] = pos[soap_mask]
            for mob_idx, site_idx in enumerate(site_traj_t):
                if site_idx >= 0 and not blocked[site_idx]:
                    # Now, for every lithium that has been associated to a site of index site_idx,
                    # I take my structure and load the position of this mobile atom:
                    structure.positions[tracer_index] = pos[mob_indices[mob_idx]]
                    # calc_connect to calculated distance
#                     structure.calc_connect()
                    #There should only be one descriptor, since there should only be one mobile
                    # I also divide  by averaging, to avoid getting into large numbers.
#                     soapv =  self._soaper.calc(structure)['descriptor'][0] / self._averaging
                    structure.set_cutoff(self._cutoff)
                    structure.calc_connect()
                    soapv = self._soaper.calc(structure, grad=False)["descriptor"]

                    #~ soapv ,_,_ = get_fingerprints([structure], d)
                    # So, now I need to figure out where to load the soapv into desc
                    idx_to_add_desc = desc_index[site_idx]
                    descs[idx_to_add_desc,  :] += soapv[0] / averaging
                    count_of_site[site_idx] += 1
                    # Now, if the count reaches the averaging I want, I augment
                    if count_of_site[site_idx] == averaging:
                        desc_index[site_idx] += 1
                        count_of_site[site_idx] = 0
                        # Now I check whether I have to block this site from accumulating more descriptors
                        if max_index[site_idx] == desc_index[site_idx]:
                            blocked[site_idx] = True

        desc_to_site = np.repeat(range(nsit), nr_of_descs)
        return descs, desc_to_site
