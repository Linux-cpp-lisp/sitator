
import numpy as np

import quippy as qp
from quippy import descriptors
import mendeleev
from ase.data import atomic_numbers

DEFAULT_SOAP_PARAMS = {
    'cutoff' : 3.0,
    'cutoff_transition_width' : 1.0,
    'l_max' : 6, 'n_max' : 6,
    'atom_sigma' : 0.4
}

class SOAP(object):
    """Compute the SOAP vectors in a SiteNetwork

    :param SiteNetwork sn:
    :param int tracer_atomic_number = None: The type of tracer atom to add. If None,
        defaults to the type of the first mobile atom in the SiteNetwork.
    :param dict soap_params = {}: Any custom SOAP params.
    """

    def __init__(self, sn, tracer_atomic_number = None, soap_params = {}):

        # Make a copy of the structure
        self._structure = qp.Atoms(sn.static_structure)
        # Add a tracer
        if tracer_atomic_number is None:
            tracer_atomic_number = sn.structure.get_atomic_numbers()[sn.mobile_mask][0]

        self.tracer_atomic_number = tracer_atomic_number

        self._structure.add_atoms((0.0, 0.0, 0.0), tracer_atomic_number)
        self._tracer_index = len(self._structure) - 1

        # Create the descriptor
        soap_opts = dict(DEFAULT_SOAP_PARAMS)
        soap_opts.update(soap_params)
        soap_cmd_line = ["soap"]
        # User options
        for opt in soap_opts:
            soap_cmd_line.append("{}={}".format(opt, soap_opts[opt]))
        # Stuff that's the same no matter what
        soap_cmd_line.append("n_Z=1") #always one tracer
        soap_cmd_line.append("Z={{{}}}".format(self.tracer_atomic_number))

        self._soaper = descriptors.Descriptor(" ".join(soap_cmd_line))

    @property
    def n_dim(self):
        return self._soaper.n_dim

    def get_descriptors(self, pts, out = None):
        assert pts.ndim == 2 and pts.shape[1] == 3

        if out is None:
            out = np.empty(shape = (len(pts), self.n_dim), dtype = np.float)

        assert out.shape == (len(pts), self.n_dim)

        self._structure.set_cutoff(self._soaper.cutoff())

        for i, pt in enumerate(pts):
            # Move tracer
            self._structure.positions[self._tracer_index] = pt

            # SOAP requires connectivity data to be computed first
            self._structure.calc_connect()

            #There should only be one descriptor, since there should only be one Li
            out[i] = self._soaper.calc(self._structure)['descriptor'][0]

        return out

    def soaps_similar_for_points(self, pts, threshold = 0.95):
        """Determine if all SOAPs for points are at least threshold similar."""
        assert pts.ndim == 2 and pts.shape[1] == 3

        self._structure.set_cutoff(self._soaper.cutoff())

        initial_soap = None
        initial_soap_norm = None

        for i, pt in enumerate(pts):
            # Move tracer
            self._structure.positions[self._tracer_index] = pt

            # SOAP requires connectivity data to be computed first
            self._structure.calc_connect()

            #There should only be one descriptor, since there should only be one Li
            soap = self._soaper.calc(self._structure)['descriptor'][0]

            if initial_soap is None:
                initial_soap = soap.copy()
                initial_soap_norm = np.linalg.norm(initial_soap)
            else:
                similarity = np.dot(soap, initial_soap)
                similarity /= np.linalg.norm(soap)
                similarity /= initial_soap_norm

                if similarity < threshold:
                    return False

        return True
