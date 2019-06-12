
import numpy as np

DEFAULT_SOAP_PARAMS = {
    'cutoff' : 3.0,
    'l_max' : 6, 'n_max' : 6,
    'atom_sigma' : 0.4,
    'rbf' : 'gto',
    'crossover' : False
}

def dscribe_soap_backend(soap_params = {}):
    from dscribe.descriptors import SOAP

    soap_opts = dict(DEFAULT_SOAP_PARAMS)
    soap_opts.update(soap_params)

    def backend(sn, soap_mask, tracer_atomic_number, environment_list):

        def dscribe_soap(structure, positions):
            soap = SOAP(
                species = environment_list,
                crossover = soap_opts['crossover'],
                rcut = soap_opts['cutoff'],
                nmax = soap_opts['n_max'],
                lmax = soap_opts['l_max'],
                rbf = soap_opts['rbf'],
                sigma = soap_opts['atom_sigma'],
                periodic = np.all(structure.pbc),
                sparse = False
            )

            out = soap.create(structure, positions = positions).astype(np.float)
            return out

        return dscribe_soap

    return backend
