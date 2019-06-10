"""
quip.py: Compute SOAP vectors for given positions in a structure using the command line QUIP tool
"""

import numpy as np

import ase

from tempfile import NamedTemporaryFile
import subprocess

DEFAULT_SOAP_PARAMS = {
    'cutoff' : 3.0,
    'cutoff_transition_width' : 1.0,
    'l_max' : 6, 'n_max' : 6,
    'atom_sigma' : 0.4
}

def quip_soap_backend(soap_params = {}, quip_path = 'quip'):
    def backend(sn, soap_mask, tracer_atomic_number, environment_list):

        soap_opts = dict(DEFAULT_SOAP_PARAMS)
        soap_opts.update(soap_params)
        soap_cmd_line = ["soap"]

        # User options
        for opt in soap_opts:
            soap_cmd_line.append("{}={}".format(opt, soap_opts[opt]))

        #
        soap_cmd_line.append('n_Z=1 Z={{{}}}'.format(tracer_atomic_number))

        soap_cmd_line.append('n_species={} species_Z={{{}}}'.format(len(environment_list), ' '.join(map(str, environment_list))))

        soap_cmd_line = " ".join(soap_cmd_line)

        def soaper(structure, positions):
            structure = structure.copy()
            for i in range(len(positions)):
                structure.append(ase.Atom(position = tuple(positions[i]), symbol = tracer_atomic_number))
            return _soap(soap_cmd_line, structure, quip_path = quip_path)

        return soaper
    return backend


def _soap(descriptor_str,
         structure,
         quip_path = 'quip'):
    """Calculate SOAP vectors by calling `quip` as a subprocess.

    Args:
        - descriptor_str (str): The QUIP descriptor str, i.e. `soap cutoff=3 ...`
        - structure (ase.Atoms)
        - quip_path (str): Path to `quip` executable
    """

    with NamedTemporaryFile() as xyz:
        structure.write(xyz.name, format = 'extxyz')

        quip_cmd = [
            quip_path,
            "atoms_filename=" + xyz.name,
            "descriptor_str=\"" + descriptor_str + "\""
        ]

        result = subprocess.run(quip_cmd, stdout = subprocess.PIPE, check = True, text = True).stdout

    lines = result.splitlines()

    soaps = []
    for line in lines:
        if line.startswith("DESC"):
            soaps.append(np.fromstring(line.lstrip("DESC"), dtype = np.float, sep = ' '))
        elif line.startswith("Error"):
            e = subprocess.CalledProcessError(returncode = 0, cmd = quip_cmd)
            e.stdout = result
            raise e
        else:
            continue

    soaps = np.asarray(soaps)

    return soaps
