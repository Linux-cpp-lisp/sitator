#zeopy: simple Python interface to the Zeo++ `network` tool.
# Alby Musaelian 2018

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import os
import tempfile
import subprocess

import ase

# According to Zeo, "Must be in the .cssr, .cif, .cuc, .car, .arc or .v1 file format."
# See http://www.maciejharanczyk.info/Zeopp/input.html
# CIF is the only one ASE supports
ZEO_INFILE_FORMAT = "cif"

PIPE_PREFIX = "zeopy_"
PIPES = {'in' : 'cif', 'out' : 'nt2'}

class Zeopy(object):
    """An instance of the `network` tool that can be repeatedly invoked."""

    def __init__(self, path_to_zeo):
        """Create a Zeopy.

        :param str path_to_zeo: Path to the `network` executable.
        """
        self._exe = path_to_zeo
        self._tmpdir = None

    def _create_pipes(self):
        if self._tmpdir is None:
            self._tmpdir = tempfile.mkdtemp()
            self._pipes = {}
            for ptype in PIPES:
                path = os.path.join(self._tmpdir, PIPE_PREFIX + ptype + "." + PIPES[ptype])
                print(path)
                os.mkfifo(path)
                self._pipes[ptype] = path

    def voronoi(self, structure):
        """
        :param Atoms structure: The ASE Atoms to compute the Voronoi decomposition of.
        """
        # Write structure to pipe
        inp = open(self._pipes['in'], "w+")
        ase.io.write(inp, structure, format = PIPES['in'])

        # Run and output NT2 file
        outp = os.open(self._pipes['out'], os.O_RDONLY | os.O_NONBLOCK)
        self._run_network("nt2")

        where = None

        vertices = []
        edges = []

        nt2str = ""
        while True:
            buf = os.read(outp, 3000)
            print(buf)
            nt2str += buf
            if not buf: break

        print(nt2str)

        for l in nt2str.splitlines():
            if not l.strip():
                continue
            elif l.startswith("Vertex table:"):
                where = 'vertex'
            elif l.startswith("Edge table:"):
                where = 'edge'
            elif where == 'vertex':
                # Line format:
                # [node_number:int] [x] [y] [z] [radius] [region-vertex-atom-indexes]
                e = l.split()
                vertices.append({
                    'number' : e[0],
                    'coords' : e[1:4],
                    'radius' : e[4],
                    'region-atom-indexes' : e[5:]
                })
            elif where == 'edge':
                # Line format:
                # [from node] -> [to node] [radius] [delta uc x] ['' y] ['' z] [length]
                e = l.split()
                edges.append({
                    'from' : e[0],
                    'to' : e[2],
                    'radius' : e[3],
                    'delta_uc' : e[4:7],
                    'length' : e[7]
                })
            else:
                raise RuntimeError("Huh?")

        return vertices, edges

    def _run_network(self, outfile_format, args = []):
        # network [opts] -* output-file input-file
        output = subprocess.check_output([self._exe] + args + ["-" + outfile_format, self._pipes['out'], self._pipes['in']], stderr=subprocess.STDOUT)
        print(output)

    def __enter__(self):
        self.open()

    def __exit__(self, exctype, excval, trace):
        self.close()

    def open(self):
        self._create_pipes()

    def close(self):
        os.remove(self._tmpdir)
        self._tmpdir = None
