
import tempfile
import subprocess
import sys

import re
from collections import OrderedDict

import numpy as np

from sklearn.neighbors import KDTree

from sitator.util import PBCCalculator

def periodic_voronoi(structure, logfile = sys.stdout):
    """
    :param ASE.Atoms structure:
    """

    pbcc = PBCCalculator(structure.cell)

    # Make a 3x3x3 supercell
    supercell = structure.repeat((3, 3, 3))

    qhull_output = None

    logfile.write("Qvoronoi ---")

    # Run qhull
    with tempfile.NamedTemporaryFile('w',
                                     prefix = 'qvor',
                                     suffix='.in', delete = False) as infile, \
         tempfile.NamedTemporaryFile('r',
                                     prefix = 'qvor',
                                     suffix='.out',
                                     delete=True) as outfile:
        #  -- Write input file --
        infile.write("3\n") # num of dimensions
        infile.write("%i\n" % len(supercell)) # num of points
        np.savetxt(infile, supercell.get_positions(), fmt = '%.16f')
        infile.flush()

        cmdline = ["qvoronoi", "TI", infile.name, "FF", "Fv", "TO", outfile.name]
        process = subprocess.Popen(cmdline, stdout = subprocess.PIPE, stderr = subprocess.STDOUT)
        retcode = process.wait()
        logfile.write(process.stdout.read())
        if retcode != 0:
            raise RuntimeError("qvoronoi returned exit code %i" % retcode)

        qhull_output = outfile.read()

    facets_regex = re.compile(
                """
                -[ \t](?P<facetkey>f[0-9]+)  [\n]
                [ \t]*-[ ]flags: .* [\n]
                [ \t]*-[ ]normal: .* [\n]
                [ \t]*-[ ]offset: .* [\n]
                [ \t]*-[ ]center:(?P<center>([ ][\-]?[0-9]*[\.]?[0-9]*(e[-?[0-9]+)?){3}) [ \t] [\n]
                [ \t]*-[ ]vertices:(?P<vertices>([ ]p[0-9]+\(v[0-9]+\))+) [ \t]? [\n]
                [ \t]*-[ ]neighboring[ ]facets:(?P<neighbors>([ ]f[0-9]+)+)
                """, re.X | re.M)

    vertices_re = re.compile('(?<=p)[0-9]+')

    # Allocate stuff
    centers = []
    vertices = []
    facet_indexes_taken = set()

    facet_index_to_our_index = {}
    all_facets_centers = []

    # ---- Read facets
    facet_index = -1
    next_our_index = 0
    for facet_match in facets_regex.finditer(qhull_output):
        center = np.asarray(map(float, facet_match.group('center').split()))
        facet_index += 1

        all_facets_centers.append(center)

        if not pbcc.is_in_image_of_cell(center, (1, 1, 1)):
            continue

        verts = map(int, vertices_re.findall(facet_match.group('vertices')))
        verts_in_main_cell = tuple(v % len(structure) for v in verts)

        facet_indexes_taken.add(facet_index)

        centers.append(center)
        vertices.append(verts_in_main_cell)

        facet_index_to_our_index[facet_index] = next_our_index

        next_our_index += 1

        end_of_facets = facet_match.end()

    facet_count = facet_index + 1

    logfile.write("  qhull gave %i vertices; kept %i" % (facet_count, len(centers)))

    # ---- Read ridges
    qhull_output_after_facets = qhull_output[end_of_facets:].strip()
    ridge_re = re.compile('^\d+ \d+ \d+(?P<verts>( \d+)+)$', re.M)

    ridges = [
        [int(v) for v in match.group('verts').split()]
        for match in ridge_re.finditer(qhull_output_after_facets)
    ]
    # only take ridges with at least 1 facet in main unit cell.
    ridges = [
        r for r in ridges if any(f in facet_indexes_taken for f in r)
    ]

    # shift centers back into normal unit cell
    centers -= np.sum(structure.cell, axis = 0)

    nearest_center = KDTree(centers)

    ridges_in_main_cell = set()
    threw_out = 0
    for r in ridges:
        ridge_centers = np.asarray([all_facets_centers[f] for f in r if f < len(all_facets_centers)])
        if not pbcc.all_in_unit_cell(ridge_centers):
            continue

        pbcc.wrap_points(ridge_centers)
        dists, ridge_centers_in_main = nearest_center.query(ridge_centers, return_distance = True)

        if np.any(dists > 0.00001):
            threw_out += 1
            continue

        assert ridge_centers_in_main.shape == (len(ridge_centers), 1), "%s" % ridge_centers_in_main.shape
        ridge_centers_in_main = ridge_centers_in_main[:,0]

        ridges_in_main_cell.add(frozenset(ridge_centers_in_main))

    logfile.write("  Threw out %i ridges" % threw_out)

    logfile.flush()

    return centers, vertices, ridges_in_main_cell
