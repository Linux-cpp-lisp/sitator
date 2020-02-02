import numpy as np

import tempfile
import tarfile
import os

import ase
import ase.io

from sitator import SiteNetwork

_STRUCT_FNAME = "structure.xyz"
_SMASK_FNAME = "static_mask.npy"
_MMASK_FNAME = "mobile_mask.npy"
_MAIN_FNAMES = ['centers', 'vertices', 'site_types']

def save(sn, file):
    """Save this SiteNetwork to a tar archive."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # -- Write the structure
        ase.io.write(os.path.join(tmpdir, _STRUCT_FNAME), sn.structure, parallel = False)
        # -- Write masks
        np.save(os.path.join(tmpdir, _SMASK_FNAME), sn.static_mask)
        np.save(os.path.join(tmpdir, _MMASK_FNAME), sn.mobile_mask)
        # -- Write what we have
        for arrname in _MAIN_FNAMES:
            if not getattr(sn, arrname) is None:
                np.save(os.path.join(tmpdir, "%s.npy" % arrname), getattr(sn, arrname))
        # -- Write all site/edge attributes
        for atype, attrs in zip(("site_attr", "edge_attr"), (sn._site_attrs, sn._edge_attrs)):
            for attr in attrs:
                np.save(os.path.join(tmpdir, "%s-%s.npy" % (atype, attr)), attrs[attr])
        # -- Write final archive
        with tarfile.open(file, mode = 'w:gz', format = tarfile.PAX_FORMAT) as outf:
            outf.add(tmpdir, arcname = "")


def from_file(file):
    """Load a SiteNetwork from a tar file/file descriptor."""
    all_others = {}
    site_attrs = {}
    edge_attrs = {}
    structure = None
    with tarfile.open(file, mode = 'r:gz', format = tarfile.PAX_FORMAT) as input:
        # -- Load everything
        for member in input.getmembers():
            if member.name == '':
                continue
            f = input.extractfile(member)
            if member.name == _STRUCT_FNAME:
                with tempfile.TemporaryDirectory() as tmpdir:
                    input.extract(member, path = tmpdir)
                    structure = ase.io.read(os.path.join(tmpdir, member.name), format = 'xyz')
            else:
                basename = os.path.splitext(os.path.basename(member.name))[0]
                data = np.load(f)
                if basename.startswith("site_attr"):
                    site_attrs[basename.split('-')[1]] = data
                elif basename.startswith("edge_attr"):
                    edge_attrs[basename.split('-')[1]] = data
                else:
                    all_others[basename] = data

    # Create SiteNetwork
    assert not structure is None
    assert all(k in all_others for k in ("static_mask", "mobile_mask")), "Malformed SiteNetwork file."
    sn = SiteNetwork(structure,
                     all_others['static_mask'],
                     all_others['mobile_mask'])
    if 'centers' in all_others:
        sn.centers = all_others['centers']
    for key in all_others:
        if key in ('centers', 'static_mask', 'mobile_mask'):
            continue
        setattr(sn, key, all_others[key])

    assert all(len(sa) == sn.n_sites for sa in site_attrs.values())
    assert all(ea.shape == (sn.n_sites, sn.n_sites) for ea in edge_attrs.values())
    sn._site_attrs = site_attrs
    sn._edge_attrs = edge_attrs

    return sn
