import numpy as np

from collections import Counter

import ase.data

from sitator.util.chemistry import identify_polyatomic_ions

import logging
logger = logging.getLogger(__name__)

def to_origin(sn):
    """Anchor all static atoms to the origin; i.e. their positions are absolute."""
    return np.full(
        shape = sn.n_static,
        fill_value = -1,
        dtype = np.int
    )

# ------

def within_polyatomic_ions(**kwargs):
    """Anchor the auxiliary atoms of a polyatomic ion to the central atom.

    In phosphate (PO4), for example, the four coordinating Oxygen atoms
    will be anchored to the central Phosphorous.

    Args:
        **kwargs: passed to ``sitator.util.chemistry.identify_polyatomic_ions``.
    """
    def func(sn):
        anchors = np.full(shape = sn.n_static, fill_value = -1, dtype = np.int)
        polyions = identify_polyatomic_ions(sn.static_structure, **kwargs)
        logger.info("Identified %i polyatomic anions: %s" % (len(polyions), Counter(i[0] for i in polyions)))
        for _, center, others in polyions:
            anchors[others] = center
        return anchors
    return func

# ------
# TODO
def to_heavy_elements(minimum_mass, maximum_distance = np.inf):
    """Anchor "light" elements to their nearest "heavy" element.

    Lightness/heaviness is determined by ``minimum_mass``, a cutoff in
    """
    def func(sn):
        pass
    return func
