import numpy as np

def within_species(sn):
    nums = sn.static_structure.numbers
    return [(nums == nums[s]).nonzero()[0] for s in range(sn.n_static)]
