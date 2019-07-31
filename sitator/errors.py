
class SiteAnaysisError(Exception):
    """An error occuring as part of site analysis."""
    pass

class MultipleOccupancyError(SiteAnaysisError):
    """Error raised when multiple mobile atoms are assigned to the same site at the same time."""
    def __init__(self, mobile, site, frame):
        super().__init__(
            "Multiple mobile particles %s were assigned to site %i at frame %i." % (mobile, site, frame)
        )
        self.mobile_particles = mobile
        self.site = site
        self.frame = frame

class InsufficientSitesError(SiteAnaysisError):
    """Site detection/merging/etc. resulted in fewer sites than mobile particles."""
    def __init__(self, verb, n_sites, n_mobile):
        super().__init__("%s resulted in only %i sites for %i mobile particles." % (verb, n_sites, n_mobile))
        self.n_sites = n_sites
        self.n_mobile = n_mobile
