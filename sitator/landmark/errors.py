
class LandmarkAnalysisError(Exception):
    pass

class StaticLatticeError(LandmarkAnalysisError):
    """Error raised when static lattice atoms break any limits on their movement/position.

    Attributes:
        lattice_atoms (list, optional): The indexes of the atoms in the static lattice that
            caused the error.
        frame (int, optional): The frame in the trajectory at which the error occured.
    """
    TRY_RECENTERING_MSG = "Try recentering the input trajectory (sitator.util.RecenterTrajectory)"

    def __init__(self, message, lattice_atoms = None, frame = None, try_recentering = False):

        if try_recentering:
            message += "\n"
            message += StaticLatticeError.TRY_RECENTERING_MSG

        super(StaticLatticeError, self).__init__(message)

        self.lattice_atoms = lattice_atoms
        self.frame = frame

class ZeroLandmarkError(LandmarkAnalysisError):
    """Error raised when a landmark vector containing only zeros is encountered.

    Attributes:
        mobile_index (int): Which mobile atom had the all-zero vector.
        frame (int): At which frame it was encountered.
    """
    def __init__(self, mobile_index, frame):

        message = "Encountered a zero landmark vector for mobile ion %i at frame %i. Try increasing `cutoff_midpoint` and/or decreasing `cutoff_steepness`." % (mobile_index, frame)

        super(ZeroLandmarkError, self).__init__(message)

        self.mobile_index = mobile_index
        self.frame = frame

class MultipleOccupancyError(LandmarkAnalysisError):
    """Error raised when multiple mobile atoms are assigned to the same site."""
    pass
