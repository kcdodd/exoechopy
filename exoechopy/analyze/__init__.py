
"""
This subpackage provides tools and functionality for analyzing lightcurves for echoes.
This includes flare identification, autocorrelators, statistical validation methods, and orbit simulations.
"""

from .autocorrelation import *
from .flare_manipulation import *
from .stats import *
from .orbit_search import *
import mpi
