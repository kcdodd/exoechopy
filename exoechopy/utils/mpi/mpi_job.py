import os
import sys
import inspect
import multiprocessing
import tempfile
import numpy as np
from types import ModuleType, FunctionType
from mpi4py import MPI

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class MPIJob ( object ):
  """Ecapsulates the job parameters, inputs, and outputs
  """
  #-----------------------------------------------------------------------------
  def __init__ (self,
    params : list,
    input : object ):
    """

    Parameters
    ----------
    params : list
      Each entry in the list of params represents a separate call to the processing
      kernel and will result in a separate entry in the results list. The parameters
      will be distributed approximately equal among the spawned kernel processes. All
      parameter arrays must be the same size or pass in MxN array

    input : list
      The same input is used for all processes and may be an pickleable object

    """
    param_size = None

    for p in params:
      if not isinstance(p, np.ndarray):
        raise ValueError("Params must be an ndarray.")

      if param_size is None:
        param_size = p.shape[0]
      elif p.shape[0] != param_size:
        raise ValueError("All params must be the same size.")

    self.params = np.asarray( params, dtype = np.float64 )
    self.input = input
