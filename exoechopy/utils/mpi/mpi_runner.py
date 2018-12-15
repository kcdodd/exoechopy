
import inspect
import tempfile
import numpy as np
from types import ModuleType
from mpi4py import MPI

class MPIJob ( object ):

  #-----------------------------------------------------------------------------
  def __init__ (self,
    params : list,
    input : list,
    result_size : int ):

    param_size = None

    for p in params:
      if not isinstance(p, np.ndarray):
        raise ValueError("Params must be an ndarray.")

      if param_size is None:
        param_size = p.shape[0]
      elif p.shape[0] != param_size:
        raise ValueError("All params must be the same size.")


    if not isinstance(input, np.ndarray):
      raise ValueError("Input must be an ndarray.")

    self.params = np.asarray(params)
    self.input = input
    self.result_size = result_size


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class MPIRunner ( object ):

  #-----------------------------------------------------------------------------
  def __init__ (self,
    kernel : ModuleType,
    reducer : FunctionType ):

    # create external source file of the module
    self._kernel = tempfile.NamedTemporaryFile( delete = False )
    self._kernel.write( inspect.getsource(kernel) )
    self._kernel.close()

    self._reducer = reducer

  #-----------------------------------------------------------------------------
  def run(self,
    num_processes : int = None,
    job : MPIJob ):

    if num_processes is None:
        num_processes = max( multi.cpu_count()-1, 1 )

    comm = MPI.COMM_SELF.Spawn(
      sys.executable,
      args = [ self._kernel.name ],
      maxprocs = num_processes )


    # divide the total number of parameters equaly amonge the processes
    params_split = np.array_split( job.params, comm.Get_size() )
    # determine how many params each process will receive
    split_sizes = np.array([arr.shape[0] for arr in param_split])

    # scatter the *number* of parameters each process will consume
    comm.Scatter( split_sizes, None, root = MPI.ROOT )

    split_sizes_total = split_sizes * job.params.shape[1]
    displacements = np.insert(np.cumsum(split_sizes_input),0,0)[0:-1]

    # initialize parameters specific to each process
    comm.Scatterv(
      [job.params, split_sizes_total, displacements, MPI.DOUBLE],
      None,
      root = MPI.ROOT )

    # send the data on which all processes will operate
    comm.Bcast( job.data, root = MPI.ROOT )

    # retrieve all results
    results = np.empty([ comm.Get_size(), job.result_size ], dtype = np.float64 )

    comm.Gatherv( None, recvbuf, root = MPI.ROOT )

    comm.Disconnect()

    if job.reducer is not None:
      results = job.reducer( results )

    return results
