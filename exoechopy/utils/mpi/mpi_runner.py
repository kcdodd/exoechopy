import os
import sys
import inspect
import multiprocessing
import tempfile
import numpy as np
from types import ModuleType, FunctionType
from mpi4py import MPI

from exoechopy.utils.mpi import MPIJob, MPIKernel

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class MPIRunner ( object ):
  """

  Examples
  --------


  .. code-block::

    # test_mpi_runner.py

    import exoechopy.utils.mpi as mpi
    import numpy as np
    import test_mpi_kernel

    if __name__ == "__main__":

      runner = mpi.MPIRunner(
        kernel = test_mpi_kernel,
        reducer = test_mpi_kernel.reducer )

      result = runner.run(
        num_processes = 8,
        job = mpi.MPIJob(
          params = np.random.rand(16, 100),
          input = np.array([2]) ))

      print("Result: {}".format( result ))


  .. code-block::

    mpiexec python examples/test_mpi_runner.py

  """

  #-----------------------------------------------------------------------------
  def __init__ (self,
    kernel : ModuleType,
    reducer : FunctionType = None):
    """

    Parameters
    ----------
    kernel : module
      This is the processing kernel to run on the data for each set of parameters

    reducer : function (optional)
      This function is used to perform any final formatting of results into a single
      output result
    """

    # convert module to source code that will be called by mpi4py
    self._kernel = tempfile.NamedTemporaryFile( delete = False )
    self._kernel.write( str.encode(inspect.getsource(kernel)) )
    self._kernel.close()

    self._reducer = reducer

  #-----------------------------------------------------------------------------
  def __del__( self ):
    os.unlink( self._kernel.name )

  #-----------------------------------------------------------------------------
  def run(self,
    job : MPIJob,
    num_processes : int = None ):
    """

    Parameters
    ----------
    job : MPIJob
      The job parameters to divide and perform the processing

    num_processes : integer
      The number of processes to spawn

    """

    if num_processes is None:
        num_processes = max( multiprocessing.cpu_count()-1, 1 )

    comm = MPI.COMM_SELF.Spawn(
      sys.executable,
      args = [ self._kernel.name ],
      maxprocs = num_processes )

    size = num_processes

    print("Spawned {} processes".format(size))

    # print("broadcasting params length {}".format(job.params.shape[1]))
    # broadcast the length of the parameters array
    comm.bcast( np.array([ job.params.shape[1] ], dtype = np.int32 ), root = MPI.ROOT )

    # divide the total number of parameters equaly amonge the processes
    params_split = np.array_split( job.params, size, axis = 0 )

    # print("params_split {}".format(params_split))

    # determine how many params each process will receive
    params_split_sizes = np.array([arr.shape[0] for arr in params_split], dtype = np.int32 )

    print("broadcasting params sizes {}".format(params_split_sizes))
    # broadcast the *number* of parameters each process will consume
    comm.bcast( params_split_sizes, root = MPI.ROOT )

    # total to each process is the number of parameters each process will get
    # times the size of each parameters array
    params_split_sizes_total = params_split_sizes * job.params.shape[1]

    # first process has displacement 0, the next + number of parameters in the first process,
    # the next the sum of first two processes, etc.
    displacements = np.insert(np.cumsum(params_split_sizes), 0, 0)[0:-1]

    # print("scattering params {}".format(job.params))
    # initialize parameters specific to each process
    comm.Scatterv(
      [ job.params, params_split_sizes_total, displacements, MPI.DOUBLE ],
      None,
      root = MPI.ROOT )

    # print("broadcasting input {}".format(job.input))
    # send the data on which all processes will operate
    comm.bcast( job.input, root = MPI.ROOT )

    # wait until all processes are ready with results
    comm.Barrier()

    # get the sizes of each result set per parameter
    result_sizes = np.empty( [ job.params.shape[0] ], dtype = np.int32 )

    print("gathering result sizes")
    # each process will fill in the result sizes for the parameters it received
    comm.Gatherv(
      None,
      [ result_sizes, params_split_sizes, displacements, MPI.INT ],
      root = MPI.ROOT )

    print("Got result_sizes {}".format(result_sizes))

    max_size = np.amax(result_sizes)

    # print("broadcasting max size {}".format(max_size))
    # broadcast maximum result size to all processes
    comm.bcast( np.array([ max_size ], dtype = np.int32 ), root = MPI.ROOT )

    # all result arrays have to be the same length to return a single 2D array
    results_split_sizes_total = params_split_sizes * max_size

    # retrieve all results
    results = np.empty([ job.params.shape[0], max_size ], dtype = np.float64 )

    print("gathering results")

    comm.Gatherv(
      None,
      [ results, results_split_sizes_total, displacements, MPI.DOUBLE ],
      root = MPI.ROOT )

    # print("Got results {}".format(results))

    comm.Disconnect()

    results_list = []

    # split the results arrays back into their correct sizes
    for i, result in enumerate(results):

      l = result_sizes[i]

      results_list.append( result[:l] )

    if self._reducer is not None:
      results_list = self._reducer( job, results_list )

    return results_list
