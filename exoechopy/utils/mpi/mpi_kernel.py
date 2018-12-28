import os
import sys
import inspect
import multiprocessing
import tempfile
import numpy as np
from types import ModuleType, FunctionType
from mpi4py import MPI


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class MPIKernel ( object ):
  """

  Examples
  --------

  .. code-block::

    import exoechopy.utils.mpi as mpi
    import numpy as np

    reducer = lambda job, results: np.sum(np.array(results)) / (job.params.shape[0] * job.params.shape[1] )

    class TestKernel ( mpi.MPIKernel ):
      def process (self, param, input):
        return np.array([np.sum( param ** input[0] )])

    if __name__ == "__main__":

      kernel = TestKernel()

      kernel.run()

  """

  #-----------------------------------------------------------------------------
  def process ( self, param, input ):
    """Implement this method for a given kernel
    """
    pass

  #-----------------------------------------------------------------------------
  def run ( self ):
    """Runs the kernel
    """
    comm = MPI.Comm.Get_parent()
    size = comm.Get_size()
    rank = comm.Get_rank()

    # get length of each parameter
    params_length = np.empty( 1, dtype = np.int32 )
    comm.Bcast( params_length, root = 0 )
    params_length = params_length[0]

    # print("pid {} got param length {}".format(rank, params_length))

    # get number of parameters per process
    params_split_sizes = np.empty( size, dtype = np.int32 )
    comm.Bcast( params_split_sizes, root = 0 )

    # number of parameters for this process
    params_size = params_split_sizes[ rank ]

    print("pid {} got {} parameters".format(rank, params_size))

    # prepare parameters array
    params = np.empty([ params_size, params_length ])

    # get the parameters
    comm.Scatterv(
      None,
      params,
      root = 0 )

    # print("pid {} got params {}".format(rank, params))

    # get the input data
    input = comm.bcast( None, root = 0 )

    # print("pid {} got input {}".format(rank, input))

    # process all parameters and input data
    results = []

    for param in params:
      results.append( np.asarray( self.process(param, input) ) )

    comm.Barrier()

    results_sizes = np.array([ r.shape[0] for r in results ], dtype = np.int32 )

    print("pid {} sending results_sizes {}".format(rank, results_sizes))

    # send the size of each result
    comm.Gatherv(
      results_sizes,
      None,
      root = 0 )

    # get the global max results size
    max_size = np.empty(1, dtype = np.int32 )
    comm.Bcast( max_size, root = 0 )
    max_size = max_size[0]

    # print("pid {} got max_size {}".format(rank, max_size))

    # combine all results into a 2D array
    results_arr = np.zeros([ params_size, max_size ], dtype = np.float64 )

    for i, result in enumerate(results):

      results_arr[i, :results_sizes[i]] = result

    # print("pid {} sending results_arr {}".format(rank, results_arr))

    # send the results array
    comm.Gatherv(
      results_arr,
      None,
      root = 0 )


    comm.Disconnect()
