import exoechopy.utils.mpi as mpi
from exoechopy.analyze.mpi import orbit_search_mpi_kernel

import numpy as np

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def run_mpi_orbit_search(
  observation_campaign,
  flare_catalog,
  lag_offset,
  max_lag,
  filter_width,
  num_interpolation_points,
  param_search ):

  parameter_order = [k for k,v in param_search.items()]

  # create single array where each row is one possible combination of parameters
  params = np.array( np.meshgrid( *[v for k,v in param_search.items()] ) ).T.reshape(-1, len(parameter_order) )

  input = {
    "observation_campaign" : observation_campaign
    "flare_catalog", flare_catalog,
    "lag_offset" : lag_offset,
    "max_lag" : max_lag,
    "filter_width" : filter_width,
    "num_interpolation_points" : num_interpolation_points,
    "parameter_order" : parameter_order }


  runner = mpi.MPIRunner(
    kernel = orbit_search_mpi_kernel )


  result = runner.run(
    num_processes = 8,
    job = mpi.MPIJob(
      params = params,
      input = input ))


  return parameter_order, params, result
