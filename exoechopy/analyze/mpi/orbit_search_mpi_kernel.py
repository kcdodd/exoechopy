import exoechopy as eep

from exoechopy.utils.mpi import MPIKernel
from exoechopy.analyze import OrbitSearch
import numpy as np

def reducer (job, results):
  """Combine results from all OrbitSearch instances
  """

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class OrbitSearchKernel ( MPIKernel ):
  """Perform OrbitSearch on multiple ranges of orbital parameters
  """

  #-----------------------------------------------------------------------------
  def process (self, param, input):

    planet = input.observation_campaign.gen_planets()[0]
    star = input.observation_campaign.gen_stars()[0]
    earth_vect = star.earth_direction_vector

    # Initialize our search object:
    dummy_object = eep.simulate.KeplerianOrbit(
      semimajor_axis=planet.semimajor_axis,
      eccentricity=planet.eccentricity,
      parent_mass=star.mass,
      initial_anomaly=planet.initial_anomaly,
      inclination=planet.inclination,
      longitude=planet.longitude,
      periapsis_arg=planet.periapsis_arg)

    all_resamples = np.random.choice(
      input.flare_catalog.num_flares,
      ( input.num_resamples, input.flare_catalog.num_flares ))


    orbit_search = OrbitSearch(
      input.flare_catalog,
      dummy_object,
      lag_offset = input.lag_offset,
      clip_range = (0, input.max_lag - input.filter_width))


    kw_params = {}

    for i, key in enumerate(input.parameter_order):
      kw_params[k] = param[i]

    search_result = orbit_search.run(
      earth_direction_vector = earth_vect,
      lag_metric = np.mean,
      num_interpolation_points = input.num_interpolation_points,
      resample_order = all_resamples,
      **kw_params )

    results = search_result.results

    lower, upper = np.percentile(results, [(100 - input.interval)/2, 50 + input.interval/2], axis=0)
    mean = np.mean(results, axis=0)

    return np.array([lower, mean, upper], dtype = np.float64 )

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
if __name__ == "__main__":

  kernel = OrbitSearchKernel()

  kernel.run()
