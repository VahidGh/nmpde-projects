#include "Heat.hpp"
#include <deal.II/base/numbers.h>
#include <cmath>

// Main function.
int
main(int argc, char *argv[])
{
  constexpr unsigned int dim = Heat::dim;

  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  const auto mu = [](const Point<dim> & /*p*/) { return 1.0; };

  const double a = 1.0;
  const unsigned int N = 1;
  const double sigma = 0.1;
  
  Point<dim> x_0;
  for (unsigned int d = 0; d < dim; ++d)
    x_0[d] = 0.5;

  const auto f  = [a, N, sigma, x_0](const Point<dim> &p, const double &t) {
    const double g_t = std::exp(-a * std::cos(2.0 * N * dealii::numbers::PI * t)) / std::exp(a);
    const double h_x = std::exp(-p.distance_square(x_0) / (sigma * sigma));
    return g_t * h_x;
  };

  Heat problem(/*mesh_filename = */ "../mesh/mesh-cube-10.msh",
               /* degree = */ 1,
               /* T = */ 1.0,
               /* theta = */ 1.0,
               /* delta_t = */ 0.0025,
               mu,
               f);

  problem.run();

  return 0;
}