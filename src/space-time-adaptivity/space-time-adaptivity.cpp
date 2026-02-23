#include "Heat.hpp"
#include <deal.II/base/numbers.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/multithread_info.h>
#include <cmath>
#include <iostream>
#include <string>

//  du/dt -div(mu * grad(u)) = f
int
main(int argc, char *argv[])
{
  constexpr unsigned int dim = Heat::dim;

  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  // Check if the user provided the mesh name argument
  if (argc < 2)
  {
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      std::cerr << "Error: Missing mesh name argument.\n"
                << "Usage:   " << argv[0] << " <mesh-name>\n"
                << "Example: mpirun -np 4 " << argv[0] << " mesh-cube-10\n";
    }
    return 1;
  }

  // Retrieve the mesh name from the command line
  std::string mesh_name = argv[1];
  
  // Automatically append ".msh" if the user didn't include it
  if (mesh_name.find(".msh") == std::string::npos)
  {
    mesh_name += ".msh";
  }

  // Construct the full path looking directly inside the ../mesh/ folder
  std::string mesh_path = "../mesh/" + mesh_name;

  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {
    std::cout << "Starting simulation using mesh: " << mesh_path << std::endl;
  }

  TimerOutput overall_timer(MPI_COMM_WORLD, std::cout,
                             TimerOutput::summary,
                            TimerOutput::wall_times);
  TimerOutput::Scope overall_scope(overall_timer, "Full Program Run");
  
  // Diffusion coefficient mu
  const auto mu = [](const Point<dim> & /*p*/) { 
    return 1.0; 
  };

  Heat::g_function g_pulsation;
  Heat::h_function h_spatial;

  // Forcing term f
  const auto f = [&g_pulsation, &h_spatial](const Point<dim> &p, const double &t) {
    // Set the current time for g(t)
    g_pulsation.set_time(t);
  
    // Return g(t) * h(x) 
    return g_pulsation.value(p) * h_spatial.value(p);
  };

  // Giving the parameters to the constructor
  Heat problem(/*mesh_file_name = */ mesh_path,
               /* degree = */ 2,
               /* T = */ 1.0,
               /* theta = */ 1.0,    
               /* delta_t = */ 0.001,
               mu,
               f);    

  problem.run();

  // Print final global statistics
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {
    std::cout << "\n===============================================" << std::endl;
    std::cout << "GLOBAL EXECUTION STATISTICS" << std::endl;
    std::cout << "-----------------------------------------------" << std::endl;
    std::cout << "  MPI Processes:        " << Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) << std::endl;
    std::cout << "  Threads Used:         " << MultithreadInfo::n_threads() << std::endl;
    std::cout << "  Output Files:         " << problem.get_timestep_number() << " VTU+PVTU records" << std::endl;
    std::cout << "===============================================\n" << std::endl;
  }

  return 0;
}