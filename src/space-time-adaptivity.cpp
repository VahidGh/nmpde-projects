#include "Heat.hpp"

//  du/dt -div(mu * grad(u)) = f
int
main(int argc, char *argv[])
{
  constexpr unsigned int dim = Heat::dim;

  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);
  
  // Overall timer for the entire program execution (costruttore corretto)
  TimerOutput overall_timer(MPI_COMM_WORLD, std::cout,
                            TimerOutput::summary,
                            TimerOutput::wall_times);
  TimerOutput::Scope overall_scope(overall_timer, "Full Program Run");

  // Coefficiente di Diffusione (mu)
  const auto mu = [](const Point<dim> & /*p*/) { 
    return 1.0; 
  };

  Heat::g_function g_pulsation;
  Heat::h_function h_spatial;

  // Forzante (f)
  const auto f = [&g_pulsation, &h_spatial](const Point<dim> &p, const double &t) {
  // Impostiamo il tempo corrente per la funzione g(t)
  g_pulsation.set_time(t);
  
  // Il risultato è il prodotto g(t) * h(x) 
  return g_pulsation.value(p) * h_spatial.value(p);
  };

  // Istanziazione del problema
  Heat problem(/*mesh_file_name = */ "../mesh/mesh-cube-10.msh",
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
    std::cout << "  Output Files:         " << problem.get_timestep_number() << " VTU+PVTU records (approx. " << problem.get_timestep_number() * 0.5 << " MB per record)" << std::endl;
    std::cout << "===============================================\n" << std::endl;
  }
  
  // Il TimerOutput stamperà automaticamente il resoconto temporale esatto qui, 
  // quando viene distrutto alla fine del main!

  return 0;
}
