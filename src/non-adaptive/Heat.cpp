#include "Heat.hpp"

// Constructor definition
Heat::Heat(const std::string                              &mesh_file_name_, 
           const unsigned int                              &r_,
           const double                                    &T_,
           const double                                    &theta_,
           const double                                    &delta_t_,
           const std::function<double(const Point<dim> &)> &mu_,
           const std::function<double(const Point<dim> &, const double &)> &f_)
  : mesh_file_name(mesh_file_name_)
  , r(r_)
  , T(T_)
  , theta(theta_)
  , delta_t(delta_t_)
  , mu(mu_)
  , f(f_)
  , mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
  , mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
  , mesh(MPI_COMM_WORLD)
  , pcout(std::cout, mpi_rank == 0)
  , computing_timer(MPI_COMM_WORLD, pcout, TimerOutput::summary, TimerOutput::wall_times)
{}

// Reads the meh from file fopr the iniziaitazion of the domain
void Heat::init_mesh() {
  TimerOutput::Scope s(computing_timer, "init_mesh");
  pcout << "Initializing the mesh from file" << std::endl;
  Triangulation<dim> mesh_serial;
  GridIn<dim> grid_in;
  grid_in.attach_triangulation(mesh_serial);
  std::ifstream mesh_file(mesh_file_name);
  if (!mesh_file.is_open()) {
    throw std::runtime_error("Could not open mesh file: " + mesh_file_name);
  }
  grid_in.read_msh(mesh_file);

  GridTools::partition_triangulation(mpi_size, mesh_serial);
  const auto construction_data = TriangulationDescription::Utilities::
  create_description_from_triangulation(mesh_serial, MPI_COMM_WORLD);
  mesh.create_triangulation(construction_data);
}

void Heat::setup_system() {
  TimerOutput::Scope s(computing_timer, "setup_system");

  if (!fe) {
    fe = std::make_unique<FE_SimplexP<dim>>(r);
    quadrature = std::make_unique<QGaussSimplex<dim>>(r + 1);
    dof_handler.reinit(mesh); 
  }

  dof_handler.distribute_dofs(*fe);

  const IndexSet locally_owned_dofs = dof_handler.locally_owned_dofs();
  const IndexSet locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);

  // Ricostruzione dello sparsity pattern
  TrilinosWrappers::SparsityPattern sparsity(locally_owned_dofs, MPI_COMM_WORLD);
  DoFTools::make_sparsity_pattern(dof_handler, sparsity);
  sparsity.compress();

  // Re-inizializzazione di matrice e vettori
  mass_matrix.reinit(sparsity);
  stiffness_matrix.reinit(sparsity);
  system_matrix.reinit(sparsity);
  system_rhs.reinit(locally_owned_dofs, MPI_COMM_WORLD);
  solution_owned.reinit(locally_owned_dofs, MPI_COMM_WORLD);
  solution_owned_old.reinit(locally_owned_dofs, MPI_COMM_WORLD);
  solution.reinit(locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);
}

// Computes the mass and stifness matrices to be re-used
void Heat::assemble_matrices() {
  TimerOutput::Scope s(computing_timer, "assemble_matrices");
  
  // Azzera le matrici globali
  mass_matrix = 0.0;
  stiffness_matrix = 0.0;

  const unsigned int dofs_per_cell = fe->dofs_per_cell;
  const unsigned int n_q = quadrature->size();

  FEValues<dim> fe_values(*fe, *quadrature,
                          update_values | update_gradients | update_quadrature_points | update_JxW_values);

  FullMatrix<double> cell_mass(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> cell_stiffness(dofs_per_cell, dofs_per_cell);
  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

  for (const auto &cell : dof_handler.active_cell_iterators()) {
    if (!cell->is_locally_owned()) continue;

    fe_values.reinit(cell);
    cell_mass = 0.0;
    cell_stiffness = 0.0;

    for (unsigned int q = 0; q < n_q; ++q) {
      const double mu_loc = mu(fe_values.quadrature_point(q));

      for (unsigned int i = 0; i < dofs_per_cell; ++i) {
        for (unsigned int j = 0; j < dofs_per_cell; ++j) {
          // Matrice di Massa M
          cell_mass(i, j) += fe_values.shape_value(i, q) * fe_values.shape_value(j, q) * fe_values.JxW(q);

          // Matrice di Rigidezza A (Diffusione)
          cell_stiffness(i, j) += mu_loc * scalar_product(fe_values.shape_grad(i, q), 
                                                 fe_values.shape_grad(j, q)) * fe_values.JxW(q);
        }
      }
    }
    
    cell->get_dof_indices(dof_indices);
    mass_matrix.add(dof_indices, cell_mass);
    stiffness_matrix.add(dof_indices, cell_stiffness);
  }

  mass_matrix.compress(VectorOperation::add);
  stiffness_matrix.compress(VectorOperation::add);
}

// System assembled from the computed matrices and assembling the rhs
void Heat::assemble_system_and_rhs() {
  TimerOutput::Scope s(computing_timer, "assemble_system_and_rhs");

  // 1. Costruiamo la matrice di sistema: S = (1/dt) * M + theta * A
  // Copiamo M in S per efficienza, la scaliamo e aggiungiamo A
  system_matrix.copy_from(mass_matrix);
  system_matrix *= (1.0 / delta_t);
  system_matrix.add(theta, stiffness_matrix);

  // 2. Calcoliamo i contributi della soluzione vecchia al RHS tramite prodotti Matrice-Vettore
  // RHS = ( (1/dt)*M - (1 - theta)*A ) * u_old
  system_rhs = 0.0;
  TrilinosWrappers::MPI::Vector tmp(solution_owned); // Vettore temporaneo con stesso layout

  // + (1/dt) * M * u_old
  mass_matrix.vmult(tmp, solution_owned_old);
  system_rhs.add(1.0 / delta_t, tmp);

  // - (1 - theta) * A * u_old
  stiffness_matrix.vmult(tmp, solution_owned_old);
  system_rhs.add(-(1.0 - theta), tmp);

  // 3. Calcoliamo la forzante f (richiede un loop sulle celle, ma solo sui valori, non sui gradienti)
  const unsigned int dofs_per_cell = fe->dofs_per_cell;
  const unsigned int n_q = quadrature->size();
  FEValues<dim> fe_values(*fe, *quadrature, update_values | update_quadrature_points | update_JxW_values);

  Vector<double> cell_rhs(dofs_per_cell);
  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

  for (const auto &cell : dof_handler.active_cell_iterators()) {
    if (!cell->is_locally_owned()) continue;

    fe_values.reinit(cell);
    cell_rhs = 0.0;

    for (unsigned int q = 0; q < n_q; ++q) {
      const double f_old_loc = f(fe_values.quadrature_point(q), time);
      const double f_new_loc = f(fe_values.quadrature_point(q), time + delta_t);

      for (unsigned int i = 0; i < dofs_per_cell; ++i) {
        cell_rhs(i) += (theta * f_new_loc + (1.0 - theta) * f_old_loc) * fe_values.shape_value(i, q) * fe_values.JxW(q);
      }
    }
    cell->get_dof_indices(dof_indices);
    system_rhs.add(dof_indices, cell_rhs);
  }
  
  system_rhs.compress(VectorOperation::add);
}

// Solving the system with the AMG precoditioner
void Heat::solve_linear_system()
{
  TimerOutput::Scope s(computing_timer, "solve_linear_system");
  
  // Usiamo il precondizionatore Algebraic Multigrid (AMG) di Trilinos
  TrilinosWrappers::PreconditionAMG preconditioner;
  
  // Configuriamo i parametri per un problema di tipo parabolico/ellittico
  TrilinosWrappers::PreconditionAMG::AdditionalData amg_data;
  amg_data.elliptic = true;
  amg_data.smoother_sweeps = 2;
  amg_data.smoother_type = "ML symmetric Gauss-Seidel";
  
  preconditioner.initialize(system_matrix, amg_data);

  ReductionControl solver_control(/* maxiter = */ 10000,
                                  /* tolerance = */ 1.0e-12,
                                  /* reduce = */ 1.0e-6);

  // CG solver for SPD matrices
  SolverCG<TrilinosWrappers::MPI::Vector> solver(solver_control);

  solver.solve(system_matrix, solution_owned, system_rhs, preconditioner);
  
  pcout << "  -> " << solver_control.last_step() << " CG iterations (AMG Preconditioned)" << std::endl;
}

// output allega i file in uscita con intervalli di tempo regolari
void Heat::output() const
{
  TimerOutput::Scope s(computing_timer, "output");
  DataOut<dim> data_out;

  data_out.add_data_vector(dof_handler, solution, "solution");

  // Add vector for parallel partition.
  std::vector<unsigned int> partition_int(mesh.n_active_cells());
  GridTools::get_subdomain_association(mesh, partition_int);
  const Vector<double> partitioning(partition_int.begin(), partition_int.end());
  data_out.add_data_vector(partitioning, "partitioning");

  data_out.build_patches();

  const std::filesystem::path mesh_path(mesh_file_name);
  const std::string output_file_name = "output-" + mesh_path.stem().string();

  data_out.write_vtu_with_pvtu_record(/* folder = */ "./",
                                      /* basename = */ output_file_name,
                                      /* index = */ timestep_number,
                                      MPI_COMM_WORLD);
}

// Function run with fixed delta_t time stepping
void Heat::run() {
  TimerOutput::Scope s(computing_timer, "run");

  init_mesh();
  setup_system();
  assemble_matrices();
  VectorTools::interpolate(dof_handler, FunctionU0(), solution_owned);
  solution = solution_owned;
  solution_owned_old = solution_owned;

  const double output_interval = 0.05;
  
  // Output a t=0 iniziale
  pcout << "  --> Salvataggio output a t = " << time << std::endl;
  output();
  
  double next_output_time = output_interval;       

  // Standard fixed time-stepping loop
  while (time < T - 0.5 * delta_t) {
      
      time += delta_t;
      timestep_number++;
      
      assemble_system_and_rhs();
      solve_linear_system();

      // Aggiorniamo le variabili
      solution = solution_owned;
      solution_owned_old = solution_owned;
      
      // Controllo del throttling dell'output
      if (time >= next_output_time || time >= T - 0.5 * delta_t) {
        pcout << "  --> Salvataggio output a t = " << time << std::endl;
        output();
        
        while (next_output_time <= time && next_output_time < T) {
          next_output_time += output_interval;
        }
      }
  }
}