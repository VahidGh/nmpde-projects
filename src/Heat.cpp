#include "Heat.hpp"

void
Heat::setup()
{
  TimerOutput::Scope s(computing_timer, "setup");
  pcout << "===============================================" << std::endl;

  // Create the mesh.
  {
    pcout << "Initializing the mesh" << std::endl;

    // Read serial mesh.
    Triangulation<dim> mesh_serial;

    {
      GridIn<dim> grid_in;
      grid_in.attach_triangulation(mesh_serial);

      std::ifstream mesh_file(mesh_file_name);
      grid_in.read_msh(mesh_file);
    }

    // Copy the serial mesh into the parallel one.
    {
      GridTools::partition_triangulation(mpi_size, mesh_serial);

      const auto construction_data = TriangulationDescription::Utilities::
        create_description_from_triangulation(mesh_serial, MPI_COMM_WORLD);
      mesh.create_triangulation(construction_data);
    }

    pcout << "  Number of elements = " << mesh.n_global_active_cells()
          << std::endl;
  }

  pcout << "-----------------------------------------------" << std::endl;

  // Initialize the finite element space.
  {
    pcout << "Initializing the finite element space" << std::endl;

    fe = std::make_unique<FE_SimplexP<dim>>(r);

    pcout << "  Degree                     = " << fe->degree << std::endl;
    pcout << "  DoFs per cell              = " << fe->dofs_per_cell
          << std::endl;

    quadrature = std::make_unique<QGaussSimplex<dim>>(r + 1);

    pcout << "  Quadrature points per cell = " << quadrature->size()
          << std::endl;
  }

  pcout << "-----------------------------------------------" << std::endl;

  // Initialize the DoF handler.
  {
    pcout << "Initializing the DoF handler" << std::endl;

    dof_handler.reinit(mesh);
    dof_handler.distribute_dofs(*fe);

    pcout << "  Number of DoFs = " << dof_handler.n_dofs() << std::endl;
  }

  pcout << "-----------------------------------------------" << std::endl;

  // Initialize the linear system.
  {
    pcout << "Initializing the linear system" << std::endl;

    const IndexSet locally_owned_dofs = dof_handler.locally_owned_dofs();
    const IndexSet locally_relevant_dofs =
      DoFTools::extract_locally_relevant_dofs(dof_handler);

    pcout << "  Initializing the sparsity pattern" << std::endl;
    TrilinosWrappers::SparsityPattern sparsity(locally_owned_dofs,
                                               MPI_COMM_WORLD);
    DoFTools::make_sparsity_pattern(dof_handler, sparsity);
    sparsity.compress();

    pcout << "  Initializing the system matrices" << std::endl;
    system_matrix.reinit(sparsity);
    mass_matrix.reinit(sparsity);
    stiffness_matrix.reinit(sparsity);

    pcout << "  Initializing vectors" << std::endl;
    system_rhs.reinit(locally_owned_dofs, MPI_COMM_WORLD);
    solution_owned.reinit(locally_owned_dofs, MPI_COMM_WORLD);
    solution_owned_old.reinit(locally_owned_dofs, MPI_COMM_WORLD);
    solution.reinit(locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);
  }
}

// Computes the mass and stiffness matrices to be re-used
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

  // 3. Calcoliamo la forzante f (richiede un loop sulle celle, ma solo sui valori)
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


void
Heat::solve_linear_system()
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

  SolverCG<TrilinosWrappers::MPI::Vector> solver(solver_control);

  solver.solve(system_matrix, solution_owned, system_rhs, preconditioner);
}

void
Heat::output() const
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

void
Heat::run()
{
  TimerOutput::Scope s(computing_timer, "run");
  // Setup initial conditions.
  {
    setup();

    // Calcoliamo M e A statiche prima di iniziare il ciclo temporale
    assemble_matrices();

    VectorTools::interpolate(dof_handler, FunctionU0(), solution_owned);
    solution = solution_owned;
    solution_owned_old = solution_owned; // Inizializza il vettore history

    time            = 0.0;
    timestep_number = 0;

    // Output initial condition.
    output();
  }

  pcout << "===============================================" << std::endl;

  const double output_interval = 0.05; // Frequenza di salvataggio I/O
  double next_output_time = output_interval;

  // Time-stepping loop (Fixed delta_t)
  while (time < T - 0.5 * delta_t)
    {
      time += delta_t;
      ++timestep_number;

      assemble_system_and_rhs();
      solve_linear_system();

      // Perform parallel communication to update the ghost values
      solution = solution_owned;
      solution_owned_old = solution_owned; // Salva per il prossimo step

      // Throttling dell'output su disco
      if (time >= next_output_time || time >= T) {
        pcout << "  --> Salvataggio output a t = " << time << std::endl;
        output();
        
        while (next_output_time <= time && next_output_time < T) {
          next_output_time += output_interval;
        }
      }
    }
}