# Adaptive Space-Time Finite Element Solver for the Heat Equation

This repository contains a high-performance, parallel C++ implementation of a space-time adaptive solver for a linear parabolic problem (the heat equation). The solver is built on top of the deal.II finite element library and utilizes MPI for distributed memory parallelism alongside the Trilinos framework for advanced algebraic preconditioning.

## Overview

The core objective of this project is to model the evolution of a localized thermal impulse over time and space. Standard numerical methods using uniform grids and fixed time steps are highly inefficient for such transient, localized phenomena, as they waste computational resources in regions where nothing is happening.

To overcome this, we implemented a fully adaptive formulation based on *a posteriori* error estimation techniques. The solver dynamically adjusts both the spatial grid resolution (*h*-adaptivity) and the temporal integration step (*k*-adaptivity) in real-time, concentrating computational effort strictly where the physical action occurs.

## Key Features

* **Space-Time Adaptivity:**
    * **Spatial (*h*-adaptivity):** Utilizes the `KellyErrorEstimator` to flag regions with high error gradients for localized mesh refinement, coarsening areas with low activity. The physical state is safely mapped across changing grids using the `SolutionTransfer` module.
    * **Temporal (*k*-adaptivity):** Dynamically modulates the integration step size $\Delta t$ based on the mathematically exact $L^2$-norm of the solution variance, aggressively skipping time steps during quiescent periods.
* **High Performance & Parallelism:**
    * Fully distributed using `parallel::fullydistributed::Triangulation` and MPI.
    * Explicit pre-computation of Mass and Stiffness matrices, reducing each time step to rapid matrix-vector multiplications.
    * Linear systems are solved using a Conjugate Gradient (CG) algorithm accelerated by a robust Algebraic Multigrid (AMG) preconditioner via Trilinos.
* **Strict Baseline Comparison:** The repository includes a purely non-adaptive twin solver (sharing the exact same AMG preconditioners and pre-assembly optimizations) to allow for mathematically rigorous "apples-to-apples" benchmarking.

## Mathematical Formulation

The solver addresses the heat equation defined on a 3D spatial domain over a time interval $(0, T)$:

$$\frac{\partial u}{\partial t} - \nabla \cdot (\mu \nabla u) = f \quad \text{in } \Omega \times (0, T)$$

With homogeneous Neumann boundary conditions and a zero initial state. The forcing term $f(x,t)$ is designed as a sequence of impulses localized in both space and time: $f(x,t) = g(t)h(x)$.

The temporal domain is integrated using the unconditionally stable Backward Euler scheme ($\theta = 1.0$), while the spatial domain is discretized using $P_2$ continuous finite elements.

## Experimental Results

Preliminary local benchmarks on an 8-core machine using various unstructured starting grids demonstrate massive computational savings. The synergistic action of $h$ and $k$ adaptivity bypasses hundreds of unnecessary numerical integration steps and drastically downsizes the algebraic systems solved at each step.

| Initial Grid | Non-Adaptive Baseline | Adaptive Solver | Speedup Factor |
| :--- | :--- | :--- | :--- |
| `mesh-cube-10` | 28.1 s | 1.62 s | ~17.3x |
| `mesh-cube-20` | 565.0 s | 19.6 s | ~28.8x |
| `mesh-cube-40` | 1670.0 s | 90.3 s | ~18.5x |

*Note: To guarantee scientific reproducibility, formal scalability benchmarks are conducted via an automated CI/CD pipeline targeting a dedicated Microsoft Azure Virtual Machine.*

## Dependencies

* [deal.II](https://www.dealii.org/) (Finite Element framework)
* [Trilinos](https://trilinos.github.io/) (Distributed linear algebra and AMG preconditioning)
* MPI (Message Passing Interface)
* CMake (Build system)

## Theoretical Background

The adaptive mechanics and error estimation formulations are largely inspired by the theoretical framework presented in:
*M. Picasso, "Adaptive finite elements for a linear parabolic problem," Comput. Methods Appl. Mech. Engrg., 1998.*
