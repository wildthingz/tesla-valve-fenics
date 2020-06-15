#include <dolfin.h>

using namespace dolfin;

int main()
{
  // Create mesh
  auto mesh = std::make_shared<UnitSquareMesh>(20, 20);

  // Create MeshFunction to hold cell process rank
  MeshFunction<std::size_t>
    processes0(mesh, mesh->topology().dim(), dolfin::MPI::rank(mesh->mpi_comm()));

  // Output cell distribution to VTK file
  File file("processes.pvd");
  file << processes0;

  // Mark all cells on process 0 for refinement
  const MeshFunction<bool>
    marker(mesh, mesh->topology().dim(), dolfin::MPI::rank(mesh->mpi_comm()));

  // Refine mesh, but keep all new cells on parent process
  auto mesh0 = std::make_shared<Mesh>(refine(*mesh, marker, false));

  // Create MeshFunction to hold cell process rank
  const MeshFunction<std::size_t>
    processes1(mesh0, mesh->topology().dim(), dolfin::MPI::rank(mesh->mpi_comm()));
  file << processes1;

  // Refine mesh, but this time repartition the mesh after refinement
  auto mesh1 = std::make_shared<Mesh>(refine(*mesh, marker, true));

  // Create MeshFunction to hold cell process rank
  MeshFunction<std::size_t>
    processes2(mesh1, mesh->topology().dim(), dolfin::MPI::rank(mesh->mpi_comm()));
  file << processes2;

  return 0;
}