#include <dolfin.h>
#include <mshr.h>
#include <mpi.h>

using namespace dolfin;

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    auto geom = std::make_shared<mshr::Surface3D>("../test.stl");
    
    std::shared_ptr<dolfin::Mesh> mesh; 
    mesh = mshr::generate_mesh(geom, 20);

    MeshFunction<std::size_t>
        processes0(mesh, mesh->topology().dim());

    // Output cell distribution to VTK file
    File file("processes.pvd");
    file << processes0;
}