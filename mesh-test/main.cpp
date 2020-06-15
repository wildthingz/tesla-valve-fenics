#include <dolfin.h>
#include <mshr.h>

using namespace dolfin;

int main(int argc, char **argv)
{
    auto geom = std::make_shared<mshr::Surface3D>("../test.stl");
    
    std::shared_ptr<dolfin::Mesh> mesh; 
    mesh = mshr::generate_mesh(geom, 100);

    // auto meshfile = std::make_shared<MeshFunction<std::size_t>>
    //     (mesh, mesh->topology().dim());
    MeshFunction<std::size_t>
        meshfile(mesh, mesh->topology().dim());

    // Output cell distribution to VTK file
    File file("mesh.pvd");
    file << meshfile;
}