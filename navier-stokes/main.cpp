#include <dolfin.h>
#include "TentativeVelocity.h"
#include "PressureUpdate.h"
#include "VelocityUpdate.h"

using namespace dolfin;

int main()
{
    // Print log messages only from the root process in parallel
    parameters["std_out_all_processes"] = false;

    // Set parameter values
    double dt = 0.01;
    double T = 1;
    double mu = 25;
    double rho = 1;

    // Load mesh from file
    // auto mesh = std::make_shared<Mesh>("../mesh/cylinder.xml");
    // MeshFunction<std::size_t> boundaries(mesh, "../mesh/cylinder_facet_region.xml");

    auto mesh = std::make_shared<Mesh>();
    XDMFFile initialmeshfile(MPI_COMM_WORLD, "../mesh/mesh.xdmf");
    initialmeshfile.read(*mesh);

    MeshValueCollection<std::size_t> mf_mvc(mesh, 2);
    XDMFFile initialmffile(MPI_COMM_WORLD, "../mesh/mf.xdmf");
    initialmffile.read(mf_mvc, "name_to_read");
    auto boundaries = std::make_shared<MeshFunction<std::size_t>>(mesh, mf_mvc);

    auto V = std::make_shared<VelocityUpdate::FunctionSpace>(mesh);
    auto Q = std::make_shared<PressureUpdate::FunctionSpace>(mesh);

    // Create boundary conditions
    auto p_in = std::make_shared<Constant>(8.0);
    auto p_out = std::make_shared<Constant>(0.0);
    auto zero_vec = std::make_shared<Constant>(0.0, 0.0);

    DirichletBC inflow(Q, p_in, boundaries, 1);
    DirichletBC outflow(Q, p_out, boundaries, 2);

    DirichletBC noslip(V, zero_vec, boundaries, 3);
    DirichletBC cylinder(V, zero_vec, boundaries, 4);

    std::vector<DirichletBC *> bcu = {&cylinder, &noslip};
    std::vector<DirichletBC *> bcp = {{&inflow, &outflow}};

    // Create functions
    auto u_ = std::make_shared<Function>(V);
    auto u_n = std::make_shared<Function>(V);
    auto p_ = std::make_shared<Function>(Q);
    auto p_n = std::make_shared<Function>(Q);

    // Create coefficients
    auto k = std::make_shared<Constant>(dt);
    auto f = std::make_shared<Constant>(0, 0);
    auto mu_c = std::make_shared<Constant>(mu);
    auto rho_c = std::make_shared<Constant>(rho);

    // Create forms
    TentativeVelocity::BilinearForm a1(V, V);
    TentativeVelocity::LinearForm L1(V);
    PressureUpdate::BilinearForm a2(Q, Q);
    PressureUpdate::LinearForm L2(Q);
    VelocityUpdate::BilinearForm a3(V, V);
    VelocityUpdate::LinearForm L3(V);

    // Set coefficients
    a1.k = k;
    a1.mu = mu_c;
    a1.rho = rho_c;
    L1.k = k;
    L1.u_n = u_n;
    L1.p_n = p_n;
    L1.mu = mu_c;
    L1.f = f;
    L1.rho = rho_c;
    L2.k = k;
    L2.p_n = p_n;
    L2.u_ = u_;
    L3.k = k;
    L3.u_ = u_;
    L3.p_n = p_n;
    L3.p_ = p_;

    // Assemble matrices
    Matrix A1, A2, A3;
    assemble(A1, a1);
    assemble(A2, a2);
    assemble(A3, a3);

    // Create vectors
    Vector b1, b2, b3;

    // Use amg preconditioner if available
    const std::string prec(has_krylov_solver_preconditioner("amg") ? "amg" : "default");

    // Create files for storing solution
    File ufile("../results/velocity.pvd");

    // Time-stepping
    double t = dt;
    while (t < T + DOLFIN_EPS)
    {
        // Compute tentative velocity step
        assemble(b1, L1);
        for (std::size_t i = 0; i < bcu.size(); i++)
            bcu[i]->apply(A1, b1);
        solve(A1, *u_->vector(), b1, "gmres", "default");

        // Pressure correction
        assemble(b2, L2);
        for (std::size_t i = 0; i < bcp.size(); i++)
        {
            bcp[i]->apply(A2, b2);
            bcp[i]->apply(*p_->vector());
        }
        solve(A2, *p_->vector(), b2, "bicgstab", prec);

        // Velocity correction
        assemble(b3, L3);
        for (std::size_t i = 0; i < bcu.size(); i++)
            bcu[i]->apply(A3, b3);
        solve(A3, *u_->vector(), b3, "gmres", "default");

        // Move to next time step
        *u_n = *u_;
        t += dt;

        cout << "t: " << t << endl;
    }

    // Save to file
    ufile << *u_n;
    return 0;
}