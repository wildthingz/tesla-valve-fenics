#include <dolfin.h>
#include "NonlinearNS.h"

using namespace dolfin;

// inlet
class InflowVelocity : public Expression
{
public:
    // Constructor
    InflowVelocity() : Expression(2) {}

    // Evaluate velocity at inflow
    void eval(Array<double> &values, const Array<double> &x) const
    {
        values[0] = 4 * 1.5 * (x[1] * (0.41 - x[1])) / (0.41 * 0.41);
        values[1] = 0;
    }
};

int main()
{
    // Print log messages only from the root process in parallel
    parameters["std_out_all_processes"] = false;

    // Set parameter values
    double dt = 0.05;
    double T = 5;
    double mu = 0.001;
    double rho = 1;
    // Crank-Nicholson
    double theta = 0.5;

    // Load mesh from file
    // auto mesh = std::make_shared<Mesh>("../mesh/cylinder.xml");
    // MeshFunction<std::size_t> boundaries(mesh, "../mesh/cylinder_facet_region.xml");

    auto mesh = std::make_shared<Mesh>();
    XDMFFile initialmeshfile(MPI_COMM_WORLD, "../../mesh/domain.xdmf");
    initialmeshfile.read(*mesh);

    MeshValueCollection<std::size_t> mf_mvc(mesh, 2);
    XDMFFile initialmffile(MPI_COMM_WORLD, "../../mesh/boundaries.xdmf");
    initialmffile.read(mf_mvc, "boundaries");
    auto boundaries = std::make_shared<MeshFunction<std::size_t>>(mesh, mf_mvc);

    auto W = std::make_shared<NonlinearNS::FunctionSpace>(mesh);

    // Create boundary conditions
    auto v_in = std::make_shared<InflowVelocity>();
    auto p_out = std::make_shared<Constant>(0.0);
    auto zero_vec = std::make_shared<Constant>(0.0, 0.0);

    auto inflow = std::make_shared<const DirichletBC>(W->sub(0), v_in, boundaries, 1);
    auto outflow = std::make_shared<const DirichletBC>(W->sub(1), p_out, boundaries, 2);

    auto noslip = std::make_shared<const DirichletBC>(W->sub(0), zero_vec, boundaries, 3);
    auto cylinder = std::make_shared<const DirichletBC>(W->sub(0), zero_vec, boundaries, 4);

    auto bcs = {cylinder, noslip, inflow, outflow};

    // Create forms
    auto F = std::make_shared<NonlinearNS::ResidualForm>(W);
    auto J = std::make_shared<NonlinearNS::JacobianForm>(W, W);

    // Create functions
    auto w_ = std::make_shared<Function>(W);
    auto w = std::make_shared<Function>(W);

    // Create coefficients
    auto k = std::make_shared<Constant>(dt);
    auto f = std::make_shared<Constant>(0, 0);
    auto nu = std::make_shared<Constant>(mu / rho);
    auto theta_c = std::make_shared<Constant>(theta);

    // Set coefficients
    std::map<std::string, std::shared_ptr<const GenericFunction>> coefficients = 
        {{"w", w}, {"w_", w_}, {"k", k}, {"nu", nu}, {"theta", theta_c}};
    J->set_coefficients(coefficients);
    coefficients.insert({"f", f});
    F->set_coefficients(coefficients);

    File file("velocity.pvd");

    // Time-stepping
    double t = 0;
    int count = 0;
    int timesteps_saved = 100;
    int save_every = (T / dt) / timesteps_saved ? 1 : (T / dt) / timesteps_saved;

    auto problem = std::make_shared<NonlinearVariationalProblem>(F, w, bcs, J);
    auto solver = NonlinearVariationalSolver(problem);

    // snes solver -- comment out to use newton solver
    solver.parameters["nonlinear_solver"] = "snes";
    solver.parameters("snes_solver")["line_search"] = "basic";
    solver.parameters("snes_solver")["linear_solver"] = "lu";

    while (t < T + DOLFIN_EPS)
    {
        *w_->vector() = *w->vector();

        solver.solve();

        // Save to file
        if (count % save_every == 0)
        {
            file << (*w_)[0];
        }

        std::cout << "t: " << t << std::endl;

        // Move to next time step
        t += dt;
        count++;
    }
    return 0;
}