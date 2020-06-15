#include <dolfin.h>
#include "NonlinearNS.h"
#include "Functionals.h"

using namespace dolfin;

struct MeshOut {
    std::shared_ptr<dolfin::Mesh> mesh;
    std::shared_ptr<dolfin::MeshFunction<std::size_t>> boundaries;
};

struct LiftDrag {
    double drag_force;
    double lift_force;
    double drag_coeff;
    double lift_coeff;
};

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

MeshOut get_mesh() {
    auto mesh = std::make_shared<Mesh>();
    XDMFFile initialmeshfile(MPI_COMM_WORLD, "../../mesh/domain.xdmf");
    initialmeshfile.read(*mesh);

    MeshValueCollection<std::size_t> mf_mvc(mesh, 2);
    XDMFFile initialmffile(MPI_COMM_WORLD, "../../mesh/boundaries.xdmf");
    initialmffile.read(mf_mvc, "boundaries");
    auto boundaries = std::make_shared<MeshFunction<std::size_t>>(mesh, mf_mvc);

    return {.mesh = mesh, .boundaries = boundaries};
};

LiftDrag lift_drag(MeshOut meshout, std::shared_ptr<dolfin::Function> w, std::shared_ptr<dolfin::Constant> nu) {
    
    auto mesh = meshout.mesh;
    auto boundaries = meshout.boundaries;
    
    Functionals::Form_lift_force lift_force(mesh, w, nu);
    Functionals::Form_drag_force drag_force(mesh, w, nu);

    lift_force.ds = boundaries;
    drag_force.ds = boundaries;



    // Assemble functionals over sub domain
    double lift = assemble(lift_force);
    double drag = assemble(drag_force);

    double Umean = 0.2;
    double L = 0.1;
    double drag_coeff = drag * 2 / (L * Umean*Umean);
    double lift_coeff = lift * 2 / (L * Umean*Umean);

    char liftdrag_buffer[50];

    sprintf(liftdrag_buffer, "DRAG FORCE: %f", liftdrag.drag_force);
    info(liftdrag_buffer);
    sprintf(liftdrag_buffer, "LIFT FORCE: %f", liftdrag.lift_force);
    info(liftdrag_buffer);
    sprintf(liftdrag_buffer, "DRAG COEFF: %f", liftdrag.drag_coeff);
    info(liftdrag_buffer);
    sprintf(liftdrag_buffer, "LIFT COEFF: %f", liftdrag.lift_coeff);
    info(liftdrag_buffer);
    
    return {drag, lift, drag_coeff, lift_coeff};
}

int main()
{
    // Print log messages only from the root process in parallel
    parameters["std_out_all_processes"] = false;

    // Set parameter values
    double dt = 0.01;
    double T = 5;
    double mu = 0.001;
    double rho = 1;
    // Crank-Nicholson
    double theta = 0.5;

    // Load mesh from file
    // auto mesh = std::make_shared<Mesh>("../mesh/cylinder.xml");
    // MeshFunction<std::size_t> boundaries(mesh, "../mesh/cylinder_facet_region.xml");
    auto meshout = get_mesh();
    auto mesh = meshout.mesh;
    auto boundaries = meshout.boundaries;

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
    auto w0 = std::make_shared<Function>(W);
    auto w = std::make_shared<Function>(W);

    // Create coefficients
    auto k = std::make_shared<Constant>(dt);
    auto f = std::make_shared<Constant>(0, 0);
    auto nu = std::make_shared<Constant>(mu / rho);
    auto theta_c = std::make_shared<Constant>(theta);

    // Set coefficients
    std::map<std::string, std::shared_ptr<const GenericFunction>> coefficients =
        {{"w", w}, {"w0", w0}, {"k", k}, {"nu", nu}, {"theta", theta_c}};
    J->set_coefficients(coefficients);
    coefficients.insert({"f", f});
    F->set_coefficients(coefficients);

    File file("velocity.pvd");

    // Time-stepping
    double t = 0;
    int count = 0;
    int timesteps_saved = 1;
    int save_every = ((T / dt) / timesteps_saved) == 0 ? 1 : (T / dt) / timesteps_saved;
    auto problem = std::make_shared<NonlinearVariationalProblem>(F, w, bcs, J);
    auto solver = NonlinearVariationalSolver(problem);

    // snes solver -- comment out to use newton solver
    solver.parameters["nonlinear_solver"] = "snes";
    solver.parameters("snes_solver")["line_search"] = "bt";
    solver.parameters("snes_solver")["linear_solver"] = "mumps";
    solver.parameters("snes_solver")["preconditioner"] = "hypre_amg";
    solver.parameters("snes_solver")["report"] = true;

    char time_stdout_buffer[50];

    while (t < T + DOLFIN_EPS)
    {
        *w0->vector() = *w->vector();

        solver.solve();

        // Save to file
        if (count % save_every == 0)
        {
            file << (*w0)[0];
        }

        // std::cout << "t: " << t << std::endl;
        sprintf(time_stdout_buffer, "CURRENT TIME: %f", t);
        info(time_stdout_buffer);

        // Move to next time step
        t += dt;
        count++;
    }

    auto liftdrag = lift_drag(meshout, w, nu);
    return 0;
}