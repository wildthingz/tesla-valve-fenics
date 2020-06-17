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

// Subdomain on which goal function should be defined
class Outflow : public SubDomain
{
  bool inside(const Array<double>& x, bool on_boundary) const
  {
    return x[0] > 2.2 - DOLFIN_EPS;
  }
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
        values[0] = 4 * 0.3 * (x[1] * (0.41 - x[1])) / (0.41 * 0.41);
        values[1] = 0;
    }
};

MeshOut get_mesh() {
    auto mesh = std::make_shared<Mesh>();
    XDMFFile initialmeshfile(MPI_COMM_WORLD, "../mesh/domain.xdmf");
    initialmeshfile.read(*mesh);

    MeshValueCollection<std::size_t> mf_mvc(mesh, 2);
    XDMFFile initialmffile(MPI_COMM_WORLD, "../mesh/boundaries.xdmf");
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

    sprintf(liftdrag_buffer, "DRAG FORCE: %f", drag);
    info(liftdrag_buffer);
    sprintf(liftdrag_buffer, "LIFT FORCE: %f", lift);
    info(liftdrag_buffer);
    sprintf(liftdrag_buffer, "DRAG COEFF: %f", drag_coeff);
    info(liftdrag_buffer);
    sprintf(liftdrag_buffer, "LIFT COEFF: %f", lift_coeff);
    info(liftdrag_buffer);
    
    return {drag, lift, drag_coeff, lift_coeff};
};

int main()
{
    // Print log messages only from the root process in parallel
    parameters["std_out_all_processes"] = false;
    parameters["allow_extrapolation"] = true;
    parameters["refinement_algorithm"] = "plaza_with_parent_facets";

    // Set parameter values
    double dt = 0.005;
    double T = 1;
    double mu = 0.001;
    double rho = 1;
    // Crank-Nicholson
    double theta = 0.5;
    // Define error tolerance
    double tol = 1.e-3;

    // Load mesh from file
    auto meshout = get_mesh();
    auto mesh = meshout.mesh;
    auto boundaries = meshout.boundaries;

    auto W = std::make_shared<NonlinearNS::BilinearForm::TrialSpace>(mesh);

    // Create boundary conditions
    auto v_in = std::make_shared<InflowVelocity>();
    auto p_out = std::make_shared<Constant>(0.0);
    auto zero_vec = std::make_shared<Constant>(0.0, 0.0);

    auto inflow = std::make_shared<const DirichletBC>(W->sub(0), v_in, boundaries, 1);
    auto outflow = std::make_shared<const DirichletBC>(W->sub(1), p_out, boundaries, 2);

    auto noslip = std::make_shared<const DirichletBC>(W->sub(0), zero_vec, boundaries, 3);
    auto cylinder = std::make_shared<const DirichletBC>(W->sub(0), zero_vec, boundaries, 4);

    std::vector<std::shared_ptr<const DirichletBC>> bcs {cylinder, noslip, inflow, outflow};

    // Create forms
    auto F = std::make_shared<NonlinearNS::LinearForm>(W);
    auto J = std::make_shared<NonlinearNS::BilinearForm>(W, W);

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

    // Define goal functional
    auto M = std::make_shared<NonlinearNS::GoalFunctional>(mesh);
    M->w = w;
    M->nu = nu;
    // Outflow outflow_m;
    // auto outflow_markers = std::make_shared<MeshFunction<std::size_t>>(mesh, mesh->topology().dim()-1, 1);
    // outflow_m.mark(*outflow_markers, 0);
    M->ds = boundaries;

    File file("velocity.pvd");
    File meshfile("mesh.pvd");
    // Time-stepping
    double t = 0;
    int count = 0;
    int timesteps_saved = 100;
    int save_every = ((T / dt) / timesteps_saved) == 0 ? 1 : (T / dt) / timesteps_saved;
    auto problem = std::make_shared<NonlinearVariationalProblem>(F, w, bcs, J);
    AdaptiveNonlinearVariationalSolver solver(problem, M);

    // // snes solver -- comment out to use newton solver
    // solver.parameters["nonlinear_solver"] = "snes";
    // solver.parameters("snes_solver")["line_search"] = "bt";
    // solver.parameters("snes_solver")["linear_solver"] = "mumps";
    // solver.parameters("snes_solver")["preconditioner"] = "hypre_amg";
    // solver.parameters("snes_solver")["report"] = true;
    char time_stdout_buffer[50];

    while (t < T + DOLFIN_EPS)
    {
        // (w0->leaf_node_shared_ptr())->vector() = (w->leaf_node_shared_ptr())->vector();
        // *w0->vector() = *w->vector();
        (w0->leaf_node()).vector() = (w->leaf_node()).vector();
        solver.solve(tol);
        
        // Save to file
        if (count % save_every == 0)
        {
            file << (*w0)[0];
            meshfile << (w0->leaf_node())[1];
        }
        // (*w).leaf_node() = (*w0).leaf_node();
        // std::cout << "t: " << t << std::endl;
        sprintf(time_stdout_buffer, "CURRENT TIME: %f", t);
        info(time_stdout_buffer);

        // Move to next time step
        t += dt;
        count++;
    }
    solver.summary();
    // auto liftdrag = lift_drag(meshout, w, nu);
    return 0;
}