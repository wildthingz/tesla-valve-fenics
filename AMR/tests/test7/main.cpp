#include <dolfin.h>
#include "AdaptiveNS.h"
#include "Adjoint.h"
#include "Functionals.h"
#include "ErrorIndicators.h"

using namespace dolfin;

struct MeshOut
{
    std::shared_ptr<dolfin::Mesh> mesh;
    std::shared_ptr<dolfin::MeshFunction<std::size_t>> boundaries;
};

struct LiftDrag
{
    double drag_force;
    double lift_force;
    double drag_coeff;
    double lift_coeff;
};

struct OptimizerFunctions
{
    std::shared_ptr<Functionals::Form_velocity_field> velocity_field;
    std::shared_ptr<Functionals::Form_drag_force> drag_force;
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

MeshOut get_mesh()
{
    auto mesh = std::make_shared<Mesh>();
    XDMFFile initialmeshfile(MPI_COMM_WORLD, "../mesh/domain.xdmf");
    initialmeshfile.read(*mesh);

    auto mf_mvc = std::make_shared<MeshValueCollection<std::size_t>>(mesh, 2);
    XDMFFile initialmffile(MPI_COMM_WORLD, "../mesh/boundaries.xdmf");
    initialmffile.read(*mf_mvc, "boundaries");
    auto boundaries = std::make_shared<MeshFunction<std::size_t>>(mesh, *mf_mvc);

    return {.mesh = mesh, .boundaries = boundaries};
};

LiftDrag lift_drag(MeshOut &meshout, std::shared_ptr<dolfin::Function> w, std::shared_ptr<dolfin::Constant> nu)
{

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
    double drag_coeff = drag * 2 / (L * Umean * Umean);
    double lift_coeff = lift * 2 / (L * Umean * Umean);

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
}

OptimizerFunctions opt_func(MeshOut &meshout, std::shared_ptr<dolfin::Function> w, std::shared_ptr<dolfin::Constant> nu)
{
    auto mesh = meshout.mesh;
    auto boundaries = meshout.boundaries;

    auto velocity_field = std::make_shared<Functionals::Form_velocity_field>(mesh, w);
    auto drag_force = std::make_shared<Functionals::Form_drag_force>(mesh, w, nu);

    drag_force->ds = boundaries;

    return {.velocity_field = velocity_field, .drag_force = drag_force};
}

int main()
{
    // Print log messages only from the root process in parallel
    parameters["std_out_all_processes"] = false;
    // Allow approximating values for points that may be generated outside
    // of domain (because of numerical inaccuracies)
    parameters["allow_extrapolation"] = true;
    parameters["refinement_algorithm"] = "plaza_with_parent_facets";

    // Set parameter values
    double dt = 0.005;
    double T = 1;
    double mu = 0.001;
    double rho = 1;
    // Crank-Nicholson
    double theta = 0.5;

    // Adaptive parameters
    bool adaptive = true;
    double adapt_ratio = 0.03;
    double tol = 5e-4;
    double maxiters = 10;
    double ref = 0;

    // Load mesh from file
    auto meshout = get_mesh();
    auto mesh = meshout.mesh;
    auto boundaries = meshout.boundaries;

    auto W = std::make_shared<AdaptiveNS::FunctionSpace>(mesh);

    // Create boundary conditions
    auto v_in = std::make_shared<InflowVelocity>();
    auto p_out = std::make_shared<Constant>(0.0);
    auto zero_vec = std::make_shared<Constant>(0.0, 0.0);

    auto inflow = std::make_shared<const DirichletBC>(W->sub(0), v_in, boundaries, 1);
    auto outflow = std::make_shared<const DirichletBC>(W->sub(1), p_out, boundaries, 2);

    auto noslip = std::make_shared<const DirichletBC>(W->sub(0), zero_vec, boundaries, 3);
    auto cylinder = std::make_shared<const DirichletBC>(W->sub(0), zero_vec, boundaries, 4);

    std::vector<std::shared_ptr<const DirichletBC>> bcs {cylinder, noslip, inflow, outflow};

    // Create functions
    auto w0 = std::make_shared<Function>(W);
    auto w = std::make_shared<Function>(W);
    auto phi = std::make_shared<Function>(W);

    // Create coefficients
    auto f = std::make_shared<Constant>(0, 0);
    auto k = std::make_shared<Constant>(dt);
    auto nu = std::make_shared<Constant>(mu / rho);
    auto theta_c = std::make_shared<Constant>(theta);

    // Create forms
    auto F = std::make_shared<AdaptiveNS::ResidualForm>(W);
    auto J = std::make_shared<AdaptiveNS::JacobianForm>(W, W);

    auto a_dual = std::make_shared<Adjoint::BilinearForm>(W, W);
    auto L_dual = std::make_shared<Adjoint::LinearForm>(W);

    // Set coefficients
    std::map<std::string, std::shared_ptr<const GenericFunction>> coefficients =
        {{"w", w}, {"w0", w0}, {"k", k}, {"nu", nu}, {"theta", theta_c}};
    J->set_coefficients(coefficients);
    a_dual->set_coefficients(coefficients);
    coefficients.insert({"f", f});
    F->set_coefficients(coefficients);

    File fileU("velocity.pvd");
    File filePhi("phi.pvd");

    auto problem = std::make_shared<NonlinearVariationalProblem>(F, w, bcs, J);
    auto adjoint_problem = std::make_shared<LinearVariationalProblem>(a_dual, L_dual, phi, bcs);
    char stdout_buffer[50];

    // Time-stepping
    double t = 0;
    int count = 0;
    // int timesteps_saved = 100;
    // int save_every = ((T / dt) / timesteps_saved) == 0 ? 1 : (T / dt) / timesteps_saved;
    while (t < T + DOLFIN_EPS)
    {   
        // auto meshout = get_mesh();
        // auto mesh = meshout.mesh;
        // auto boundaries = meshout.boundaries;
        *w0->vector() = *w->vector();
        sprintf(stdout_buffer, "CURRENT TIME: %f", t);
        info(stdout_buffer);

        // if (t != 0) {
        //     MeshFunction<bool> cell_markers(mesh, mesh->topology().dim(), false);
        //     mesh = std::make_shared<Mesh>(refine(*mesh, cell_markers, false));
        //     problem = adapt(*problem, mesh);
        //     adjoint_problem = adapt(*adjoint_problem, mesh);
        //     w = adapt(*w, mesh);
        //     w0 = adapt(*w0, mesh);
        //     phi = adapt(*phi, mesh);
        //     meshout = {.mesh = mesh, .boundaries = boundaries};
        // }

        for (int i = 0; i < maxiters; i++)
        {
            auto solver = NonlinearVariationalSolver(problem);

            // snes solver -- comment out to use newton solver
            solver.parameters["nonlinear_solver"] = "snes";
            solver.parameters("snes_solver")["line_search"] = "bt";
            solver.parameters("snes_solver")["linear_solver"] = "mumps";
            solver.parameters("snes_solver")["preconditioner"] = "hypre_amg";
            solver.parameters("snes_solver")["report"] = true;

            solver.solve();

            // auto M = opt_func(meshout, w, nu);
            // double mval = assemble(*M.velocity_field);

            
            // if (abs(ref - mval) < tol)
            // {
            //     info("CONVERGENCE REACHED!");
            //     info("--------------------------------------");
            //     break;
            // }

            // ref = mval;

            auto adjoint_solver = LinearVariationalSolver(adjoint_problem);
            adjoint_solver.solve();

            filePhi << *phi;

            auto Z = std::make_shared<ErrorIndicators::FunctionSpace>(mesh);
            auto Lei = std::make_shared<ErrorIndicators::LinearForm>(Z);
            // std::map<std::string, std::shared_ptr<const GenericFunction>> coefficients =
            //     {{"w", w}, {"w0", w0}, {"f", f}, {"nu", nu}, {"theta", theta_c}, {"phi", phi}};
            std::map<std::string, std::shared_ptr<const GenericFunction>> coefficients =
                {{"w", w}, {"w0", w0}, {"f", f}, {"k", k}, {"nu", nu}, {"phi", phi}, {"theta", theta_c}};
            Lei->set_coefficients(coefficients);

            auto ei = std::make_shared<Vector>(mesh->mpi_comm());
            assemble(*ei, *Lei);

            auto gamma_global = std::make_shared<std::vector<double>>();

            ei->gather_on_zero(*gamma_global);

            MPI::broadcast(MPI_COMM_WORLD, *gamma_global);

            for (unsigned int i = 0; i < (*gamma_global).size(); i++)
            {
                if ((*gamma_global)[i] < 0)
                    (*gamma_global)[i] *= -1;
            }

            std::sort((*gamma_global).begin(), (*gamma_global).end(), std::greater<double>());

            int index = (int)(*gamma_global).size() * adapt_ratio;
            double gamma_0 = (*gamma_global)[index - 1];


            if (abs(ref - gamma_0) < tol)
            {
                sprintf(stdout_buffer, "GAMMA_0 VALUE: %f, ERORR: %f", gamma_0, abs(ref - gamma_0));
                info(stdout_buffer);
                info("MESH CONVERGENCE REACHED!");
                info("--------------------------------------");
                break;
            }

            ref = gamma_0;

            auto gamma = std::make_shared<std::vector<double>>(mesh->topology().dim());
            ei->get_local(*gamma);
            for (unsigned int i = 0; i < (*gamma).size(); i++)
            {
                if ((*gamma)[i] < 0)
                    (*gamma)[i] = abs((*gamma)[i]);
            }
        
            MeshFunction<bool> cell_markers(mesh, mesh->topology().dim(), false);

            for (CellIterator c(*mesh); !c.end(); ++c)
            {
                cell_markers[*c] = ((*gamma)[c->index()] > gamma_0);
            }

            if (adaptive)
            {
                // sprintf(stdout_buffer, "GAMMA_0: %f", gamma_0);
                // info(stdout_buffer);

                mesh = std::make_shared<Mesh>(refine(*mesh, cell_markers, false));
                boundaries = adapt(*boundaries, mesh);
                problem = adapt(*problem, mesh);
                adjoint_problem = adapt(*adjoint_problem, mesh);
                w = adapt(*w, mesh);
                w0 = adapt(*w0, mesh);
                phi = adapt(*phi, mesh);
                meshout = {.mesh = mesh, .boundaries = boundaries};
            }
            else
            {
                break;
            }

            info("--------------------------------------");
        }

        fileU << (*w0)[0];
        t += dt;
        count++;
        
    }
    auto liftdrag = lift_drag(meshout, w, nu);
    return 0;
}