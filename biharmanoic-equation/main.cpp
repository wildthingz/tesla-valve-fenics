#include <dolfin.h>
#include "biharmonic.h"

using namespace dolfin;

// Source term
class Source : public Expression
{
public:
    void eval(Array<double> &values, const Array<double> &x) const
    {
        values[0] = 4.0 * std::pow(DOLFIN_PI, 4) *
                    std::sin(DOLFIN_PI * x[0]) * std::sin(DOLFIN_PI * x[1]);
    }
};

// Sub domain for Dirichlet boundary condition
class DirichletBoundary : public SubDomain
{
    bool inside(const Array<double> &x, bool on_boundary) const
    {
        return on_boundary;
    }
};

int main()
{
    // Make mesh ghosted for evaluation of DG terms
    parameters["ghost_mode"] = "shared_facet";

    // Create mesh
    auto mesh = std::make_shared<UnitSquareMesh>(32, 32);

    // Create functions
    auto f = std::make_shared<Source>();
    auto alpha = std::make_shared<Constant>(8.0);

    // Create function space
    auto V = std::make_shared<biharmonic::FunctionSpace>(mesh);

    // Define boundary condition
    auto u0 = std::make_shared<Constant>(0.0);
    auto boundary = std::make_shared<DirichletBoundary>();
    DirichletBC bc(V, u0, boundary);

    // Define variational problem
    biharmonic::BilinearForm a(V, V);
    biharmonic::LinearForm L(V);
    a.alpha = alpha;
    L.f = f;

    // Compute solution
    Function u(V);
    solve(a == L, u, bc);

    // Save solution in VTK format
    File file("biharmonic.pvd");
    file << u;

    return 0;
}
