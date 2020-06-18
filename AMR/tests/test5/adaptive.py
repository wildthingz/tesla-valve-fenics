#!/usr/bin/env python

# pylint: disable-all

from fenics import *

# Use compiler optimizations
parameters["form_compiler"]["cpp_optimize"] = True

# Allow approximating values for points that may be generated outside
# of domain (because of numerical inaccuracies)
parameters["allow_extrapolation"] = True
parameters["refinement_algorithm"] = "plaza_with_parent_facets"

cpp_code = """
#include <pybind11/pybind11.h>
#include <dolfin/adaptivity/adapt.h>
#include <dolfin/fem/NonlinearVariationalProblem.h>
#include <dolfin/mesh/Mesh.h>
namespace py = pybind11;
PYBIND11_MODULE(SIGNATURE, m)
{
m.def("adapt", [](const dolfin::NonlinearVariationalProblem& problem,
		  std::shared_ptr<const dolfin::Mesh> adapted_mesh)
          {
	         return dolfin::adapt(problem, adapted_mesh);
          }
     );
}
"""
m = compile_cpp_code(cpp_code)

def adaptProblem(problem, mesh):
    return m.adapt(problem, mesh)

def readMesh(path):
    mesh = Mesh()
    with XDMFFile(path + "domain.xdmf") as infile:
        infile.read(mesh)
    mvc = MeshValueCollection("size_t", mesh, 2)
    with XDMFFile(path + "boundaries.xdmf") as infile:
        infile.read(mvc, "boundaries")
    boundaries = cpp.mesh.MeshFunctionSizet(mesh, mvc)
    return (mesh, boundaries)


def r(w, w0, wt, nu, dt, f, theta, stab=False):
    u, p = split(w)
    (v, q) = (as_vector((wt[0], wt[1])), wt[2])
    u0, p0 = split(w0)

    # Define coefficients
    k = dt
    um = theta*u + (1.0-theta)*u0

    F = (1/k)*inner(u - u0, v)*dx + inner(grad(um)*um, v)*dx + \
        nu*inner(grad(um), grad(v))*dx + inner(grad(p), v) * \
        dx - inner(f, v)*dx + div(um)*q*dx

    if stab:
        # add stabilization here
        F_stab = 0
        F += F_stab

    return F


def ufl(mesh, boundaries, f, dt, nu, theta):
    Ve = VectorElement("CG", mesh.ufl_cell(), 2)
    Qe = FiniteElement("CG", mesh.ufl_cell(), 1)
    W = FunctionSpace(mesh, Ve * Qe)

    # Define test functions
    w = Function(W)
    wt = TestFunction(W)
    w0 = Function(W)

    F = r(w, w0, wt, nu, dt, f, theta)

    J = derivative(F, w)

    return (F, J, W, w, w0, wt)


def M(mesh, boundaries, w, nu):
    u, p = split(w)

    ds_cylinder = Measure(
        'ds', domain=mesh, subdomain_data=boundaries,  subdomain_id=4)

    n = FacetNormal(mesh)
    F = p * n - nu * dot(grad(u), n)

    M1 = F[0] * ds_cylinder
    M2 = u[0] * dx
    return M2


def boundaryCondition(mesh, boundaries, W):
    v_in = Expression(
        ("4 * 0.3 * (x[1] * (0.41 - x[1])) / (0.41 * 0.41)", "0."), degree=2)
    p_out = Constant(0.0)
    zero_vec = Constant((0, 0))

    bcp_inflow = DirichletBC(W.sub(0), v_in, boundaries, 1)
    bcp_outflow = DirichletBC(W.sub(1), p_out, boundaries, 2)
    top_bottom_boundaries = DirichletBC(W.sub(0), zero_vec, boundaries, 3)
    cylinder_boundaries = DirichletBC(W.sub(0), zero_vec, boundaries, 4)

    bcs = [bcp_inflow, bcp_outflow, top_bottom_boundaries, cylinder_boundaries]

    return bcs


def force(mesh, boundaries, w, nu):
    # Initialize mesh function for boundary domains
    # R = w.leaf_node().function_space()
    ds_cylinder = Measure(
        'ds', domain=mesh, subdomain_data=boundaries,  subdomain_id=4)

    # Calculate force
    n = FacetNormal(mesh)

    u = w.split()[0]
    p = w.split()[1]

    F = nu * dot(grad(u), n) - p * n

    D = -F[0] * ds_cylinder
    L = -F[1] * ds_cylinder

    drag_force = assemble(D)
    lift_force = assemble(L)

    Umean = 0.2
    L = 0.1
    drag_coeff = drag_force * 2 / (L * Umean**2)
    lift_coeff = lift_force * 2 / (L * Umean**2)

    return (drag_force, lift_force, drag_coeff, lift_coeff)


def main():

    meshPath = "../mesh/"

    # physical parameters
    f = Constant((0, 0))
    mu = 0.001
    rho = 1
    nu = mu / rho

    # control parameters
    dt = 0.01
    T = 1
    theta = 0.5  # Crank-Nicholson

    # adaptive parameters
    adaptive = True
    adapt_ratio = 0.03
    tol = 1e-3
    ref = 0
    maxiters = 1

    # output file initialization
    fileU = File("velocity.pvd")
    filePhi = File("phi.pvd")
    fileMesh = File("mesh.pvd")

    # initial mesh
    (mesh, boundaries) = readMesh(meshPath)

    
    (F, J, W, w, w0, wt) = ufl(mesh, boundaries, f, dt, nu, theta)
    bcs = boundaryCondition(mesh, boundaries, W)
    problem = NonlinearVariationalProblem(F, w, bcs, J)

    t = 0
    while t <= T:
        for i in range(maxiters):
            solver = NonlinearVariationalSolver(problem)
            solver.solve()

            fileU << w

            output = assemble(M(mesh, boundaries, w, nu))
            print("adaptive iter: {0:3d} num_vertices: {1:5d} output: {2:2.5f} err: {3:2.5f}".format
                  (i, mesh.num_vertices(), output, abs(ref - output)))

            if abs(ref - output) < tol:
                print("convergence reached!")
                break

            ref = output

            phi = Function(W)
            a_dual = adjoint(J)  # Construct the adjoint problem
            L_dual = M(mesh, boundaries, wt, nu)
            adjoint_problem = LinearVariationalProblem(
                a_dual, L_dual, phi, bcs)
            adjoint_solver = LinearVariationalSolver(adjoint_problem)
            adjoint_solver.solve()

            # Plot all quantities (see implementation above)
            filePhi << phi

            ZE = FiniteElement("DG", mesh.ufl_cell(), 0)
            Z = FunctionSpace(mesh, ZE)
            z = TestFunction(Z)

            # Construct error indicators
            Lei = r(w, w0, z*phi, nu, dt, f, theta)
            ei = assemble(Lei)

            gamma = abs(ei.get_local())

            tmp = Vector(MPI.comm_world, Z.dim())
            tmp = ei.gather_on_zero()
            tmp = abs(tmp)
            gammaGlobal = MPI.comm_world.bcast(tmp)
            # Mark cells for refinement
            cell_markers = MeshFunction(
                "bool", mesh, mesh.topology().dim(), MPI.rank(mesh.mpi_comm()))
            gamma_0 = sorted(gammaGlobal, reverse=True)[
                int(len(gammaGlobal)*adapt_ratio) - 1]

            for c in cells(mesh):
                cell_markers[c] = gamma[c.index()] > gamma_0

            if adaptive:  # Refine mesh
                rmesh = refine(mesh, cell_markers, False)
                # meshdata = mesh.data()
                # print(meshdata.array("parent_cell", 2))
                problem = adaptProblem(problem, rmesh)
                mesh = rmesh  # Shift to next adaptive iteration
            else:
                break

        w0.assign(w)
        t += dt


if __name__ == "__main__":
    main()
