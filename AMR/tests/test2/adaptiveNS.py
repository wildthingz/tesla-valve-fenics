#!/usr/bin/env python
from fenics import *
import numpy as np
import time
import matplotlib.pyplot as plt

eps = 1e-5

# Use compiler optimizations
parameters["form_compiler"]["cpp_optimize"] = True

# Allow approximating values for points that may be generated outside
# of domain (because of numerical inaccuracies)
parameters["allow_extrapolation"] = True
parameters["refinement_algorithm"] = "plaza_with_parent_facets"

def readMesh(path):
    mesh = Mesh()
    with XDMFFile(path + "domain.xdmf") as infile:
        infile.read(mesh)
    mvc = MeshValueCollection("size_t", mesh, 2)
    with XDMFFile(path + "boundaries.xdmf") as infile:
        infile.read(mvc, "boundaries")
    boundaries = cpp.mesh.MeshFunctionSizet(mesh, mvc)
    return (mesh, boundaries)


def ufl(mesh, boundaries, f, dt, nu, theta):
    Ve = VectorElement("CG", mesh.ufl_cell(), 2)
    Qe = FiniteElement("CG", mesh.ufl_cell(), 1)
    W = FunctionSpace(mesh, Ve * Qe)

    # Define test functions
    w = Function(W)
    u, p = split(w)
    (v, q) = TestFunctions(W)
    w0 = Function(W)
    u0, p0 = split(w0)

    # Define coefficients
    k = dt
    um = theta*u + (1.0-theta)*u0

    F = (1/k)*inner(u - u0, v)*dx + inner(grad(um)*um, v)*dx + \
        nu*inner(grad(um), grad(v))*dx + inner(grad(p), v) * \
        dx - inner(f, v)*dx + div(um)*q*dx

    J = derivative(F, w)

    ds_cylinder = Measure(
        'ds', domain=mesh, subdomain_data=boundaries,  subdomain_id=4)

    n = FacetNormal(mesh)
    F1 = nu * dot(grad(u), n) - p * n
    M = -F1[0]*ds_cylinder

    # M=u[0]*dx

    return (F, J, M, W, w, w0)


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


def force(mesh, boundaries, w, w0, nu):
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

    dt = 0.01
    f = Constant((0, 0))
    mu = 0.001
    rho = 1
    tol = 1e-3
    theta = 0.5  # Crank-Nicholson
    T = 1

    (mesh, boundaries) = readMesh(meshPath)
    (F, J, M, W, w, w0) = ufl(mesh, boundaries, f, dt, mu/rho, theta)
    bcs = boundaryCondition(mesh, boundaries, W)

    problem = NonlinearVariationalProblem(F, w, bcs, J)
    solver = AdaptiveNonlinearVariationalSolver(problem, M)

    file = File("velocity.pvd")
    meshfile = File("mesh.pvd")

    t = 0
    while t <= T:
    
        w0.leaf_node().vector()[:] = w.leaf_node().vector()
        # w0 = w.copy()
        solver.solve(tol)
        file << w.leaf_node().split()[0]
        meshfile << w.leaf_node()
        info("Current time: {0:1.3f}".format(t))
        t += dt

    drag_force_t = force(mesh, boundaries, w, w0, mu/rho)[0]
    lift_force_t = force(mesh, boundaries, w, w0, mu/rho)[1]
    drag_coeff_t = force(mesh, boundaries, w, w0, mu/rho)[2]
    lift_coeff_t = force(mesh, boundaries, w, w0, mu/rho)[3]

    info("drag force: {0:2.5f}\nlift force: {1:2.5f}\ndrag coeff: {2:2.5f}\nlift coeff: {3:2.5f}".format(
        drag_force_t, lift_force_t, drag_coeff_t, lift_coeff_t))

    # Show solver summary
    solver.summary()

    # Show all timings
    list_timings(TimingClear.clear, [TimingType.wall])

if __name__ == "__main__":

    main()
