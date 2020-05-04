#!/usr/bin/env python
from fenics import *
import numpy as np
import time

def readMesh(path):
    mesh = Mesh()
    with XDMFFile(path + "domain.xdmf") as infile:
        infile.read(mesh)
    mvc = MeshValueCollection("size_t", mesh, 2)
    with XDMFFile(path + "boundaries.xdmf") as infile:
        infile.read(mvc, "boundaries")
    boundaries = cpp.mesh.MeshFunctionSizet(mesh, mvc)
    return (mesh, boundaries)

def ufl(mesh, f, dt, nu, theta):
    Ve = VectorElement("CG", mesh.ufl_cell(), 2)
    Qe = FiniteElement("CG", mesh.ufl_cell(), 1)
    W = FunctionSpace(mesh, Ve * Qe)

    # Define test functions
    w = Function(W)
    u, p = split(w)
    (v, q) = TestFunctions(W)
    w0 = Function(W)
    u0, _ = split(w0)

    # Define coefficients
    k = dt 
    um = theta*u + (1.0-theta)*u0

    F = (1/k)*inner(u - u0, v)*dx + inner(grad(um)*um, v)*dx + \
        nu*inner(grad(um), grad(v))*dx - inner(grad(p), v) * \
        dx - inner(f, v)*dx + div(um)*q*dx

    J = derivative(F, w)

    return (F, J, W, w, w0)

def boundaryCondition(mesh, boundaries, W):
    v_in = Expression(("4 * 1.5 * (x[1] * (0.41 - x[1])) / (0.41 * 0.41)", "0."), degree=2) 
    p_out = Constant(0.0)
    zero_vec = Constant((0, 0))

    bcp_inflow = DirichletBC(W.sub(0), v_in, boundaries, 1)
    bcp_outflow = DirichletBC(W.sub(1), p_out, boundaries, 2)
    top_bottom_boundaries = DirichletBC(W.sub(0), zero_vec, boundaries, 3)
    cylinder_boundaries = DirichletBC(W.sub(0), zero_vec, boundaries, 4)

    bcs = [bcp_inflow, bcp_outflow, top_bottom_boundaries, cylinder_boundaries] 
    
    return bcs

def main():
    meshPath = "../mesh/"
    
    dt = 0.01
    f = Constant((0,0))
    mu = 0.001
    rho = 1
    theta = 0.5 # Crank-Nicholson
    T = 5

    (mesh, boundaries) = readMesh(meshPath)
    (F, J, W, w, w0) = ufl(mesh, f, dt, mu/rho, theta)
    bcs = boundaryCondition(mesh, boundaries, W)

    problem = NonlinearVariationalProblem(F, w, bcs, J)
    solver = NonlinearVariationalSolver(problem)

    file = File("velocity.pvd")

    t = 0
    while t <= T:
        w0.assign(w)
        solver.solve()
        file << w0.split()[0]
        info("Current time: {0:1.3f}".format(t))
        t += dt

if __name__ == "__main__":

    main()