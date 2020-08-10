#!/usr/bin/env python
from fenics import *
import numpy as np
import time

eps = 1e-5

def readMesh(path):
    mesh = Mesh()
    with XDMFFile(path + "domain.xdmf") as infile:
        infile.read(mesh)

    mvc_boundaries = MeshValueCollection("size_t", mesh, 3)
    with XDMFFile(path + "boundaries.xdmf") as infile:
        infile.read(mvc_boundaries, "boundaries")
    boundaries = cpp.mesh.MeshFunctionSizet(mesh, mvc_boundaries)

    return (mesh, boundaries)

def boundaryCondition(mesh, boundaries, W):
    #v_in = Expression(("4 * 0.3 * (x[1] * (0.41 - x[1])) / (0.41 * 0.41)", "0."), degree=2)
    v_in = Expression(("0.3 * (1 - 4 * x[2]*x[2] - 4 * x[1] * x[1] )", "0.", "0."), degree = 2)
    # v_in = Expression(("4.", "0.", "0."), degree = 2)
    p_in = Constant(3)
    p_out = Constant(0.0)
    zero_vec = Constant((0., 0., 0.))

    bcp_inflow = DirichletBC(W.sub(0), v_in, boundaries, 1)
    bcp_outflow = DirichletBC(W.sub(1), p_out, boundaries, 2)
    noslip_boundaries = DirichletBC(W.sub(0), zero_vec, boundaries, 3)

    bcs = [bcp_inflow, bcp_outflow, noslip_boundaries]

    return bcs

def ufl(mesh, f, dt, nu, theta):
    Ve = VectorElement("CG", mesh.ufl_cell(), 2)
    Qe = FiniteElement("CG", mesh.ufl_cell(), 1)
    W = FunctionSpace(mesh, Ve * Qe)

    # Define test functions
    w = Function(W)
    u, p = split(w)
    # (u, p) = (as_vector((w[0], w[1], w[2])), w[3])
    (v, q) = TestFunctions(W)
    w0 = Function(W)
    u0, _ = split(w0)
    # (u0, p0) = (as_vector((w0[0], w0[1], w0[2])), w0[3])

    # Define coefficients
    k = dt
    um = theta*u + (1.0-theta)*u0

    F = (1/k)*inner(u - u0, v)*dx + inner(grad(um)*um, v)*dx \
        + nu*inner(grad(um), grad(v))*dx + inner(grad(p), v)*dx + div(um)*q*dx\
        - inner(f, v)*dx

    #Alternative formulations    
    #F = (1/k)*inner(u - u0, v)*dx + inner(grad(um)*um, v)*dx \
    #    + nu*inner(grad(um), grad(v))*dx - inner(div(v),p)*dx + inner(div(u),q)*dx - inner(f, v)*dx
    
    #F = (1/k)*inner(u - u0, v)*dx\
    #    + theta*nu*inner(grad(u), grad(v))*dx + (1-theta)*nu*inner(grad(u), grad(v))*dx\
    #    + theta*inner(grad(u)*u, v)*dx + (1-theta)*inner(grad(u0)*u0, v)*dx\
    #    - inner(div(v),p)*dx - inner(div(u),q)*dx - inner(f, v)*dx

    #F = (1/k)*inner(u - u0, v)*dx + inner(grad(um)*um, v)*dx \
    #    +2*nu*inner(epsilon(um),epsilon(v))*dx - inner(p,div(v))*dx + div(um)*q*dx\
    #    - inner(f, v)*dx
        
    # stabilization parameter
    # h = CellDiameter(mesh)
    # u_norm = sqrt(dot(u, u))
    # tau = ((2.0/k)**2 + (2.0*u_norm/h)**2 + (4.0*nu/h**2)**2)**(-0.5)
    # F += tau*(inner(grad(p) + grad(u)*u - f, grad(q) + grad(v)*u) + inner(div(u), div(v)))*dx


    J = derivative(F, w)

    return (F, J, W, w, w0)

def main():

    # parameters["form_compiler"]["quadrature_degree"] = 10
    parameters["form_compiler"]["cpp_optimize"] = True

    # meshPath = "/content/drive/My Drive/NSE/NS_3Dtesla/mesh/"
    meshPath = "../mesh/old-mesh/"

    dt = 0.01
    f = Constant((0, 0, 0))
    mu = 0.01
    rho = 1
    nu = mu/rho
    theta = 0.5  # Crank-Nicholson
    T = 5.0

    (mesh, boundaries)= readMesh(meshPath)
    (F, J, W, w, w0) = ufl(mesh, f, dt, nu, theta)
    bcs = boundaryCondition(mesh, boundaries, W)

    problem = NonlinearVariationalProblem(F, w, bcs, J)
    solver = NonlinearVariationalSolver(problem)

    prm = solver.parameters

    prm["nonlinear_solver"] = "snes"
    prm["snes_solver"]["line_search"] = "bt"
    prm["snes_solver"]["linear_solver"] = "mumps"
    prm['snes_solver']['maximum_iterations'] = 500
    prm['snes_solver']['absolute_tolerance'] = 1E-5
    prm['snes_solver']['relative_tolerance'] = 1E-5
    prm["snes_solver"]["preconditioner"] = "hypre_amg"
    prm['snes_solver']['krylov_solver']['nonzero_initial_guess'] = True
    prm["snes_solver"]["report"] = True

    # file = File("/content/drive/My Drive/NSE/NS_3Dtesla/output/velocity.pvd")
    file = File("velocity.pvd")
    pfile = File("pressure.pvd")

    t = 0
    while t <= T:
        w0.assign(w)
        solver.solve()
        file << w0.split()[0]
        pfile << w0.split()[1]
        print(t)
        t += dt
  
    Re = 0.2*0.1/(mu/rho)

    print('Reynolds number: ', Re)

if __name__ == "__main__":

    main()
