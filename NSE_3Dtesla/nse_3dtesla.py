"""## 3D tesla"""

#!/usr/bin/env python
from fenics import *
import numpy as np
import time

eps = 1e-5

def readMesh(path):
    mesh = Mesh()
    with XDMFFile(path + "domain.xdmf") as infile:
        infile.read(mesh)
    mvc = MeshValueCollection("size_t", mesh, 2)
    with XDMFFile(path + "boundaries.xdmf") as infile:
        infile.read(mvc, "boundaries")
    boundaries = cpp.mesh.MeshFunctionSizet(mesh, mvc)
    return (mesh, boundaries)

def boundaryCondition(mesh, boundaries, W):
    v_in = Expression(("4 * 0.3 * (x[1] * (0.5 - x[1])) * (x[2] * (0.5 - x[2])) / (0.5 * 0.5)", "0.", "0."), degree = 2)
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
    (v, q) = TestFunctions(W)
    w0 = Function(W)
    u0, _ = split(w0)

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
    h = CellDiameter(mesh)
    u_norm = sqrt(dot(w.split()[0], w.split()[0]))
    tau = ((2.0/k)**2 + (2.0*u_norm/h)**2 + (4.0*nu/h**2)**2)**(-0.5)
    F += tau*(inner(grad(p) + grad(um)*um, grad(q) + grad(um)*v) + inner(div(um), div(v)))*dx

    J = derivative(F, w)

    return (F, J, W, w, w0)

def main():
    meshPath = "/content/drive/My Drive/NSE/NS_3Dtesla/mesh/"

    dt = 0.01
    f = Constant((0, 0, 0))
    mu = 0.01
    rho = 1
    nu = mu/rho
    theta = 0.5  # Crank-Nicholson
    T = 5.0

    (mesh, boundaries) = readMesh(meshPath)
    (F, J, W, w, w0) = ufl(mesh, f, dt, nu, theta)
    bcs = boundaryCondition(mesh, boundaries, W)

    problem = NonlinearVariationalProblem(F, w, bcs, J)
    solver = NonlinearVariationalSolver(problem)

    prm = solver.parameters

    #Create solver type SNES
    prm['nonlinear_solver'] = 'snes'
    prm['snes_solver']['line_search'] = 'bt'
    prm['snes_solver']['linear_solver']= 'lu'
    prm['snes_solver']['maximum_iterations'] = 500
    prm['snes_solver']['absolute_tolerance'] = 1E-8
    prm['snes_solver']['relative_tolerance'] = 1E-7
    prm['snes_solver']['krylov_solver']['nonzero_initial_guess'] = True

    file = File("/content/drive/My Drive/NSE/NS_3Dtesla/output/velocity.pvd")

    t = 0
    while t <= T:
        w0.assign(w)
        solver.solve()
        file << w0.split()[0]
        info("Current time: {0:1.3f}".format(t))
        t += dt
  
    Re = 0.2*0.1/(mu/rho)

    print('Reynolds number: ', Re)

if __name__ == "__main__":

    main()
