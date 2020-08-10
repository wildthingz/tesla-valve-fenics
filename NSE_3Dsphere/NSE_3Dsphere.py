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
    #v_in = Expression(("4 * 0.3 * (x[1] * (0.41 - x[1])) / (0.41 * 0.41)", "0."), degree=2)
    v_in = Expression(("4 * 0.3 * (x[1] * (0.41 - x[1])) * (x[2] * (0.41 - x[2])) / (0.41 * 0.41)", "0.", "0."), degree = 3)
    p_out = Constant(0.0)
    zero_vec = Constant((0., 0., 0.))

    bcp_inflow = DirichletBC(W.sub(0), v_in, boundaries, 4)
    bcp_outflow = DirichletBC(W.sub(1), p_out, boundaries, 6)
    noslip_boundaries = DirichletBC(W.sub(0), zero_vec, boundaries, 5)
    sphere_boundaries = DirichletBC(W.sub(0), zero_vec, boundaries, 7)

    bcs = [bcp_inflow, bcp_outflow, noslip_boundaries, sphere_boundaries]

    return bcs

def force(mesh, boundaries, w, nu):
    #Initialize mesh function for boundary domains
    ds_sphere = Measure('ds', domain = mesh, subdomain_data = boundaries, subdomain_id = 7)

    #Calculate force
    n = FacetNormal(mesh)

    u = w.split()[0]
    p = w.split()[1]

    F = nu*dot(grad(u), n) - p*n

    D = -F[0] * ds_sphere
    L = -F[1] * ds_sphere

    drag_force = assemble(D)
    lift_force = assemble(L)

    Umean = 0.2
    L = 0.1
    drag_coeff = drag_force * 2 / (L * Umean**2)
    lift_coeff = lift_force * 2 / (L * Umean**2)

    return (drag_force, lift_force, drag_coeff, lift_coeff)

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
    # meshPath = "/content/drive/My Drive/NSE/NS_3Dsphere/mesh/"
    meshPath = "./mesh/"
    dt = 0.1
    f = Constant((0, 0, 0))
    mu = 0.001
    rho = 1
    nu = mu/rho
    theta = 0.5  # Crank-Nicholson
    T = 1.0

    (mesh, boundaries) = readMesh(meshPath)
    (F, J, W, w, w0) = ufl(mesh, f, dt, nu, theta)
    bcs = boundaryCondition(mesh, boundaries, W)

    problem = NonlinearVariationalProblem(F, w, bcs, J)
    solver = NonlinearVariationalSolver(problem)

    # file = File("/content/drive/My Drive/NSE/NS_3Dsphere/output_sphere/velocity.pvd")
    file = File("./results/velocity.pvd")
    
    values = []
    t = 0
    while t <= T:
        w0.assign(w)
        solver.solve()
        drag_force_t = force(mesh, boundaries, w, nu)[0]
        lift_force_t = force(mesh, boundaries, w, nu)[1]
        drag_coeff_t = force(mesh, boundaries, w, nu)[2]
        lift_coeff_t = force(mesh, boundaries, w, nu)[3]
        values.append(np.array([t,drag_force_t,lift_force_t, drag_coeff_t, lift_coeff_t]))
        file << w0.split()[0]
        info("Current time: {0:1.3f}".format(t))
        t += dt
    
    # np.savetxt("/content/drive/My Drive/NSE/NS_3Dsphere/output_sphere/drag_lift.txt", values, delimiter=",")
    np.savetxt("./results/drag_lift.txt", values, delimiter=",")

    Re = 0.2*0.1/(mu/rho)

    print('Reynolds number: ', Re)

    print ('drag force: ', force(mesh, boundaries, w, nu)[0])
    print ('lift force: ', force(mesh, boundaries, w, nu)[1])
    print ('drag coeff: ', force(mesh, boundaries, w, nu)[2])
    print ('lift coeff: ', force(mesh, boundaries, w, nu)[3])

if __name__ == "__main__":

    main()
