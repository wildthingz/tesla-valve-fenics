#!/usr/bin/env python
from fenics import *
from ufl import nabla_div
import numpy as np
import time
import csv

U_mean = Constant(0.2)
L = Constant(0.1)

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
    v_in = Expression(("0.3 * 4 * (x[1] * (0.41 - x[1])) / (0.41 * 0.41)", "0."), degree = 2)
    p_out = Constant(0.0)
    zero_vec = Constant((0, 0))

    bcp_inflow = DirichletBC(W.sub(0), v_in, boundaries, 1)
    bcp_outflow = DirichletBC(W.sub(1), p_out, boundaries, 2)
    top_bottom_boundaries = DirichletBC(W.sub(0), zero_vec, boundaries, 3)
    cylinder_boundaries = DirichletBC(W.sub(0), zero_vec, boundaries, 4)

    bcs = [bcp_inflow, bcp_outflow, top_bottom_boundaries, cylinder_boundaries] 
    
    return bcs

class Obstacle(SubDomain):
    def inside(self, x, on_boundary):
        return pow(x[0] - 0.2, 2) + pow(x[1] - 0.2, 2) <= pow(0.05, 2)

def sigma(w, nu, mesh):
    return -w.split()[1]*Identity(mesh.topology().dim()) + 2.0*nu*sym(grad(w.split()[0]))

def force(mesh, boundaries, w, nu):
    # Initialize mesh function for boundary domains
    obstacle = Obstacle()
    obstacle_boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
    obstacle_boundaries.set_all(7)
    obstacle.mark(obstacle_boundaries, 7)
    ds_cylinder = Measure('ds', domain=mesh, subdomain_data=obstacle_boundaries)
    
    #Calculate force
    n = FacetNormal(mesh)
    #I = Identity(mesh.geometry().dim())
    D = -dot(sigma(w, nu, mesh), n)[0]*ds_cylinder
    L = -dot(sigma(w, nu, mesh), n)[1]*ds_cylinder

    drag_force = assemble(D)
    lift_force = assemble(L)

    #Umean = 0.2 ; L = 0.1
    #drag_coeff = drag_force * 2 /(L*U_mean**2)
    #lift_coeff = lift_force * 2 /(L*U_mean**2)
    drag_coeff = drag_force * 2 /(0.1*0.2**2)
    lift_coeff = lift_force * 2 /(0.1*0.2**2)

    return (drag_force, lift_force, drag_coeff, lift_coeff)

"""## Navier Stokes solver with SNES"""

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

    #Define coefficients
    v_in = Expression(("0.3 * 4 * (x[1] * (0.41 - x[1])) / (0.41 * 0.41)", "0."), degree = 2)
    k = dt 
    um = theta*u + (1.0-theta)*u0
    
    #Define system
    F = (1/k)*inner(u - u0, v)*dx + inner(grad(um)*um, v)*dx \
        + nu*inner(grad(um), grad(v))*dx + inner(grad(p), v)*dx + div(um)*q*dx\
        - inner(f, v)*dx

    #r = gamma*(om*p*q + im*inner(u - uin, v) + nm*inner(u, v))*ds + # Weak boundary conditions

    #Residual of strong Navier-Stokes
    r = 1/k*(w.split()[0]- w0.split()[0]) + theta*grad(w.split()[0])*w.split()[0] \
        + (1.0-theta)*grad(w0.split()[0])*w0.split()[0] \
        - theta*div(sigma(w, nu, mesh)) - (1.0-theta)*div(sigma(w0, nu, mesh))
        
    # stabilization parameter
    h = CellDiameter(mesh)
    u_norm = sqrt(dot(w.split()[0], w.split()[0]))
    tau = ((2.0/k)**2 + (2.0*u_norm/h)**2 + (4.0*nu/h**2)**2)**(-0.5)
    #F += tau * inner(grad(u)*v, r)*dx

    #CG(1) stabilization
    #d = .2*h**(3./2.)
    d = 2.0*u_norm/h
    F += tau*(inner(grad(p) + grad(um)*um, grad(q) + grad(um)*v) + inner(div(um), div(v)))*dx

    #F += (h/(2.0*u_norm))*inner(grad(v)*u, r)*dx

    #tau = tau_supg = tau_pspg = 0.000001
    #tau = ((2.0/k)**2 + (2.0*u_norm/h)**2 + (4.0*nu/h**2)**2)**(-0.5)
    #tau_supg = ((2.0/k)**2 + (2.0*u_norm/h)**2 + (4.0*nu/h**2)**2)**(-0.5)
    #tau_pspg = h**2/(4*nu)
    
    #SUPG stabilization
    #F_supg = tau_supg*inner(grad(u)*v, r)*dx

    #PSPG stabilization
    #F_pspg = tau_pspg*inner(grad(q), r)*dx

    #LSIC stabilization
    #tau_lsic = -2/3 * nu
    #Flsic = tau_lsic*inner(div(v), div(u))*dx

    #F += Flsic

    J = derivative(F, w)

    return (F, J, W, w, w0)


def main():
    meshPath = "/content/drive/My Drive/NS_2Dcylinder/Benchmark test/mesh/"
    
    dt = 0.05
    f = Constant((0,0))
    mu = 0.001
    rho = 1
    theta = 0.5 # Crank-Nicholson
    T = 1.0

    (mesh, boundaries) = readMesh(meshPath)
    (F, J, W, w, w0) = ufl(mesh, f, dt, mu/rho, theta)
    bcs = boundaryCondition(mesh, boundaries, W)

    #iterative_solver = True

    problem = NonlinearVariationalProblem(F, w, bcs, J)
    solver = NonlinearVariationalSolver(problem)

    #Create solver type SNES
    prm = solver.parameters
    prm['nonlinear_solver'] = 'snes'
    prm['snes_solver']['line_search'] = 'bt'
    prm['snes_solver']['linear_solver']= 'lu'
    prm['snes_solver']['maximum_iterations'] = 500
    prm['snes_solver']['absolute_tolerance'] = 1E-8
    prm['snes_solver']['relative_tolerance'] = 1E-7
   
    file = File("/content/drive/My Drive/NS_2Dcylinder/Benchmark test/output/velocity.pvd")
    #drag_lift_file = File("/content/drive/My Drive/NS_2Dcylinder/Benchmark test/output/drag_lift.bin")
    values = []

    t = 0
    #u_norm = 0.01
    while t <= T:
        w0.assign(w)
        solver.solve()
        drag_force_t = force(mesh, boundaries, w, mu/rho)[0]
        lift_force_t = force(mesh, boundaries, w, mu/rho)[1]
        drag_coeff_t = force(mesh, boundaries, w, mu/rho)[2]
        lift_coeff_t = force(mesh, boundaries, w, mu/rho)[3]
        values.append(np.array([t,drag_force_t,lift_force_t, drag_coeff_t, lift_coeff_t]))

        file << w0.split()[0]
        info("Current time: {0:1.3f}".format(t))

        t += dt

    np.savetxt("/content/drive/My Drive/NS_2Dcylinder/Benchmark test/output/drag_lift.txt", values, delimiter=",")

    Re = 0.2*0.1/(mu/rho)

    print('Reynolds number: ', Re)

    print ('drag force: ', force(mesh, boundaries, w, mu/rho)[0])
    print ('lift force: ', force(mesh, boundaries, w, mu/rho)[1])
    print ('drag coeff: ', force(mesh, boundaries, w, mu/rho)[2])
    print ('lift coeff: ', force(mesh, boundaries, w, mu/rho)[3])

if __name__ == "__main__":

    main()
