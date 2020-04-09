#!/usr/bin/env python
from fenics import *
import numpy as np
import time

T = 1.0             # final time
num_steps = 100     # number of time steps
dt = T / num_steps  # time step size
mu = 25             # kinematic viscosity
rho = 1             # density

#####################################################################
#############LOAD MESH FILE##########################################
#####################################################################
# Export .msh mesh into dolfin .xml with command below
# dolfin-convert geometry-name.msh geometry-name.xml
mesh = Mesh('mesh/cylinder.xml')
markers = MeshFunction('size_t', mesh, 'mesh/cylinder_physical_region.xml')
boundaries = MeshFunction('size_t', mesh, 'mesh/cylinder_facet_region.xml')

# Define function spaces (P2-P1)
V = VectorFunctionSpace(mesh, "P", 2)
Q = FunctionSpace(mesh, "P", 1)

# Define trial and test functions
u = TrialFunction(V)
p = TrialFunction(Q)
v = TestFunction(V)
q = TestFunction(Q)

#####################################################################
##############DEFINE BOUNDARY CONDITIONS#############################
#####################################################################
# Last number refers to the surface number in gmsh, see file attached
bcp_inflow = DirichletBC(Q, Constant(8), boundaries, 1)
bcp_outflow = DirichletBC(Q, Constant(0), boundaries, 2)

# top and bottom boundaries conditions are defined equals with surface number 3
top_bottom_boundaries = DirichletBC(V, Constant((0, 0)), boundaries, 3)

# Inner cylinder is surface numer 4
cylinder_boundaries = DirichletBC(V, Constant((0, 0)), boundaries, 4)

bcu = [cylinder_boundaries, top_bottom_boundaries]
bcp = [bcp_inflow, bcp_outflow]

# HERE BELOW NO CHANGES FROM THE DEMO
# Define functions for solutions at previous and current time steps
u_n = Function(V)
u_ = Function(V)
p_n = Function(Q)
p_ = Function(Q)

# Define expressions used in variational forms
U = 0.5*(u_n + u)
n = FacetNormal(mesh)
f = Constant((0, 0))
k = Constant(dt)
mu = Constant(mu)
rho = Constant(rho)

# Define strain-rate tensor


def epsilon(u):
    return sym(nabla_grad(u))

# Define stress tensor


def sigma(u, p):
    return 2*mu*epsilon(u) - p*Identity(len(u))


# Define variational problem for step 1
F1 = rho*dot((u - u_n) / k, v)*dx + \
    rho*dot(dot(u_n, nabla_grad(u_n)), v)*dx \
    + inner(sigma(U, p_n), epsilon(v))*dx \
    + dot(p_n*n, v)*ds - dot(mu*nabla_grad(U)*n, v)*ds \
    - dot(f, v)*dx
a1 = lhs(F1)
L1 = rhs(F1)

# Define variational problem for step 2
a2 = dot(nabla_grad(p), nabla_grad(q))*dx
L2 = dot(nabla_grad(p_n), nabla_grad(q))*dx - (1/k)*div(u_)*q*dx

# Define variational problem for step 3
a3 = dot(u, v)*dx
L3 = dot(u_, v)*dx - k*dot(nabla_grad(p_ - p_n), v)*dx

# Assemble matrices
A1 = assemble(a1)
A2 = assemble(a2)
A3 = assemble(a3)

# Apply boundary conditions to matrices
[bc.apply(A1) for bc in bcu]
[bc.apply(A2) for bc in bcp]

# Time-stepping
t = 0
for n in range(num_steps):

    # Update current time
    t += dt

    # Step 1: Tentative velocity step
    b1 = assemble(L1)
    [bc.apply(b1) for bc in bcu]
    solve(A1, u_.vector(), b1)

    # Step 2: Pressure correction step
    b2 = assemble(L2)
    [bc.apply(b2) for bc in bcp]
    solve(A2, p_.vector(), b2)

    # Step 3: Velocity correction step
    b3 = assemble(L3)
    solve(A3, u_.vector(), b3)

    # Compute error
    u_e = Expression(('4*x[1]*(1.0 - x[1])', '0'), degree=2)
    u_e = interpolate(u_e, V)
    error = np.abs(u_e.vector() - u_.vector()).max()

    print('t = {0:4.2f}: error = {1:4.2f}'.format(t, error))
    print('max u:', u_.vector().max())

    # Update previous solution
    u_n.assign(u_)
    p_n.assign(p_)

file = File('results/u.pvd')
file << u_n
