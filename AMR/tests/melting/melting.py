#!/usr/bin/env python
from fenics import *
import numpy as np
import time
import matplotlib.pyplot as plt


def Plot(w):
    p, u, T = split(w.leaf_node())

    plot(phi(T))

    plt.title("Solid volume fraction")

    plt.xlabel("$x$")

    plt.ylabel("$y$")

    plt.show()

    plot(u)

    plt.title("Velocity vector field")

    plt.xlabel("$x$")

    plt.ylabel("$y$")

    plt.show()
    R = w.leaf_node().function_space()
    plot(R.mesh())

    plt.title("Adapted mesh")

    plt.xlabel("$x$")

    plt.ylabel("$y$")
    plt.show()


N = 1

mesh = UnitSquareMesh(N, N)


class HotWall(SubDomain):

    def inside(self, x, on_boundary):

        return on_boundary and near(x[0], 0.)


hot_wall = HotWall()

initial_hot_wall_refinement_cycles = 6

for cycle in range(initial_hot_wall_refinement_cycles):

    edge_markers = MeshFunction("bool", mesh, 1, False)

    hot_wall.mark(edge_markers, True)

    mesh = refine(mesh, edge_markers)

plot(mesh)
plt.show()
P1 = FiniteElement('P', mesh.ufl_cell(), 1)

P2 = VectorElement('P', mesh.ufl_cell(), 2)

mixed_element = MixedElement([P1, P2, P1])

W = FunctionSpace(mesh, mixed_element)


psi_p, psi_u, psi_T = TestFunctions(W)

w = Function(W)

p, u, T = split(w)


prandtl_number = 56.2

Pr = Constant(prandtl_number)

rayleigh_number = 3.27e5

Ra = Constant(rayleigh_number)

stefan_number = 0.045

Ste = Constant(stefan_number)


f_B = Ra/Pr*T*Constant((0., -1.))


regularization_central_temperature = 0.01

T_r = Constant(regularization_central_temperature)

regularization_smoothing_parameter = 0.025

r = Constant(regularization_smoothing_parameter)


def tanh(x):
    return ((exp(x) - exp(-x)) / (exp(x) + exp(-x)))


def phi(T):
    return 0.5*(1. + tanh((T_r - T)/r))


liquid_dynamic_viscosity = 1.

mu_L = Constant(liquid_dynamic_viscosity)

solid_dynamic_viscosity = 1.e8

mu_S = Constant(solid_dynamic_viscosity)


def mu(phi):

    return mu_L + (mu_S - mu_L)*phi


hot_wall_temperature = 1.

T_h = Constant(hot_wall_temperature)

cold_wall_temperature = -0.01

T_c = Constant(cold_wall_temperature)


initial_melt_thickness = 1./2.**(initial_hot_wall_refinement_cycles - 1)

w_n = interpolate(
    Expression(("0.", "0.", "0.", "(T_h - T_c)*(x[0] < x_m0) + T_c"),
               T_h=hot_wall_temperature, T_c=cold_wall_temperature,
               x_m0=initial_melt_thickness,
               element=mixed_element), W)

p_n, u_n, T_n = split(w_n)

plot(T_n)
plt.show()

timestep_size = 10.

Delta_t = Constant(timestep_size)

u_t = (u - u_n)/Delta_t

T_t = (T - T_n)/Delta_t

phi_t = (phi(T) - phi(T_n))/Delta_t


dx = dx(metadata={'quadrature_degree': 8})

mass = -psi_p*div(u)

momentum = dot(psi_u, u_t + dot(grad(u), u) + f_B) - div(psi_u)*p \
    + 2.*mu(phi(T))*inner(sym(grad(psi_u)), sym(grad(u)))

enthalpy = psi_T*(T_t - 1./Ste*phi_t) + dot(grad(psi_T), 1./Pr*grad(T) - T*u)

F = (mass + momentum + enthalpy)*dx


penalty_stabilization_parameter = 1.e-7

gamma = Constant(penalty_stabilization_parameter)

F += -psi_p*gamma*p*dx


JF = derivative(F, w, TrialFunction(W))


hot_wall = "near(x[0],  0.)"

cold_wall = "near(x[0],  1.)"

adiabatic_walls = "near(x[1],  0.) | near(x[1],  1.)"

walls = hot_wall + " | " + cold_wall + " | " + adiabatic_walls

W_u = W.sub(1)

W_T = W.sub(2)

boundary_conditions = [
    DirichletBC(W_u, (0., 0.), walls),
    DirichletBC(W_T, hot_wall_temperature, hot_wall),
    DirichletBC(W_T, cold_wall_temperature, cold_wall)]


problem = NonlinearVariationalProblem(F, w, boundary_conditions, JF)


M = phi_t*dx

epsilon_M = 4.e-5


solver = AdaptiveNonlinearVariationalSolver(problem, M)
prm = solver.parameters
prm["nonlinear_variational_solver"]["newton_solver"]["absolute_tolerance"] = 1E-9
info(prm, True)
time.sleep(2)
w.leaf_node().vector()[:] = w_n.leaf_node().vector()

solver.solve(epsilon_M)


for timestep in range(5):
    Plot(w)
    w_n.leaf_node().vector()[:] = w.leaf_node().vector()
    # w_n.vector()[:] = w.vector()
    # w_n.assign(w)
    solver.solve(epsilon_M)
