#!/usr/bin/env python

# pylint: disable-all
# Copyright (c) 2017-2018 Johan Jansson, Cem Degirmenci
from fenics import *
from mshr import *
import matplotlib.pyplot as plt
from ufl import replace

# Compact plot utility function
def plot_compact(u, t, stepcounter, QQ, pl, ax, label="Velocity", clear=True):
    if stepcounter == 0:
        pl, ax = plt.subplots()
        display(pl)
        if clear:
            clear_output()  # Plotting setup
    if stepcounter % 5 == 0:
        uEuclidnorm = project(sqrt(inner(u, u)), QQ)
        ax.cla()
        fig = plt.gcf()
        fig.set_size_inches(16, 4)
        plt.subplot(1, 2, 1)
        pp = plot(uEuclidnorm, cmap="coolwarm")
        plt.title("%s at t=%f" % (label, t))  # Plot norm of velocity
        if t == 0.:
            plt.axis(G)
            plt.colorbar(pp, shrink=0.5)
        plt.subplot(1, 2, 2)
        if t == 0.:
            plot(Q.mesh())
            plt.title("Mesh")  # Plot mesh
        plt.tight_layout()

    return (pl, ax)


def plot_compact_static(u, QQ, label="Velocity"):  # Compact plot utility function
    pl, ax = plt.subplots()
    uEuclidnorm = project(sqrt(inner(u, u)), QQ)
    ax.cla()
    fig = plt.gcf()
    fig.set_size_inches(16, 4)
    plt.subplot(1, 2, 1)
    pp = plot(uEuclidnorm, cmap="coolwarm")
    plt.title("%s" % (label))  # Plot norm of velocity
    plt.axis(G)
    plt.colorbar(pp, shrink=0.5)
    plt.subplot(1, 2, 2)
    plot(Q.mesh())
    plt.title("Mesh")  # Plot mesh
    plt.tight_layout()

    return (pl, ax)


XMIN = 0.
XMAX = 2.5
YMIN = 0
YMAX = 0.41
G = [XMIN, XMAX, YMIN, YMAX]
eps = 1e-5
mresolution = 10  # Geometry and mesh
ghole = [0.15, 0.25, 0.15, 0.25]
mesh = generate_mesh(Rectangle(Point(G[0], G[2]), Point(
    G[1], G[3])) - Rectangle(Point(ghole[0], ghole[2]), Point(ghole[1], ghole[3])), mresolution)


XMIN = 0.
XMAX = 2.5
YMIN = 0
YMAX = 0.41
G = [XMIN, XMAX, YMIN, YMAX]
eps = 1e-5
mresolution = 10  # Geometry and mesh
ghole = [0.15, 0.25, 0.15, 0.25]
mesh = generate_mesh(Rectangle(Point(G[0], G[2]), Point(
    G[1], G[3])) - Rectangle(Point(ghole[0], ghole[2]), Point(ghole[1], ghole[3])), mresolution)


def M(mesh, u, p):
    """Output functional"""
    def epsilon(z):
        return 0.5*(grad(z) + grad(z).T)

    n = FacetNormal(mesh)

    I = Identity(2)
    sigma = p*I - 2*nu*epsilon(u)
    theta = Constant((1.0, 0.0))

    g = Expression(
        ("200.0*exp(-200.0*(pow(x[0] - 0.5, 2) + pow(x[1] - 0.3, 2)))", "0.0"), degree=2)

    M1 = psimarker*p*n[0]*ds  # Drag (only pressure)
    M2 = psimarker*p*n[1]*ds  # Lift (only pressure)
    M3 = inner(g, u)*dx  # Mean of the velocity in a region
    M4 = psimarker*dot(dot(sigma, n), theta)*ds  # Drag (full stress)
    M5 = u[0]*dx  # Mean of the x-velocity in the whole domain

    return M1


ref = 0.128069

nu = 4e-3  # Viscosity
adaptive = True
adapt_ratio = 0.03
maxiters = 20

for i in range(0, maxiters):
    # Declare FE spaces
    VE = VectorElement("CG", mesh.ufl_cell(), 1)
    QE = FiniteElement("CG", mesh.ufl_cell(), 1)
    h = CellDiameter(mesh)  # FEM functions
    WE = VE * QE
    W = FunctionSpace(mesh, WE)
    V = FunctionSpace(mesh, VE)
    Q = FunctionSpace(mesh, QE)  # Make a mixed space
    wt = TestFunction(W)
    (v, q) = (as_vector((wt[0], wt[1])), wt[2])
    w = Function(W)
    (u, p) = (as_vector((w[0], w[1])), w[2])
    u0 = Function(V)
    phi = Function(W)
    (phi_u, phi_p) = (as_vector((phi[0], phi[1])), phi[2])

    ZE = FiniteElement("DG", mesh.ufl_cell(), 0)
    Z = FunctionSpace(mesh, ZE)
    z = TestFunction(Z)

    uin = Expression(("4*(x[1]*(YMAX-x[1]))/(YMAX*YMAX)", "0."),
                     YMAX=YMAX, element=V.ufl_element())  # Inflow velocity
    om = Expression("x[0] > XMAX - eps ? 1. : 0.", XMAX=XMAX, eps=eps,
                    element=Q.ufl_element())  # Mark regions for boundary conditions
    im = Expression("x[0] < XMIN + eps ? 1. : 0.", XMIN=XMIN,
                    eps=eps, element=Q.ufl_element())
    nm = Expression("x[0] > XMIN + eps && x[0] < XMAX - eps ? 1. : 0.",
                    XMIN=XMIN, XMAX=XMAX, eps=eps, element=Q.ufl_element())
    psimarker = Expression("x[0] > XMIN + eps && x[1] >  YMIN + eps && x[0] < XMAX - eps && x[1] < YMAX - eps ? 1. : 0.",
                           XMIN=XMIN, XMAX=XMAX, YMIN=YMIN, YMAX=YMAX, eps=eps, element=Q.ufl_element())

    n = FacetNormal(mesh)
    d = .5*h  # Stabilization parameter
    gamma = 10*1./h  # Penalty parameter

    r = ((inner(grad(p) + grad(u)*u, v) + nu*inner(grad(u), grad(v)) + div(u)*q)*dx +
         # Weak boundary conditions
         gamma*(om*p*q + im*inner(u - uin, v) + nm*inner(u, v))*ds +
         d*(inner(grad(p) + grad(u)*u, grad(q) + grad(u)*v) + inner(div(u), div(v)))*dx)  # Stabilization
    solve(r == 0, w)  # Solve the Navier-Stokes PDE (stationary)
    # Plot all quantities (see implementation above)
    plot_compact_static(u, Q, "Velocity")

    output = assemble(M(mesh, u, p))
    print("adaptive iter: %d num_vertices: %d output: %f err: %f" %
          (i, mesh.num_vertices(), output, abs(ref - output)))

    a_dual = adjoint(derivative(r, w))  # Construct the adjoint problem
    L_dual = M(mesh, v, q)
    # Solve the adjoint Navier-Stokes PDE (stationary)
    solve(a_dual == L_dual, phi)

    # Plot all quantities (see implementation above)
    plot_compact_static(phi, Q, "Adjoint Velocity")

    Lei = replace(r, {wt: phi*z})  # Construct error indicators
    ei = assemble(Lei)
    # Get value array (assume serial implementation)
    gamma = abs(ei.get_local())

    # Mark cells for refinement
    cell_markers = MeshFunction("bool", mesh, mesh.topology().dim())
    gamma_0 = sorted(gamma, reverse=True)[int(len(gamma)*adapt_ratio) - 1]
    for c in cells(mesh):
        cell_markers[c] = gamma[c.index()] > gamma_0

    if adaptive:  # Refine mesh
        rmesh = refine(mesh, cell_markers)
    else:
        rmesh = refine(mesh)
    mesh = rmesh  # Shift to next adaptive iteration

plt.show()
