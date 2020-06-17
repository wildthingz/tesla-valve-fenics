#!/usr/bin/env python

# pylint: disable-all

from fenics import *
from mshr import *


def make_mesh(xmin, xmax, ymin, ymax):
    mresolution = 10
    G = [xmin, xmax, ymin, ymax]
    ghole = [0.15, 0.25, 0.15, 0.25]
    mesh = generate_mesh(Rectangle(Point(G[0], G[2]), Point(
        G[1], G[3])) - Rectangle(Point(ghole[0], ghole[2]), Point(ghole[1], ghole[3])), mresolution)
    return mesh


def M(mesh, w, nu, psimarker):
    """Output functional"""
    (u, p) = (as_vector((w[0], w[1])), w[2])

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
    return M5


def R(w, w2):
    (u, p) = (as_vector((w[0], w[1])), w[2])
    (u2, p2) = (as_vector((w2[0], w2[1])), w2[2])
    Au = grad(p) + grad(u2)*u
    Ap = div(u)
    Aui = [Au[i] for i in range(0, 2)]
    return as_vector(Aui + [Ap])


def r(W, w, wt, nu, stab=True):
    (u, p) = (as_vector((w[0], w[1])), w[2])
    (v, q) = (as_vector((wt[0], wt[1])), wt[2])

    F = (nu * inner(grad(u), grad(v)) +
         inner(grad(p) + grad(u)*u, v) + div(u)*q) * dx
    if stab:
        delta = 0.5 * CellDiameter(W.mesh())  # GLS stabilization parameter
        F_stab = delta * inner(R(w, w), R(wt, w)) * dx
        F += F_stab
    return F


def main():

    xmin = 0.
    xmax = 2.5
    ymin = 0
    ymax = 0.41

    eps = 1e-5
    ref = 0.0
    tol = 5e-4

    nu = 4e-3  # Viscosity
    adaptive = True
    adapt_ratio = 0.03
    maxiters = 30

    fileU = File("velocity.pvd")
    filePhi = File("phi.pvd")
    # initial mesh
    mesh = make_mesh(xmin, xmax, ymin, ymax)

    def om(x, on_boundary):
        return on_boundary and (x[0] > (xmax - eps))

    def im(x, on_boundary):
        return on_boundary and (x[0] < (xmin + eps))

    def nm(x, on_boundary):
        return on_boundary and (
            (x[0] > (xmin + eps)) and
            (x[0] < (xmax - eps))
        )

    for i in range(maxiters):
        # Declare FE spaces
        VE = VectorElement("CG", mesh.ufl_cell(), 1)
        QE = FiniteElement("CG", mesh.ufl_cell(), 1)

        WE = VE * QE
        W = FunctionSpace(mesh, WE)
        V = FunctionSpace(mesh, VE)
        Q = FunctionSpace(mesh, QE)  # Make a mixed space

        wt = TestFunction(W)
        w = Function(W)
        u0 = Function(V)
        phi = Function(W)

        uin = Expression(("4*(x[1]*(ymax-x[1]))/(ymax*ymax)", "0."),
                         ymax=ymax, element=V.ufl_element())  # Inflow velocity

        psimarker = Expression("x[0] > xmin + eps && x[1] >  ymin + eps && x[0] < xmax - eps && x[1] < ymax - eps ? 1. : 0.",
                               xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, eps=eps, element=Q.ufl_element())

        noslip = DirichletBC(W.sub(0), Constant((0, 0)), nm)
        inflow = DirichletBC(W.sub(0), uin, im)
        outflow = DirichletBC(W.sub(1), Constant(0), om)
        

        bcs = [inflow, outflow, noslip]
        F = r(W, w, wt, nu, stab=True)
        # Solve the Navier-Stokes PDE (stationary)
        solve(F == 0, w, bcs=bcs)
        # Plot all quantities (see implementation above)
        fileU << w

        output = assemble(M(mesh, w, nu, psimarker))
        print("adaptive iter: %d num_vertices: %d output: %f err: %f" %
              (i, mesh.num_vertices(), output, abs(ref - output)))
        if abs(ref-output) < tol:
            print("convergence reached!")
            break
        ref = output
        
        a_dual = adjoint(derivative(F, w))  # Construct the adjoint problem
        L_dual = M(mesh, wt, nu, psimarker)
        # Solve the adjoint Navier-Stokes PDE (stationary)
        solve(a_dual == L_dual, phi, bcs)

        # Plot all quantities (see implementation above)
        filePhi << phi

        ZE = FiniteElement("DG", mesh.ufl_cell(), 0)
        Z = FunctionSpace(mesh, ZE)
        z = TestFunction(Z)

        Lei = r(W, w, z*phi, nu, stab=False)  # Construct error indicators
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
            rmesh = refine(mesh, cell_markers)
        else:
            rmesh = refine(mesh)
        mesh = rmesh  # Shift to next adaptive iteration


if __name__ == "__main__":
    main()
