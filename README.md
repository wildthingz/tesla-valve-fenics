# daniele gmsh's trial

What I tried and learnt

1) create simple geometry in gmsh
2) define surfaces (lines) in gmsh
3) import geometry and mesh in fenics-python
4) define boundary conditions on gmsh geometry

# Drag force
- [this](https://readthedocs.org/projects/fenics-handson/downloads/pdf/latest/)
- [this](http://www.featflow.de/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark1_re20.html)

# Stabilization Methods
1) numerical instabilities arise from two sources: inappropriate use of finite element spaces and in convection dominated problema (when Re>>1).
2) using mixed fe spaces should solve the first instability.
3) Adding stabilization terms should solve the second instabilities when Re is high

## SUPG stabilization

To add SUPG stabilization we first need to define the residual of Navier Stokes equations
```python
#Residual of strong Navier-Stokes
r = 1/k*(w.split()[0]- w0.split()[0]) + theta*grad(w.split()[0])*w.split()[0] \
    + (1.0-theta)*grad(w0.split()[0])*w0.split()[0] \
    - theta*div(sigma(w, nu, mesh)) - (1.0-theta)*div(sigma(w0, nu, mesh))
```

with the following stabilization parameters
```python
h = CellDiameter(mesh)
u_norm = sqrt(dot(w.split()[0], w.split()[0]))
tau_supg = ((2.0/k)**2 + (2.0*u_norm/h)**2 + (4.0*nu/h**2)**2)**(-0.5)
```

SUPG stabilization term is then
```python
F_supg = tau_supg*inner(grad(u)*v, r)*dx
```

## PSPG stabilization
```python
tau_pspg = h**2/(4*nu)
F_pspg = tau_pspg*inner(grad(q), r)*dx
```
Note: solver does not converge with PSPG, more tests are needed.

## LSIC stabilization
```python
tau_lsic = -2/3 * nu
Flsic = tau_lsic*inner(div(v), div(u))*dx
```

Note: solver does not converge with LSIC