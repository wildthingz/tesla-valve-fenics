# daniele gmsh's trial

What I tried and learnt

1) create simple geometry in gmsh
2) define surfaces (lines) in gmsh
3) import geometry and mesh in fenics-python
4) define boundary conditions on gmsh geometry

## Stabilization Methods
1) numerical instabilities arise from two sources: inappropriate use of finite element spaces and in convection dominated problema (when Re>>1).
2) using mixed fe spaces should solve the first instability.
3) Adding stabilization terms should solve the second instabilities when Re is high

**SUPG stabilization**
