# Navier-Stokes Benchmark

## Introduction

The goal of this code is to benchmark different schemes to solve incompressible Navier-Stokes equations.

The benchmark problem is stated [here](https://fenicsproject.org/pub/tutorial/html/._ftut1009.html).

## `fenics` installation

In order to run in parallel, ones needs to compile the in following order:

- `hdf5-openmpi`/`hdf5`
- `hypre`
- `scotch`
- `mumps`
- `petsc`
- `dolfin`

## Creating mesh

- Create 2D geometry in gmesh OR Create 3D geometry in OnShape, export as `.step` and import to gmesh.
- Name subdomains in gmesh.
- Mesh and save as `.msh` file.
- Use meshio to convert it to `.xdmf`.
- Read mesh and boundaries in c++ as such:

  ```cpp
  auto mesh = std::make_shared<Mesh>();
  XDMFFile initialmeshfile(MPI_COMM_WORLD, <path-to-domain.xdmf>);
  initialmeshfile.read(*mesh);

  MeshValueCollection<std::size_t> mf_mvc(mesh, 2);
  XDMFFile initialmffile(MPI_COMM_WORLD, <path-to-boundaries.xdmf>);
  initialmffile.read(mf_mvc, "boundaries");
  auto boundaries = std::make_shared<MeshFunction<std::size_t>>(mesh, mf_mvc);
  ```

  OR in python as such:

  ```python
  mesh = Mesh()
  with XDMFFile(<path-to-domain.xdmf>) as infile:
      infile.read(mesh)
  mvc = MeshValueCollection("size_t", mesh, 2)
  with XDMFFile(<path-to-boundaries.xdmf>) as infile:
      infile.read(mvc, "boundaries")
  mf = cpp.mesh.MeshFunctionSizet(mesh, mvc)
  ```

## Problem formulation

- [**IPCS**](./IPCS/README.md)
- [**Chorin**](./Chorin/README.md)
- [**Mixed**](./Mixed/README.md)

## Run problem

Run using `run.sh` script
