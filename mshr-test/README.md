# `mshr` test

The idea here is to make a simple geometry file and mesh it using `mshr`.

## Things I've learned so far

### 3D meshes

- `mshr` only accepts limited number file formats, these include: `.stl`, `.off` and `.vtp` files
- `.stl` files are easy to generate. The catch is the `STLReader` only reads txt formatted `stl`s. Binaries are no good!
- `openmpi` is used to handle the parallel processing, but there is a bug when a custom file is being meshed. I managed to figure out a way around it by manually initializing the mpi.
- A good way of looking at the mesh files and results in general in ParaView. However, there is bug in the lastest version of the ParaView (5.8). Downgrading to version 5.7 works.
- According to [this](https://fenicsproject.discourse.group/t/creating-subdomains-in-mshr-with-an-stl-file/2868) `mshr` is obsolete. Although it is a pain but we should use the following workflow:
  - Generate `.step` file in OnShape
  - Mesh and name the subdomains in `gmsh`
  - Write `.xdmf` file using meshio
  - Simulate in `fenics`

### 2D meshes

- Overall it was straight-forward the only pain point was reading the mesh file.
- According to [this](https://fenicsproject.discourse.group/t/converting-simple-2d-mesh-from-gmsh-to-dolfin/583) `dolfin-convert` and overall `.xml` type meshed are going to be depreciated.
- It is recommended to use `.XMDF` type mesh formats.
- In order to convert `.msh` files to `.XMDF` format you need to use `meshio`.
- I used [this](https://fenicsproject.discourse.group/t/transitioning-from-mesh-xml-to-mesh-xdmf-from-dolfin-convert-to-meshio/412/110) and [this](https://fenicsproject.discourse.group/t/load-marked-mesh-in-cpp-for-parallel-running/2318) with slight modifications to convert the mesh and read it in the c++ file.
- In order to run in parallel you need you compile `dolfin` with MPI configured `hdf5` libraries. I did this by installing a package called `hdf5-openmpi` and then compiling `dolfin` again.
