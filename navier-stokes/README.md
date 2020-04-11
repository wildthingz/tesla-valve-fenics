# Navier Stokes C++

Translating the python version to C++

## Things I've learned so far

- Overall it was straight-forward the only pain point was reading the mesh file.
- According to [this](https://fenicsproject.discourse.group/t/converting-simple-2d-mesh-from-gmsh-to-dolfin/583) `dolfin-convert` and overall `.xml` type meshed are going to be depreciated.
- It is recommended to use `.XMDF` type mesh formats.
- In order to convert `.msh` files to `.XMDF` format you need to use `meshio`.
- I used [this](https://fenicsproject.discourse.group/t/transitioning-from-mesh-xml-to-mesh-xdmf-from-dolfin-convert-to-meshio/412/110) and [this](https://fenicsproject.discourse.group/t/load-marked-mesh-in-cpp-for-parallel-running/2318) with slight modifications to convert the mesh and read it in the c++ file.
- In order to run in parallel you need you compile `dolfin` with MPI configured `hdf5` libraries. I did this by installing a package called `hdf5-openmpi` and then compiling `dolfin` again.
- The simulation is now significantly faster.
- The results are the same.
- In order to run in parallel, ones needs to compile the in following order:
  - `hdf5-openmpi`/`hdf5`
  - `scotch`
  - `mumps`
  - `petsc`
  - `dolfin`
- run `mpirun -n <num_threads> <file_name>`
