# `mshr` test

The idea here is to make a simple geometry file and mesh it using `mshr`.

## Things I've learned so far

- `mshr` only accepts limited number file formats, these include: `.stl`, `.off` and `.vtp` files
- `.stl` files are easy to generate. The catch is the `STLReader` only reads txt formatted `stl`s. Binaries are no good!

- `openmpi` is used to handle the parallel processing, but there is a bug when a custom file is being meshed. I managed to figure out a way around it by manually initializing the mpi.

- A good way of looking at the mesh files and results in general in ParaView. However, there is bug in the lastest version of the ParaView (5.8). Downgrading to version 5.7 works.
