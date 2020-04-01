#!/usr/bin/env python

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import matplotlib.pyplot as plt

reader = vtk.vtkXMLUnstructuredGridReader()
reader.SetFileName("./results/biharmonic000000.vtu")
reader.Update()
data = reader.GetOutput()

# print(data)

points = data.GetPoints()
npts = points.GetNumberOfPoints()
x = vtk_to_numpy(points.GetData())

triangles=  vtk_to_numpy(data.GetCells().GetData())
ntri = triangles.size//4
tri = np.take(triangles,[n for n in range(triangles.size) if n%4 != 0]).reshape(ntri,3)

n_arrays = reader.GetNumberOfPointArrays()

for i in range(n_arrays):
    print(reader.GetPointArrayName(i))


u = vtk_to_numpy(data.GetPointData().GetArray(0))

# Mesh
plt.figure(figsize=(8, 8))
plt.triplot(x[:,0], x[:,1], tri)
plt.gca().set_aspect('equal')



# abs_u
plt.figure(figsize=(8, 8))
plt.tricontour(x[:,0], x[:,1], tri, u, 16)



plt.figure(figsize=(8, 8))
plt.tricontourf(x[:,0], x[:,1], tri, u, 16)

plt.show()





