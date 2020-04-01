#!/usr/bin/env python

import numpy as np
import vtk
import sys
import os
from vtk.util.numpy_support import vtk_to_numpy
import matplotlib.pyplot as plt


class view:
    def __init__(self):
        if len(sys.argv) == 1:
            raise IOError('Must input a file name')
        else:
            self.filepath = sys.argv[1]
            print(self.filepath)
            if not os.path.exists(self.filepath):
                raise IOError('File does not exist')

        self._get_data()
        self._get_data()
        self._get_mesh()
        self._get_results()

    def _get_data(self):
        self.reader = vtk.vtkXMLUnstructuredGridReader()
        self.reader.SetFileName(self.filepath)
        self.reader.Update()
        self.data = self.reader.GetOutput()

    def _get_mesh(self):
        points = self.data.GetPoints()
        npts = points.GetNumberOfPoints()
        self.x = vtk_to_numpy(points.GetData())

        triangles = vtk_to_numpy(self.data.GetCells().GetData())
        ntri = triangles.size//4
        self.tri = np.take(triangles, [n for n in range(
            triangles.size) if n % 4 != 0]).reshape(ntri, 3)

    def _get_results(self):
        self.u = vtk_to_numpy(self.data.GetPointData().GetArray(0))

    def _draw_mesh(self):
        plt.figure(figsize=(8, 8))
        plt.triplot(self.x[:, 0], self.x[:, 1], self.tri)
        plt.gca().set_aspect('equal')

    def _draw_u(self):
        plt.figure(figsize=(8, 8))
        plt.tricontourf(self.x[:, 0], self.x[:, 1], self.tri, self.u, 16)

    def draw(self):
        self._draw_mesh()
        self._draw_u()
        plt.show()


if __name__ == "__main__":
    viewer = view()
    viewer.draw()
