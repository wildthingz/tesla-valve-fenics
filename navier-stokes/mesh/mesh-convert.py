#!/usr/bin/env python

import meshio
import numpy as np
mesh = meshio.read("cylinder.msh")

line_cells = []
for cell in mesh.cells:
    if cell.type == "triangle":
        triangle_cells = cell.data
    elif  cell.type == "line":
        if len(line_cells) == 0:
            line_cells = cell.data
        else:
            line_cells = np.vstack([line_cells, cell.data])

line_data = []
for key in mesh.cell_data_dict["gmsh:physical"].keys():
    if key == "line":
        if len(line_data) == 0:
            line_data = mesh.cell_data_dict["gmsh:physical"][key]
        else:
            line_data = np.vstack([line_data, mesh.cell_data_dict["gmsh:physical"][key]])
    elif key == "triangle":
        triangle_data = mesh.cell_data_dict["gmsh:physical"][key]

triangle_mesh = meshio.Mesh(points=mesh.points[:,:2], cells={"triangle": triangle_cells})
line_mesh =meshio.Mesh(points=mesh.points[:,:2],
                           cells=[("line", line_cells)],
                           cell_data={"name_to_read":[line_data]})
meshio.write("mesh.xdmf", triangle_mesh)

meshio.xdmf.write("mf.xdmf", line_mesh)