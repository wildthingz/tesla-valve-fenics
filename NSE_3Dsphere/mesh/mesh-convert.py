import meshio
msh = meshio.read("3Dsphere.msh")

for key in msh.cell_data_dict["gmsh:physical"].keys():
    if key == "triangle":
        triangle_data = msh.cell_data_dict["gmsh:physical"][key]
    elif key == "tetra":
        tetra_data = msh.cell_data_dict["gmsh:physical"][key]

for cell in msh.cells:
    if cell.type == "tetra":
        tetra_cells = cell.data
    elif cell.type == "triangle":
        triangle_cells = cell.data

tetra_mesh = meshio.Mesh(points=msh.points, cells={"tetra": tetra_cells},
                         cell_data={"domain":[tetra_data]})

triangle_mesh =meshio.Mesh(points=msh.points,
                           cells=[("triangle", triangle_cells)],
                           cell_data={"boundaries":[triangle_data]})

meshio.write("domain.xdmf", tetra_mesh)
meshio.write("boundaries.xdmf", triangle_mesh)

