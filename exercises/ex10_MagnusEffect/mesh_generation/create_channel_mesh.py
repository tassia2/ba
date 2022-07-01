import os
import convert_vtk_2_inp as v2i

# box size
dx = 2.0
dy = 1.0
cx = 0.5
cy = 0.5
r = 0.05

# mesh resolution near boundary and near foil. small values -> finer mesh
resolution_bdy = 0.1
resolution_foil = 0.005

# 2D or 3D
dim = 2

fileprefix = "channel"
path_2_gmsh = "../../gmsh/gmsh"

#####################################
# write into gmsh param file
f = open("channel_params.geo", "w")
f.write("dx = " + str(dx) 
        + "; \n dy = " + str(dy) 
        + "; \n r = " + str(r)
        + "; \n cx = " + str(cx)
        + "; \n cy = " + str(cy)
        + "; \n resolution_bdy = " + str(resolution_bdy) 
        + "; \n resolution_foil = " + str(resolution_foil) + "; \n")
f.close()

# call gmsh to create mesh in vtk format
os.system(path_2_gmsh + " channel.geo -2 -o " + fileprefix + ".vtk")

# get data from vtk file
points, entity_num, entity_data, entity_types, entity_matnumber  = v2i.read_vtk_file("channel.vtk")

# write data to inp file
v2i.write_inp_file("../" + fileprefix + ".inp", dim, points, entity_num, entity_data, entity_types, entity_matnumber)







