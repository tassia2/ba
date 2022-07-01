import os
import convert_vtk_2_inp as v2i

# box size
dx = 0.1
dy = 0.1

# scaling of foil: size of foild relative to dx
scaling_foil = 0.1

# angle of attack, in radians i.e. values in (-pi, pi)
angle_foil_1 = -0.4
angle_foil_2 = -0.4

foil_dist = 0.25

# mesh resolution near boundary and near foil. small values -> finer mesh
resolution_bdy = 0.4
resolution_foil = 0.04

# 2D or 3D
dim = 2

fileprefix = "channel"

path_2_gmsh = "../../gmsh/gmsh"

#####################################
# write into gmsh param file
f = open("channel_params.geo", "w")
f.write("dx = " + str(dx) + "; \n dy = " + str(dy) 
        + "; \n scaling_foil = " + str(scaling_foil) + "; \n foil_rot1 = " + str(angle_foil_1) + "; \n foil_rot2 = " + str(angle_foil_2)
        + "; \n foil_distance = " + str(foil_dist)  
        + "; \n resolution_bdy = " + str(resolution_bdy) + "; \n resolution_foil = " + str(resolution_foil) + "; \n")
f.close()

# call gmsh to create mesh in vtk format
os.system(path_2_gmsh + " channel.geo -2 -o " + fileprefix + ".vtk")

# get data from vtk file
points, entity_num, entity_data, entity_types, entity_matnumber  = v2i.read_vtk_file("channel.vtk")

# write data to inp file
v2i.write_inp_file("../" + fileprefix + ".inp", dim, points, entity_num, entity_data, entity_types, entity_matnumber)







