import numpoly 
import numpy as np
import matplotlib.pyplot as plt
import os
from configparser import ConfigParser
import re 

import IO_vtk
import qoi_postprocessing as qoipp
import chaospy as cp
import PC_collocation as pcc

work_folder =  '/home/tassia/Hiflow-2-22/ba-v2/build/examples/boussinesq2d'

qoi_names = ["p"]
PC_degree = 8
quadrature_rule = 'g'
sparse_flag = False
dist_list = [cp.Uniform(2, 13.5), cp.Uniform(1e4, 1e6)]
basis, basis_norms, nodes, weights, dist, num_nodes = pcc.create_PC_collocation (PC_degree, quadrature_rule, sparse_flag, dist_list)
vtk_filename = 'ChannelBenchmark_solution0050_0.vtu'

nn = len( weights )
nodes_data, offsets = \
    IO_vtk.read_vtk_nodes ( work_folder, vtk_filename, qoi_names, nn, [])

x = np.array(nodes_data)
X = x[50,:].reshape((65,65))

plt.imshow(X)
plt.show()
print("done")
