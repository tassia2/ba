import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

pathlist = sorted(list(Path(" '/home/tassia/Hiflow-2-22/ba-v2/build/examples/boussinesq2d/node_0/results_points").rglob('*.csv')))
num_points = len(pathlist)
dim = int(np.sqrt(num_points))
start = True
shape = (np.loadtxt(pathlist[0],skiprows=1, delimiter=",")).shape
points = np.empty((dim,dim,shape[0],shape[1]))

# for path in pathlist:
#     path_in_str = str(path.resolve())
#     print(path_in_str)
#     point = np.loadtxt(path_in_str,skiprows=1, delimiter=",")
#     if(start):
#         points = point[None]
#         start = False
#     else:
#         points = np.append(points,point[None],axis=0)
#     print(points.shape)
y = 0
x = 0
for path in pathlist:
    path_in_str = str(path.resolve())
    point = np.loadtxt(path_in_str,skiprows=1, delimiter=",")
    if (y == dim):
        x += 1
        y = 0
    points[x,y] = point
    y += 1
       

std = np.std(points[:,:,:,2], axis=2)
for i in range(14):
    fig = plt.figure()
    plt.title("pressure timestep "+ str(i+1))
    plt.imshow(points[:,:,i,2], cmap='Blues')
    plt.colorbar()
    plt.savefig('/home/tassia/Hiflow-2-22/ba-v2/build/examples/boussinesq2d/node_0/img/' + str(i) +".png")
