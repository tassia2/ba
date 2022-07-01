import numpy as np

def read_vtk_file(filename):

  read_points = False
  read_entities = False
  read_types = False
  read_scalar = False
  
  entity_data = []
  entity_types = []
  entity_num = []
  entity_matnumber = []
  
  with open(filename) as fp:
    line = fp.readline()
    cnt = 1
    ent_ctr = 0
    tmp_ent_data = []
    
    while line:
      words = line.split()
              
      if (len(words) == 0):
        line = fp.readline()
        cnt += 1
        continue
        
      #print (words)
      if read_points:
        points[point_id,0] = float(words[0])
        points[point_id,1] = float(words[1])
        points[point_id,2] = float(words[2])
        point_id += 1
        
        if point_id == num_points:
          read_points = False
      
      if read_entities:         
        tmptmp = []
        for val in words[1:]:
          tmptmp.append(int(val))
          
        entity_data.append(tmptmp)
        ent_ctr += 1
        if ent_ctr == num_entities:
          read_entities = False
          
      if read_types:
        entity_types.append(int(words[0]))
        ent_ctr += 1
        if ent_ctr == num_entities:
          read_types = False
      
      if read_scalar:
        entity_matnumber.append(int(words[0]))
        ent_ctr += 1
        if ent_ctr == num_entities:
          read_scalar = False
                    
      if words[0] == "POINTS":
        num_points = int(words[1])
        points = np.zeros((num_points, 3))
        read_points = True
        point_id = 0
      if words[0] == "CELLS":
        read_entities = True
        ent_ctr = 0
        num_entities = int(words[1])
        num_if = words[2]
      if words[0] == "CELL_TYPES":
        read_types = True
        num_entities = int(words[1])
        ent_ctr = 0
 
      if words[0] == "CELL_DATA":
        line = fp.readline()
        line = fp.readline()
        cnt += 2
        read_scalar = True
        
      line = fp.readline()
      cnt += 1

  for i in range(0, len(entity_data)):
    entity_num.append(len(entity_data[i]))
    
  return points, entity_num, entity_data, entity_types, entity_matnumber

# inp keyword for entity
def convert_entity_type(int_type):
  if int_type == 1:
    return "point"
  elif int_type == 3:
    return "line"
  elif int_type == 5:
    return "tri"
  else :
    return "none"

# topological dimension of entity
def get_entity_dim(str_type):
  if str_type == "point":
    return 0
  elif str_type == "line":
    return 1
  elif str_type == "tri":
    return 2
  elif str_type == "quad":
    return 2
  elif str_type == "tet":
    return 3
  elif str_type == "hex":
    return 3
  else:
    return -1
  
def write_inp_file (filename, dim, points, entity_num, entity_data, entity_types, entity_matnumber):
  f = open( filename,"w+")
  
  num_points, d = np.shape(points)
  num_ent = 0
  
  for i in range(0, len(entity_types)):
    if get_entity_dim(convert_entity_type(entity_types[i])) >= dim-1:
      num_ent += 1
  
  f.write(str(num_points) + " " + str(num_ent) + " 0 0 0\n")
  
  # write points
  for p in range(0, num_points):
    f.write(str(p))
    for d in range(0, 3):
      f.write("  " + "{:.9f}".format(points[p][d]))
    f.write("\n")
  
  ent_ctr = 0
  ent_id = 0
  
  # write entities
  for l in range(0, len(entity_types)):
    ent_type = entity_types[l]
    ent_str_type = convert_entity_type(ent_type)
    ent_tdim = get_entity_dim(ent_str_type)
    ent_matnum = entity_matnumber[l]
    
    if ent_tdim < dim-1:
      # only cells and facets
      continue
      
    f.write(str(ent_ctr) + " " + str(ent_matnum) + " " + ent_str_type)
      
    for d in range(0, len(entity_data[l])):
      f.write(" " + str(entity_data[l][d]))
    f.write("\n")
    ent_ctr += 1
      
  f.close()
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  


