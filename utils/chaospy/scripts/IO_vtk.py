###-------------------------------------------------------------------------###
### Import python modules
###-------------------------------------------------------------------------###

from os import path
import numpy as np
import vtk
from vtk.util import numpy_support as vtknp
import xml.etree.ElementTree as xmlET
vtk_version = int( vtk.vtkVersion.GetVTKVersion( ).split( '.' )[0] )
if ( vtk_version < 8 ):
  print ( "Error: VTK version >= 8 needed!" )
  quit( )

console_info = False
###-------------------------------------------------------------------------###
### Import UQ scripts
###-------------------------------------------------------------------------###

import IO_basics as iob

def is_xml_vtk ( filename ):
  """
  Checks if vtk file is organised as xml or not
  input:
    filename: path and name of vtk file
  return:
    True if vtk file is xml and False of not
  """
  is_xml = False
  with open( filename ) as f:
    first_line = f.readline( )
    if 'xml' in first_line:
      is_xml = True
  return is_xml

def vtk_reader ( filename ):
  """
  Setup a vtk reader
  input:
    filename: path and name of vtk file
  return:
    grid
  """
  if console_info:
    print ( "Read vtk file:\n", filename )

  ### Define vtk reader
  is_xml = is_xml_vtk ( filename )
  vtk_type = filename.split( '.' )[-1]
  reader = 0
  if ( vtk_type == 'vtu' or vtk_type == 'vtk' ):
    if is_xml:
      reader = vtk.vtkXMLUnstructuredGridReader( )
    else:
      reader = vtk.vtkUnstructuredGridReader( )
  elif ( vtk_type == 'pvtu' ):
    if is_xml:
      reader = vtk.vtkXMLPUnstructuredGridReader( )
    else:
      reader = vtk.vtkPUnstructuredGridReader( )
  else:
    print ( "Error: Unknown vtk file extension:\n", vtk_type )
    print ( "Of file:\n", filename )
    quit()

  reader.SetFileName( filename )
  try:
    reader.Update( )
  except:
    print ( "Could not open vtk file:\n", filename )
    quit( )

  return reader.GetOutput( )

def read_vtk_file ( filename, array_names, points=[],console_info=False ):
  """
  Read vtk file and return its data
  input:
    filename: path and name of vtk file
    array_names: names of the data arrays to be read
  return:
    data: 1D numpy array containing the data arrays from input file
    offsets: offsets for indexing original data arrays
  """
  ### prepare return variables
  data = []
  offsets = [0]
  
  ### Get vtk grid
  grid = vtk_reader( filename )

  if console_info:
    print ( "Read arrays by the names:\n", array_names )
  
  if console_info:
    ### Print information on number of arrays in the vtk file
    num_arrays = [grid.GetPointData( ).GetNumberOfArrays( ), 
                  grid.GetCellData( ).GetNumberOfArrays( )]
    print ( "Number of point and cell arrays in vtk file:\n", num_arrays )

  ### Iterate array names
  for an in array_names:
    if console_info:
      print ( "Getting:\n", an )
    array = None
    ### Firstly, check if point coordinates should be read
    # if ( an == "Points" ):
    array_index, cut_index = np.unique(np.rint(vtknp.vtk_to_numpy( grid.GetPoints( ).GetData( ) )/0.015625).astype(int), axis=0, return_index=True)
    # X = array_index[:,0]
    # Y = array_index[:,1]
    # array_index = vtknp.vtk_to_numpy( grid.GetCells( ).GetData( ) )
    if console_info:
        print ( "Type: Point coordinates" )
    # else:
    ### Try to read point data array
    array1 = vtknp.vtk_to_numpy( grid.GetPointData( ).GetArray( an ) )
    array = array1[cut_index]
    if console_info and ( array != None ):
        print ( "Type: PointData" )
    ### Check if it's a cell data array
    # if ( array == None ):
    #   array = grid.GetCellData( ).GetArray( an )
    #   if console_info and ( array != None ):
    #       print ( "Type: CellData" )
    # ### Exit if array name could not be found
    # if ( array == None ):
    #   print ( "Error: Unknown array name:\n", an )
    #   print ( "In file:\n", filename )
    #   quit()

    ### Conversion of the vtk array to a numpy array
    # data_tmp = vtknp.vtk_to_numpy( array )
    data_tmp = array 
    if console_info:
      print ( "Data shape:\n", np.shape( data_tmp ) )

    data_tmp2 = []
    #only specific points
    if len(points) >0:
        data_tmp2 = [data_tmp[point[1]*128+point[0]] for point in points]
    else:
      data_tmp2 = data_tmp          
    
    ### Append the data with a flattened array
    if ( data_tmp.ndim > 1 ):
      data.append( np.concatenate( data_tmp2 ) )
    else:
      data.append( data_tmp2 )

    ### Add array size to the offsets list
    offsets.append( offsets[-1] + np.shape( data[-1] )[0] )

  if console_info:
    print ( "Index offsets in vtk data:\n", offsets )

  ### flatten data to a single array
  data = np.concatenate( data )

  if console_info:
    print ( "Overall size of vtk data:\n", np.shape( data ) )

  return data, offsets

def read_vtk_nodes ( work_folder, vtk_filename, array_names, num_nodes, points, console_info=False ):
  """
  Read vtk file from all nodes and return its data in a 2D numpy array
  input:
    work_folder: path to base folder containing all node folder
    vtk_filename: vtk file within a respective node folder
    array_names: names of the data arrays to be read
    num_nodes: number of nodes
    return:
      nodes_data: 2D numpy array containing all data arrays
        i-th row corresponds tp concatendated data from  i-th node
      offsets: offsets for indexing original data arrays
  """
  nodes_data = []
  offsets = []
  input_folder_prefix = path.join( work_folder, 'node_' )
  
  for n in range( num_nodes ):
    node_folder = input_folder_prefix + str(n)
    node_filename = path.join( node_folder, vtk_filename )
  
    if console_info:
      print ( "Read vtk file:\n", node_filename )
    data, offsets_tmp = read_vtk_file ( node_filename, array_names, points, console_info=False )

    if ( n == 0 ):
      offsets = offsets_tmp
    elif ( offsets != offsets_tmp ):
      print ( "Error: vtk data offsets missmatch:\n",
              offsets, " != ", offsets_tmp )
      quit()

    nodes_data.append( data )

  nodes_data = np.array( nodes_data )
  
  return nodes_data, offsets

def write_vtu ( dummy_filename, write_filename, data, offsets, array_names ):
  """
  Write vtu file
  input:
    dummy_filename: A dummy vtk file as template for the structure
    write_filename: path and name of the output vtu file
    data: 1D numpy array containing all data arrays
    offsets: offsets for indexing data arrays
    array_names: names of the data arrays to be written
  writes:
    New vtu file containing the data
  """

  ### load vtk dummy file
  grid = vtk_reader( dummy_filename )
  is_xml = is_xml_vtk ( dummy_filename )

  if console_info:
    print ( "Writing vtu data." )
    print ( "Index offsets for vtu data:\n", offsets )
    print ( "Names of the arrays to be changed:\n", array_names )

  ### Iterate array names
  for i_an in range( len( array_names ) ):
    if console_info:
      print ( "Setting:\n", array_names[i_an] )
    array = None
    ### Firstly, check if point coordinates should be read
    if ( array_names[i_an] == "Points" ):
      array = grid.GetPoints( ).GetData( )
      if console_info:
         print ( "Type: Point coordinates" )
    else:
      ### Try to read point data array
      array = grid.GetPointData( ).GetArray( array_names[i_an] )
      if console_info and ( array != None ):
         print ( "Type: PointData" )
      ### Check if it's a cell data array
      if ( array == None ):
        array = grid.GetCellData( ).GetArray( array_names[i_an] )
        if console_info and ( array != None ):
           print ( "Type: CellData" )
    ### Exit if array name could not be found
    if ( array == None ):
      print ( "Error: Unknown array name:\n", array_names[i_an] )
      print ( "In file:\n", dummy_filename )
      quit()

    ### Get current subset of the data
    data_tmp = data[offsets[i_an]:offsets[i_an+1]]
    ### Reshape w.r.t. the number of components of the array
    nb_comp = array.GetNumberOfComponents( )
    if console_info:
      print ( "Number of Components:\n", nb_comp )
    if ( nb_comp > 1 ):
      data_tmp = data_tmp.reshape( int( len( data_tmp )/nb_comp ), nb_comp )
    if console_info:
      print ( "Data shape:\n", np.shape( data_tmp ) )
    ### Conversion from numpy array to vtk array
    data_tmp = vtknp.numpy_to_vtk( data_tmp )
    data_tmp.SetName( array_names[i_an] )
    ### overwrite dummy array data with input data
    array.ShallowCopy( data_tmp )

  ### write vtu file
  if console_info:
    print ( "Write vtu file:\n", write_filename)
  writer = 0
  if is_xml:
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetDataModeToAscii( )
  else:
    writer = vtk.vtkUnstructuredGridWriter()

  writer.SetFileName(write_filename)
  writer.SetInputData( grid )
  writer.Write( )

def read_pvtu_structure ( filename, node_2_pvtu_path = "" ):
  """
  Read structure of a pvtu file
  input:
    filename: path and name of the input pvtu file
  return:
    vtu_filenames: List of attached vtu files
  """
  pvtu2vtu_filenames = []
  node2vtu_filenames = []
  
  tree = 0
  try:
    tree = xmlET.parse ( filename ).getroot( )
  except:
    print ( "Could not open pvtu file:\n", filename )
    quit( )

  for piece in tree[0].findall( 'Piece' ):
    node2vtu_filenames.append( path.join(node_2_pvtu_path, piece.attrib['Source']) )
    pvtu2vtu_filenames.append( piece.attrib['Source'] )

  if console_info:
    print ( "Pieces of pvtu:\n", pvtu2vtu_filenames)

  return node2vtu_filenames, pvtu2vtu_filenames

def adapt_pvtu ( dummy_filename, write_filename,
                 old_vtu_filenames, new_vtu_filenames ):
  """
  Write pvtu file
  input:
    dummy_filename: A dummy pvtu file as template for the structure
    write_filename: path and name of the output pvtu file
    old_vtu_filenames: List of vtu path and filenames to be replaced
    new_vtu_filenames: List of vtu path and filenames to be set
  writes:
    New pvtu file
  """

  num_vtu = len( old_vtu_filenames )
  if console_info:
    print ( "pvtu adaption for number of vtu pieces:\n", num_vtu)

  if ( num_vtu != len( new_vtu_filenames ) ):
    print ( "Error: Length of file names does not match:\n",
            num_vtu, "!=", len( new_vtu_filenames ) )
    quit( )

  content = iob.read_file ( dummy_filename )

  if console_info:
    print ( "pvtu dummy:\n", content)

  ### Change vtu filenames from old to new
  for p in range( num_vtu ):
    if True:
      print ( "replace ",  old_vtu_filenames[p], " by ", new_vtu_filenames[p] )
    content = content.replace( old_vtu_filenames[p], new_vtu_filenames[p] )

  if console_info:
    print ( "New pvtu content:\n", content )

  iob.write_file ( content, write_filename )

