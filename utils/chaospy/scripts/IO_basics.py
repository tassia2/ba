console_info = False

###-------------------------------------------------------------------------###
### Import python modules
###-------------------------------------------------------------------------###

import os
import sys
import numpy as np

def make_folder ( folder ):
  """
  Make folder
  Creation of a folder for a collocation node
  input:
    folder: Folder name
  return:
    success state
  """
  if not os.path.exists( folder ):
    try:
      os.mkdir( folder )
    except:
      print ( "Could not create folder", folder )
      print ( "Make sure that parent folder exists." )
      quit( )

  if console_info:
    print ( "Created folder:\n", folder )

  return True

def copy_folder ( source_folder, dest_folder ):
  """
  Copy folder
  Creation of a folder for a collocation node
  input:
    source_folder: Folder to copy from
    dest_folder: Folder to be created
  return:
    success state
  """
  if not os.path.exists( dest_folder ):
    try:
      os.system('cp -r ' + source_folder + ' ' + dest_folder)
    except:
      print ( "Could not create folder", dest_folder )
      print ( "Make sure that parent folder exists." )
      quit( )

  if console_info:
    print ( "Copied folder:\n", dest_folder )

  return True

def read_file ( filename ):
  """
  Open file
  input:
    filename: Path and filename
  return:
    content
  """
  content = ""
  try:
    f = open( filename, 'r' )
    content = f.read( )
    f.close( )
  except:
    print( "Could not open file", filename )
    quit( )

  if console_info:
    print ( "Opened file:\n", filename )
    print ( "Content:\n", content )

  return content

def read_numbers ( filename ):
  """
  Open file with numbers
  input:
    filename: Path and filename
  return:
    content
  """
  numbers = 0
  try:
    numbers = np.loadtxt( filename )
  except:
    print( "Could not get numbers from file", filename )
    quit( )

  if console_info:
    print ( "Read numbers from file:\n", filename )
    print ( "Numbers:\n", numbers )

  return numbers

def write_file ( content, filename ):
  """
  Write file
  input:
    filename: Path and filename
  writes:
    file
  """
  try:
    f = open( filename, 'w' )
    f.write( content )
    f.close( )
  except:
    print( "Could not write file", filename )
    quit( )

  if console_info:
    print ( "Wrote file:\n", filename )

def generate_file_list ( file_prefix, file_postfix, start, end, increment, nb_digits ):
  """
  Generate file list for example for time series
  input:
    file_prefix: prefix of the files
    file_postfix: postfix of the files
    start: start value
    end: end value to be included in the list
    increment: increment from one file to the next
    nb_digits: number of digits of the counter
  return:
    files: file list
  """

  files = [ file_prefix + str( i ).zfill( nb_digits ) + file_postfix
            for i in range( start, end + increment, increment ) ]

  if console_info:
    print ( "File list:\n", files )
    print ( "Number of files in file list:\n", len( files ) )

  return files

