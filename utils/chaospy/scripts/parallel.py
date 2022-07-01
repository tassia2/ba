console_info = True

###-------------------------------------------------------------------------###
### Import python modules
###-------------------------------------------------------------------------###

from mpi4py import MPI

def num_proc ( ):
  """
  Get number of utilised processors
  input:
  return:
    nproc: Number of processors
  """
 
  nproc = MPI.COMM_WORLD.Get_size()

  if console_info:
    print ( "Number of processors:\n", nproc )

  return nproc

def rank ( ):
  """
  Get rank of processor
  input:
  return:
    r: Rank of processor
  """

  r = MPI.COMM_WORLD.Get_rank()

  if console_info:
    print ( "Local rank:\n", r )

  return r

def chunkify ( num_tasks, num_proc ):
  """
  Split a number of tasks in possibly equal sized lists of task
  for a number of processor or threads
  input:
    num_tasks: Number of tasks
    num_proc: Number of processor or threads to be utilized
  return:
    task_lists: List of local task lists
  """

  tasks = list( range( num_tasks ) )

  task_lists = [tasks[i::num_proc] for i in range( num_proc )[::-1]]

  if console_info:
    print ( "Task lists:\n", task_lists )

  return task_lists

def local_chunk ( num_tasks ):
  """
  Get local chunk of tasks
  input:
    num_tasks: Number of tasks
  return:
    tasks: Local list of tasks
  """

  task_lists = chunkify ( num_tasks, num_proc( ) )

  tasks = task_lists[rank( )]

  if console_info:
    print ( "Local tasks:\n", tasks )

  return tasks



