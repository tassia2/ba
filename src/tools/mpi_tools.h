// Copyright (C) 2011-2021 Vincent Heuveline
//
// HiFlow3 is free software: you can redistribute it and/or modify it under the
// terms of the European Union Public Licence (EUPL) v1.2 as published by the
// European Union or (at your option) any later version.
//
// HiFlow3 is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
// A PARTICULAR PURPOSE. See the European Union Public Licence (EUPL) v1.2 for
// more details.
//
// You should have received a copy of the European Union Public Licence (EUPL)
// v1.2 along with HiFlow3.  If not, see
// <https://joinup.ec.europa.eu/page/eupl-text-11-12>.

#ifndef HIFLOW_MPITOOLS__
#define HIFLOW_MPITOOLS__

///
/// \author Martin Wlotzka
///

#include <mpi.h>
#include "common/vector_algebra.h"

template < typename DataType > struct mpi_data_type {};

template <> struct mpi_data_type< long double > {

  static MPI_Datatype get_type() { return MPI_LONG_DOUBLE; }
};

template <> struct mpi_data_type< double > {

  static MPI_Datatype get_type() { return MPI_DOUBLE; }
};

template <> struct mpi_data_type< float > {

  static MPI_Datatype get_type() { return MPI_FLOAT; }
};

template <> struct mpi_data_type< int > {

  static MPI_Datatype get_type() { return MPI_INT; }
};

template <> struct mpi_data_type< unsigned int > {

  static MPI_Datatype get_type() { return MPI_UNSIGNED; }
};

template <> struct mpi_data_type< char > {

  static MPI_Datatype get_type() { return MPI_CHAR; }
};

#if 0
template < class T > 
int MPI_Allreduce_T  (T* send_data,
                                              T* recv_data,
                                              int count,
                                              MPI_Datatype datatype,
                                              MPI_Op op,
                                              MPI_Comm communicator );

template <> 
int MPI_Allreduce_T < double > (double* send_data,
                                          double* recv_data,
                                          int count,
                                          MPI_Datatype datatype,
                                          MPI_Op op,
                                          MPI_Comm communicator )
{
  return MPI_Allreduce(send_data, recv_data, count, datatype, op, communicator);
}

template <> 
int MPI_Allreduce_T < hiflow::Vec<2, double> > (hiflow::Vec<2, double>* send_data,
                                                  hiflow::Vec<2, double>* recv_data,
                                                  int count,
                                                  MPI_Datatype datatype,
                                                  MPI_Op op,
                                                  MPI_Comm communicator )
{
  std::vector<double> send_data_array;
  std::vector<double> recv_data_array(2*count);
  
  send_data_array.reserve(2*count);
  
  for (size_t l=0; l<count; ++l)
  {
    for (size_t d=0; d<2; ++d)
    {
      send_data_array.push_back(send_data[l][d]);
    }
  }

  int status = MPI_Allreduce(&send_data_array[0], &recv_data_array[0], count * 2, datatype, op, communicator);
  
  for (size_t l=0; l<count; ++l)
  {
    for (size_t d = 0; d < 2; ++d)
    {
      recv_data[l][0] = recv_data_array[l];
    }
  }
  
  return status;
}
#endif

#endif
