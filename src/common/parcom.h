// Copyright (C) 2011-2020 Vincent Heuveline
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

#ifndef HIFLOW_COMMON_PARCOM_H
#define HIFLOW_COMMON_PARCOM_H

/// \author Philipp Gerstner

#include <mpi.h>
#include <vector>
#include <cassert>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <numeric>
#include "common/array_tools.h"
#include "common/sorted_array.h"
#include "mesh/types.h"

namespace hiflow {
///
/// \brief   Toolbox for MPI communication
///
/// \details 
///
class ParCom 
{
public:

  /// \brief   
  ParCom(const MPI_Comm &comm, int master_rank)
  : comm_(comm), master_rank_(master_rank), rank_(-1), size_(0)
  {
    MPI_Comm_rank(comm_, &this->rank_);
    MPI_Comm_size(comm_, &this->size_);
  }
  
  ParCom(const MPI_Comm &comm)
  : comm_(comm), master_rank_(0), rank_(-1), size_(0)
  {
    MPI_Comm_rank(comm_, &this->rank_);
    MPI_Comm_size(comm_, &this->size_);
  }
  
  // default constructor should not be used
  ParCom()
  : comm_(MPI_COMM_WORLD), master_rank_(0), rank_(-1), size_(0)
  {
    MPI_Comm_rank(comm_, &this->rank_);
    MPI_Comm_size(comm_, &this->size_);
  }
  
  ~ParCom()
  {
  }
  
  inline const MPI_Comm& comm() const 
  {
    return this->comm_;
  }
  
  inline int rank() const 
  {
    return this->rank_;
  }
  
  inline int size() const 
  {
    return this->size_;
  }
  
  inline int master_rank() const 
  {
    return this->master_rank_;
  }
  
  inline bool is_master() const 
  {
    return (this->rank_ == this->master_rank_);
  }
  
  template <typename T>
  MPI_Datatype get_mpi_type(const T &dummy) const;

  int barrier() const
  {
    int state = MPI_Barrier(this->comm_);
    return state;
  }
  
  int wait(MPI_Request& request) const
  {
    MPI_Status state;
    int err = MPI_Wait(&request, &state);
    return err;
  }
  
  void determine_neighbor_domains(mesh::ConstMeshPtr mesh) const;
  
  /// \brief return maximum over all processes
  template <typename T>
  int max(const T &local_val, T& result) const;

  template <typename T>
  int max(const std::vector<T>& local_vals, T& result) const;

  template <typename T>
  int min(const T& local_val, T& result) const;

  template <typename T>
  int min(const std::vector<T>& local_vals, T& result) const;

  template <typename T>
  int sum(const T& local_val, T& result) const;

  template <typename T>
  int sum(const std::vector<T>& local_vals, std::vector<T>& result) const;
  
  template <typename T>
  int sum(const std::vector< std::vector<T> >& local_vals, std::vector< std::vector<T> >& result) const;
 
  template <typename T>
  int sum(const std::vector<T>& local_vals, T& result) const;
   
  template <typename T, int DIM, typename VectorType>
  int sum(const std::vector< std::vector< VectorType > >& local_vals, 
                std::vector< std::vector< VectorType > >& result) const;

  int global_and(bool local_val, bool& global_val) const;

  int global_or(bool local_val, bool& global_val) const;
  
  int global_xor(bool local_val, bool& global_val) const;

  /// brief various send and receive routines.
  /// Import: always use matching send/receive functions (same "option")
  
  // send option I
  template <typename T>
  void send(const T& local_val, int recv_rank, int tag) const;
  
  // send option II  
  template <typename T>
  void send(const std::vector<T>& local_vals, int recv_rank, int tag) const;

  // send option III
  template <typename T>
  void send(const std::vector<T>& local_vals, int size, int recv_rank, int tag) const;

  // send option IV
  //template <typename T>
  //void Isend(const std::vector<T>& local_vals, int recv_rank, int tag, MPI_Request& request) const;

  // send option V
  template <typename T>
  void Isend(const std::vector<T>& local_vals, int size, int recv_rank, int tag, MPI_Request& request) const;
  
  // send option VI
  template <typename T>
  void send(T* local_vals, int size, int recv_rank, int tag) const;
  
  // send option VII
  template <typename T>
  void Isend(T const * local_vals, int size, int recv_rank, int tag, MPI_Request& request) const;
  
  // receive option I
  template <typename T>
  int recv(T& local_val, int send_rank, int tag) const;

  // receive option II
  template <typename T>
  int recv(std::vector<T>& local_vals, int send_rank, int tag) const;

  // receive option III
  template <typename T>
  int recv(std::vector<T>& local_vals, int size, int send_rank, int tag) const;
  
  // receive option IV
  //template <typename T>
  //int Irecv(std::vector<T>& local_vals, int send_rank, int tag, MPI_Request& request) const;

  // receive option V
  template <typename T>
  int Irecv(std::vector<T>& local_vals, int size, int send_rank, int tag, MPI_Request& request) const;

  // receive option VI
  template <typename T>
  int recv(T* local_vals, int size, int send_rank, int tag) const;
  
  // receive option VII
  template <typename T>
  int Irecv(T* local_vals, int size, int send_rank, int tag, MPI_Request& request) const;
  
  // allgather, one element per process
  template <typename T>
  int allgather(const T& local_val, std::vector<T>& recv_vals) const;

  template <typename T>
  void broadcast_data_to_neighbors (mesh::ConstMeshPtr mesh,
                                    const std::vector<T>& my_data,
                                    std::vector< std::vector<T> >& neighbor_data) const;  

  template <typename T>
  int broadcast (std::vector<T>& data, int root) const;

private:
  const MPI_Comm comm_;
  
  int rank_;
  int size_;
  int master_rank_;
  
  mutable SortedArray<int> neighbor_domains_;
    
};

template <typename T>
int ParCom::broadcast (std::vector<T>& data, int root) const
{
  int size = 0;
  T dummy;
  if (this->rank_ == root)
  {
    size = data.size();
  }
  int err = MPI_Bcast(&size, 1, MPI_INT, root, this->comm_);

  if (err != MPI_SUCCESS)
  {
    return err;
  }
  if (this->rank_ != root)
  {
    data.clear();
    data.resize(size);
  }
  err = MPI_Bcast(&data[0], size, this->get_mpi_type(dummy), root, this->comm_);
  return err;
}

template <typename T>
void ParCom::broadcast_data_to_neighbors (mesh::ConstMeshPtr mesh,
                                          const std::vector<T>& my_data,
                                          std::vector< std::vector<T> >& neighbor_data) const
{
  // determine neighboring subdomains
  this->determine_neighbor_domains(mesh);
  neighbor_data.clear();
  neighbor_data.resize(this->size());
  
  const std::vector<int>& my_neighbors = this->neighbor_domains_.data();
  const int num_neighbors = my_neighbors.size();
  
  // send size of my_data to my neighbors
  std::vector<MPI_Request> send_reqs_1 (num_neighbors);
  std::vector<MPI_Request> recv_reqs_1 (num_neighbors);
  const int my_data_size = my_data.size();
  std::vector<int> neighbor_data_sizes(num_neighbors, 0);
  
  for (int n=0; n!=num_neighbors; ++n)
  {
    const int neighbor = my_neighbors[n];
    this->Isend(&(my_data_size), 1, neighbor, 0, recv_reqs_1[n]);
  }
  
  for (int n=0; n!=num_neighbors; ++n)
  {
    const int neighbor = my_neighbors[n];
    this->Irecv(&(neighbor_data_sizes[n]), 1, neighbor, 0, send_reqs_1[n]);
  }
  
  for (int n=0; n!=num_neighbors; ++n)
  {
    int err = this->wait(send_reqs_1[n]);
    assert (err == 0);
  }
  
  // allocate memory
  for (int n=0; n!=num_neighbors; ++n)
  {
    const int neighbor = my_neighbors[n];
    neighbor_data[neighbor].resize(neighbor_data_sizes[n], 0);
  }
  
  // broadcast actual data
  std::vector<MPI_Request> send_reqs (num_neighbors);
  std::vector<MPI_Request> recv_reqs (num_neighbors);
  
  for (int n=0; n!=num_neighbors; ++n)
  {
    const int neighbor = my_neighbors[n];
    if (my_data_size == 0)
    {
      break;
    }
    this->Isend(&(my_data[0]), my_data_size, neighbor, 2, send_reqs[n]);
  }
  
  for (int n=0; n!=num_neighbors; ++n)
  {
    const int neighbor = my_neighbors[n];
    if (neighbor_data_sizes[n] == 0)
    {
      continue;
    }
    this->Irecv(&(neighbor_data[neighbor][0]), neighbor_data_sizes[n], neighbor, 2, recv_reqs[n]);
  }
  
  for (int n=0; n!=num_neighbors; ++n)
  {
    const int neighbor = my_neighbors[n];
    if (neighbor_data_sizes[n] == 0)
    {
      continue;
    }
    int err = this->wait(recv_reqs[n]);
    assert (err == 0);
  }
}

template <typename T>
int ParCom::max(const T& local_val, T& result) const
{
  result = -1e10;
  int err = MPI_Allreduce(&local_val, &result, 1, this->get_mpi_type(local_val), MPI_MAX, this->comm_); 
  return err;
}

template <typename T>
int ParCom::max(const std::vector<T>& local_vals, T& result) const
{
  assert (local_vals.size() > 0);
  
  T local_max = -1e10;
  result = -1e10;
  
  local_max = *std::max_element(local_vals.begin(), local_vals.end());
    
  int err = MPI_Allreduce(&local_max, &result, 1, this->get_mpi_type(local_vals[0]), MPI_MAX, this->comm_); 
  return err;
}

template <typename T>
int ParCom::min(const T& local_val, T& result) const
{
  result = 1e10;
  int err = MPI_Allreduce(&local_val, &result, 1, this->get_mpi_type(local_val), MPI_MIN, this->comm_); 
  return err;
}

template <typename T>
int ParCom::min(const std::vector<T>& local_vals, T& result) const
{
  assert (local_vals.size() > 0);
  
  T local_min = 1e10;
  result = 1e10;
  
  local_min = *std::min_element(local_vals.begin(), local_vals.end());
    
  int err = MPI_Allreduce(&local_min, &result, 1, this->get_mpi_type(local_vals[0]), MPI_MIN, this->comm_); 
  return err;
}

template <typename T>
int ParCom::sum(const T& local_val, T& result) const
{
  result = 0;
  int err = MPI_Allreduce(&local_val, &result, 1, this->get_mpi_type(local_val), MPI_SUM, this->comm_); 
  return err;
}

template <typename T>
int ParCom::sum(const std::vector<T>& local_vals, T& result) const
{
  assert (local_vals.size() > 0);
  
  T local_sum = 0.;
  result = 0;
  
  local_sum = std::accumulate(local_vals.begin(), local_vals.end(), 0);
    
  int err = MPI_Allreduce(&local_sum, &result, 1, this->get_mpi_type(local_vals[0]), MPI_SUM, this->comm_); 
  return err;
}

template <typename T>
int ParCom::sum(const std::vector<T>& local_vals, std::vector<T>& result) const
{
  assert (local_vals.size() > 0);
  
  if (result.size() != local_vals.size())
  {
    result.clear();
    result.resize(local_vals.size());
  }
       
  int err = MPI_Allreduce(&local_vals[0], &result[0], result.size(), 
                          this->get_mpi_type(local_vals[0]), MPI_SUM, this->comm_); 
  return err;
}

template <typename T>
int ParCom::sum(const std::vector< std::vector<T> >& local_vals, std::vector< std::vector<T> >& result) const
{
  assert (local_vals.size() > 0);
  std::vector<size_t> sizes;
  size_t total_size = 0;
  
  for (size_t d=0; d<local_vals.size(); ++d)
  {
    total_size += local_vals[d].size();
  }
  
  std::vector<T> tmp_local;
  flatten_2d_array(local_vals, tmp_local, sizes);
  
  std::vector<T> tmp_result(total_size);
  
  int err = MPI_Allreduce(&tmp_local[0], &tmp_result[0], total_size, 
                          this->get_mpi_type(tmp_local[0]), MPI_SUM, this->comm_); 
  
  expand_to_2d_array (tmp_result, sizes, result);
  
  assert (result.size() == local_vals.size());
  return err;
}

template <typename T, int DIM, typename VectorType>
int ParCom::sum(const std::vector< std::vector< VectorType > >& local_vals, 
                std::vector< std::vector< VectorType > >& result) const
{
  assert (local_vals.size() > 0);
  std::vector<size_t> sizes;
  size_t total_size = 0;
  
  for (size_t d=0; d<local_vals.size(); ++d)
  {
    total_size += local_vals[d].size();
  }
  
  std::vector<T> tmp_local;
  flatten_2d_vec_array(local_vals, tmp_local, sizes);
  
  std::vector<T> tmp_result(total_size * DIM);
  
  int err = MPI_Allreduce(&tmp_local[0], &tmp_result[0], total_size, 
                          this->get_mpi_type(tmp_local[0]), MPI_SUM, this->comm_); 
  
  expand_to_2d_vec_array<T, DIM, VectorType> (tmp_result, sizes, result);
  
  assert (result.size() == local_vals.size());
  return err;
}

template <typename T>
void ParCom::send(const T& local_val, int recv_rank, int tag) const
{
  MPI_Send(&local_val, 1, this->get_mpi_type(local_val), recv_rank, tag, this->comm_);
}

template <typename T>
void ParCom::send(const std::vector<T>& local_vals, int recv_rank, int tag) const
{
  int size = local_vals.size(); 
  MPI_Send(&size,1, MPI_INT, recv_rank, tag, this->comm_);
  MPI_Send(&local_vals[0],size, this->get_mpi_type(local_vals[0]), recv_rank, 10000+tag, this->comm_);
}

template <typename T>
void ParCom::send(const std::vector<T>& local_vals, int size, int recv_rank, int tag) const
{
  assert (size >= 0);
  MPI_Send(&local_vals[0],size, this->get_mpi_type(local_vals[0]), recv_rank, tag, this->comm_);
}

template <typename T>
void ParCom::send(T* local_vals, int size, int recv_rank, int tag) const
{
  assert (size >= 0);
  MPI_Send(local_vals, size, this->get_mpi_type(local_vals[0]), recv_rank, tag, this->comm_);
}

/*
template <typename T>
void ParCom::Isend(const std::vector<T>& local_vals, int recv_rank, int tag, MPI_Request& request) const
{
  int size = local_vals.size(); 
  MPI_Request tmp_request;
  MPI_Isend(&size,1, MPI_INT, recv_rank, tag, this->comm_, &tmp_request);
  MPI_Isend(&local_vals[0], size, this->get_mpi_type(local_vals[0]), recv_rank, 10000+tag, this->comm_, &request);
}
*/

template <typename T>
void ParCom::Isend(const std::vector<T>& local_vals, int size, int recv_rank, int tag, MPI_Request& request) const
{
  assert (size >= 0);
  MPI_Isend(&local_vals[0], size, this->get_mpi_type(local_vals[0]), recv_rank, tag, this->comm_, &request);
}

template <typename T>
void ParCom::Isend(T const * local_vals, int size, int recv_rank, int tag, MPI_Request& request) const
{
  assert (size >= 0);
  MPI_Isend(local_vals, size, this->get_mpi_type(local_vals[0]), recv_rank, tag, this->comm_, &request);
}

template <typename T>
int ParCom::recv(T& local_val, int send_rank, int tag) const
{
  MPI_Status state;
  int err = MPI_Recv(&local_val, 1, this->get_mpi_type(local_val), send_rank, tag, this->comm_, &state);
  return err;
}

template <typename T>
int ParCom::recv(std::vector<T>& local_vals, int send_rank, int tag) const
{
  MPI_Status state;
  int size = -1;
  MPI_Recv(&size, 1, MPI_INT, send_rank, tag, this->comm_, &state);
  
  if (size >= 0 && state.MPI_ERROR == 0)
  {
    local_vals.clear();
    local_vals.resize(size);
  
    int err = MPI_Recv(&local_vals[0], size, this->get_mpi_type(local_vals[0]), send_rank, 10000+tag, this->comm_, &state);
  
    return err;
  }
  else
  {
    return state.MPI_ERROR;
  }
}

template <typename T>
int ParCom::recv(std::vector<T>& local_vals, int size, int send_rank, int tag) const
{
  MPI_Status state;
  assert (size >= 0);
  local_vals.clear();
  local_vals.resize(size);
  
  int err = MPI_Recv(&local_vals[0], size, this->get_mpi_type(local_vals[0]), send_rank, tag, this->comm_, &state);
  return err;
}

template <typename T>
int ParCom::recv(T* local_vals, int size, int send_rank, int tag) const
{
  MPI_Status state;
  assert (size >= 0);
  int err = MPI_Recv(local_vals, size, this->get_mpi_type(local_vals[0]), send_rank, tag, this->comm_, &state);
  return err;
}

/*
template <typename T>
int ParCom::Irecv(std::vector<T>& local_vals, int send_rank, int tag, MPI_Request& request) const
{
  MPI_Status state;
  int size = -1;
  int err = MPI_Recv(&size, 1, MPI_INT, send_rank, tag, this->comm_, &state);
  
  //std::cout << this->rank_ << " :: " <<  send_rank << " : " << size << " " << err 
  //        << " " << state.MPI_SOURCE << " " << state.MPI_TAG << " " << state.MPI_ERROR << std::endl;
  assert (err == 0);
  if (size >= 0)
  {
    local_vals.clear();
    local_vals.resize(size);
  
    int err = MPI_Irecv(&local_vals[0], size, this->get_mpi_type(local_vals[0]), send_rank, 10000+tag, this->comm_, &request);
    return err;
  }
  return -1;
}
*/

template <typename T>
int ParCom::Irecv(std::vector<T>& local_vals, int size, int send_rank, int tag, MPI_Request& request) const
{
  assert (size >= 0);
  local_vals.clear();
  local_vals.resize(size);

  int err = MPI_Irecv(&local_vals[0], size, this->get_mpi_type(local_vals[0]), send_rank, tag, this->comm_, &request);
  return err;
}

template <typename T>
int ParCom::Irecv(T* local_vals, int size, int send_rank, int tag, MPI_Request& request) const
{
  assert (size >= 0);
  
  int err = MPI_Irecv(local_vals, size, this->get_mpi_type(local_vals[0]), send_rank, tag, this->comm_, &request);
  return err;
}

template <typename T>
int ParCom::allgather(const T& local_val, std::vector<T>& recv_vals) const
{
  if (recv_vals.size() != this->size_)
  {
    recv_vals.resize(this->size_);
  }
  recv_vals[this->rank_] = local_val;
  int err = MPI_Allgather(&local_val, 1, this->get_mpi_type(local_val), &recv_vals[0], 1, this->get_mpi_type(local_val), this->comm_);
  return err;
}

  


} // namespace hiflow

#endif
