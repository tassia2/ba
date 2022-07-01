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

/// @author Philipp Gerstner

#include "common/parcom.h"
#include "mesh/mesh.h"
#include "mesh/attributes.h"
#include "mesh/iterator.h"
#include "mesh/types.h"

namespace hiflow {

template <>
MPI_Datatype ParCom::get_mpi_type(const float& dummy) const
{
  return MPI_FLOAT;
}

template <>
MPI_Datatype ParCom::get_mpi_type(const double& dummy) const
{
  return MPI_DOUBLE;
}

template <>
MPI_Datatype ParCom::get_mpi_type(const int& dummy) const
{
  return MPI_INT;
}

template <>
MPI_Datatype ParCom::get_mpi_type(const size_t& dummy) const
{
  return MPI_UNSIGNED_LONG;
}

void ParCom::determine_neighbor_domains(mesh::ConstMeshPtr mesh) const
{
  this->neighbor_domains_.clear();
     
  // determine cells to be requested from other procs
  mesh::AttributePtr sub = mesh->get_attribute("_sub_domain_", mesh->tdim());
  mesh::AttributePtr remote = mesh->get_attribute("_remote_index_", mesh->tdim());
  mesh::EntityIterator e_it = mesh->end(mesh->tdim());
  for (mesh::EntityIterator it = mesh->begin(mesh->tdim()); it != e_it; ++it) 
  {
    const int remote_index = remote->get_int_value(it->index());
    const int sub_domain = sub->get_int_value(it->index());
  
    if (remote_index >= 0)
    {
      this->neighbor_domains_.find_insert(sub_domain);
    }
  }
}

int ParCom::global_and(bool local_val, bool& global_val) const
{
  int local_int = static_cast<int>(local_val);
  int result = 0;
  int err = this->sum(local_int, result);

  if (result == this->size())
  {
    global_val = true;
  } 
  else 
  {
    global_val = false;
  }
  return err;
}

int ParCom::global_or(bool local_val, bool& global_val) const
{
  int local_int = static_cast<int>(local_val);
  int result = 0;
  int err = this->sum(local_int, result);

  if (result > 0)
  {
    global_val = true;
  } 
  else 
  {
    global_val = false;
  }
  return err;
}

int ParCom::global_xor(bool local_val, bool& global_val) const
{
  int local_int = static_cast<int>(local_val);
  int result = 0;
  int err = this->sum(local_int, result);

  if (result == 1)
  {
    global_val = true;
  } 
  else 
  {
    global_val = false;
  }
  return err;
}
} // namespace hiflow
