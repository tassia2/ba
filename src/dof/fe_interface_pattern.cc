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

#include "fe_interface_pattern.h"
#include "fem/fe_reference.h"

namespace hiflow {
namespace doffem {


template < class DataType, int DIM >
FEInterfacePattern< DataType, DIM >::FEInterfacePattern(
    mesh::InterfacePattern interface_pattern,
    CRefElementSPtr< DataType, DIM > fe_type_master,
    std::vector< CRefElementSPtr< DataType, DIM > > fe_type_slaves) 
{
  interface_pattern_ = interface_pattern;
  fe_type_master_ = fe_type_master;
  fe_type_slaves_ = fe_type_slaves;
}

template < class DataType, int DIM >
FEInterfacePattern< DataType, DIM >::~FEInterfacePattern() {}

template < class DataType, int DIM >
int FEInterfacePattern< DataType, DIM >::get_interface_degree(int *which_slave) const 
{
  int min_degree = fe_type_master_->max_deg();

  if (which_slave != 0) {
    *which_slave = -1;
  }

  for (size_t i = 0, end = num_slaves(); i != end; ++i) {
    const int deg = fe_type_slaves_[i]->max_deg();
    if (deg < min_degree) {
      min_degree = deg;
      if (which_slave != 0) {
        *which_slave = i;
      }
    }
  }

  return min_degree;
}

template < class DataType, int DIM >
bool FEInterfacePattern< DataType, DIM >::operator==(const FEInterfacePattern< DataType, DIM > &test) 
{
  bool tmp  = (interface_pattern_ == test.interface_pattern_) &&
              ( (*fe_type_master_.get()) == (*test.fe_type_master_.get()) );
  if (!tmp)
  {
    return false;
  }

  if (fe_type_slaves_.size() != test.fe_type_slaves_.size())
  {
    return false;
  }
  for (size_t i=0, e = fe_type_slaves_.size(); i!=e; ++i )
  {
    if ( (*fe_type_slaves_[i].get()) != (*test.fe_type_slaves_[i].get()))
    {
      return false;
    }
  }
  return false;
}

/// first check InterfacePattern, as FEInterfacePattern is a specialization

template < class DataType, int DIM >
bool FEInterfacePattern< DataType, DIM >::operator<(const FEInterfacePattern< DataType, DIM > &test) const 
{
  if (interface_pattern_ < test.interface_pattern_) 
  {
    return true;
  } 
  else if (interface_pattern_ == test.interface_pattern_) 
  {
    if ((*fe_type_master_.get()) < (*test.fe_type_master_.get())) 
    {
      return true;
    } 
    else if ((*fe_type_master_.get()) == (*test.fe_type_master_.get())) 
    {
      const size_t my_size = fe_type_slaves_.size();
      const size_t test_size = test.fe_type_slaves_.size();
      if (my_size < test_size)
      {
        return true;
      }
      else if (my_size == test_size)
      {
        for (size_t i = 0; i != my_size; ++i)
        {
          if ((*fe_type_slaves_[i].get()) < (*test.fe_type_slaves_[i].get()))
          {
            return true;
          } 
          else if ((*fe_type_slaves_[i].get()) == (*test.fe_type_slaves_[i].get()))
          {
            continue;
          }
          return false;
        }
      }
    }
  }
  return false;
}

template < class DataType, int DIM >
std::ostream &operator<<(std::ostream &s, const FEInterfacePattern< DataType, DIM > &pattern) 
{
  s << pattern.interface_pattern();
  s << "Ansatz Master: " << pattern.fe_type_master()->name() << std::endl;
  if (pattern.num_slaves() == 1) 
  {
    s << "Ansatz Slave:  " << pattern.fe_type_slaves()[0]->name()
      << std::endl;
  } 
  else 
  {
    for (size_t i = 0, e_i = pattern.num_slaves(); i != e_i; ++i) 
    {
      s << "Ansatz Slave " << i << ": "
        << pattern.fe_type_slaves()[i]->name() << std::endl;
    }
  }
  return s;
}

template class FEInterfacePattern<float, 3>;
template class FEInterfacePattern<float, 2>;
template class FEInterfacePattern<float, 1>;

template class FEInterfacePattern<double, 3>;
template class FEInterfacePattern<double, 2>;
template class FEInterfacePattern<double, 1>;

template std::ostream &operator<< <double, 1> (std::ostream &s, const FEInterfacePattern< double, 1 > &pattern);
template std::ostream &operator<< <double, 2> (std::ostream &s, const FEInterfacePattern< double, 2 > &pattern);
template std::ostream &operator<< <double, 3> (std::ostream &s, const FEInterfacePattern< double, 3 > &pattern);

template std::ostream &operator<< <float, 1> (std::ostream &s, const FEInterfacePattern< float, 1 > &pattern);
template std::ostream &operator<< <float, 2> (std::ostream &s, const FEInterfacePattern< float, 2 > &pattern);
template std::ostream &operator<< <float, 3> (std::ostream &s, const FEInterfacePattern< float, 3 > &pattern);
} // namespace doffem
} // namespace hiflow
