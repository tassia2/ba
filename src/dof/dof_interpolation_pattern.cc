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

#include "dof_interpolation_pattern.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <string>

#include "common/macros.h"

/// \author Michael Schick<br>Martin Baumann

namespace hiflow {
namespace doffem {

template<class DataType>
void DofInterpolationPattern<DataType>::set_number_slaves(int i) {
  num_slaves_ = i;
  ic_slave_.resize(i);
}

template<class DataType>
void DofInterpolationPattern<DataType>::insert_interpolation_master(
    std::pair< int, std::vector< std::pair< int, DataType > > > interpolation) {
  assert(!interpolation.second.empty());

  if ((interpolation.second.size() == 1) &&
      (std::abs(interpolation.second[0].second - 1.) <= 1.e-10)) {
    ic_master_.insert_dof_identification(interpolation.first,
                                         interpolation.second[0].first,
                                         1.);
    return;
  }
  ic_master_.push_back(interpolation);
  return;

  quit_program();
}

template<class DataType>
void DofInterpolationPattern<DataType>::insert_interpolation_slave(
    int s,
    std::pair< int, std::vector< std::pair< int, DataType > > > interpolation) 
{
  assert(num_slaves_ > 0);
  
  assert(!interpolation.second.empty());
  /*if (interpolation.second.empty())
  {
    return;
  }*/
  
  // TODO: assert necessary?
  if (interpolation.second.size() == 1) 
  {
/*
    assert(std::abs(interpolation.second[0].second - 1.) <= 1.e-10 ||
           std::abs(interpolation.second[0].second + 1.) <= 1.e-10 );
*/
    // beta = c * alpha 
    ic_slave_[s].insert_dof_identification(interpolation.first,    // dof to be interpolated beta
                                           interpolation.second[0].first, // interpolating dof alpha
                                           interpolation.second[0].second); // interpolation factor c
                                           
  } 
  else 
  {
    ic_slave_[s].push_back(interpolation);
  }
}

template<class DataType>
std::ostream &operator<<(std::ostream &s, const DofInterpolationPattern<DataType> &ic) {
  s << "DofInterpolationPattern" << std::endl;
  s << "=======================" << std::endl;

  // Interpolation information of master

  typename DofInterpolation<DataType>::const_iterator first_ic;
  typename DofInterpolation<DataType>::const_iterator last_ic;

  first_ic = ic.ic_master_.begin();
  last_ic = ic.ic_master_.end();

  if (first_ic != last_ic) {
    s << "  Interpolation of master by master:" << std::endl;
  }

  while (first_ic != last_ic) {
    s << "\t" << (*first_ic).first << " -> ";
    typename std::vector< std::pair< int, DataType > >::const_iterator itv;
    itv = ((*first_ic).second).begin();

    while (itv != ((*first_ic).second).end()) {
      s << "\t(" << (*itv).first << ", " << (*itv).second << ")";
      ++itv;
    }
    s << std::endl;

    ++first_ic;
  }

  //   // Identification information of master
  //
  //   s << "  Identification of master:" << std::endl;
  //   std::vector<std::pair<int,int> >::const_iterator it =
  //   ic.ic_master_.dof_identification_list().begin(); while(it !=
  //   ic.ic_master_.dof_identification_list().end())
  //   {
  //     s << "\t" << it->first << "\t <->\t " << it->second << std::endl;
  //     ++it;
  //   }

  // Slaves

  for (size_t slave = 0; slave != ic.ic_slave_.size(); ++slave) {
    // Interpolation information of slave

    typename DofInterpolation<DataType>::const_iterator first_ic;
    typename DofInterpolation<DataType>::const_iterator last_ic;

    first_ic = ic.ic_slave_[slave].begin();
    last_ic = ic.ic_slave_[slave].end();

    if (first_ic != last_ic) {
      s << "  Interpolation of slave " << slave << " by master:" << std::endl;
    }

    while (first_ic != last_ic) {
      s << "\t" << (*first_ic).first << " -> ";
      typename std::vector< std::pair< int, DataType > >::const_iterator itv;
      itv = ((*first_ic).second).begin();

      while (itv != ((*first_ic).second).end()) {
        s << "\t(" << (*itv).first << ", " << (*itv).second << ")";
        ++itv;
      }
      s << std::endl;

      ++first_ic;
    }

    // Identification information of slave

    s << "  Identification of slave: " << slave << " by master:" << std::endl;
    std::vector< std::pair< int, int > >::const_iterator it =
        ic.ic_slave_[slave].dof_identification_list().begin();
    while (it != ic.ic_slave_[slave].dof_identification_list().end()) {
      s << "\t" << it->first << " <-> " << it->second << std::endl;
      ++it;
    }
  } // for (int slave=0; slave<ic_slave_.size(); ++slave)

  return s;
}

template class DofInterpolationPattern<double>;
template class DofInterpolationPattern<float>;

template std::ostream &operator<< <float> (std::ostream &s, const DofInterpolationPattern<float> &ic);
template std::ostream &operator<< <double> (std::ostream &s, const DofInterpolationPattern<double> &ic);


} // namespace doffem
} // namespace hiflow
