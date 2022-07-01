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

#include "dof_interpolation.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <list>
#include <sstream>
#include <stack>
#include <string>

#include "common/log.h"
#include "common/macros.h"
#include "mpi.h"

/// \author Michael Schick<br>Martin Baumann

namespace hiflow {
namespace doffem {

template<class DataType>
bool DofInterpolation<DataType>::push_back(DofID id, std::vector< DofID > const &dofs,
                                 std::vector< DataType > const &weights) {
  // checks

  assert(dofs.size() == weights.size());

  // prepare data vector and key

  std::vector< std::pair< int, DataType > > data;
  data.reserve(dofs.size());

  for (size_t i = 0, e_i = dofs.size(); i != e_i; ++i) {
    data.push_back(std::make_pair(dofs[i], weights[i]));
  }

  DofID key = id;

  // don't save identification description as interpolation

  if (dofs.size() == 1) {
    std::cout
        << "We don't want to handle corresponding DoFs by an interpolation"
        << " description, do we?" << std::endl;
    quit_program();

    // weight should be 1.0
    assert(std::abs(data[0].second - 1.0) < 1.0e-14);
    data[0].second = 1.0;

    // if identification already exists return
    if (this->find(key) != this->end()) {
      assert(this->find(key)->second.size() == 1);
      return false;
    }
  }

  // insert data to container

  bool status;
  status = (this->insert(std::pair< DofID, std::vector< std::pair< DofID, DataType > > >( id, data))).second;

  return status;
}

template<class DataType>
bool DofInterpolation<DataType>::push_back(
    std::pair< DofID, std::vector< std::pair< DofID, DataType > > >
        interpolation) {
  // don't save identification description as interpolation

  if (interpolation.second.size() == 1) {
    std::cout
        << "We don't want to handle corresponding DoFs by an interpolation"
        << " description, do we?" << std::endl;
    quit_program();
  }

  // insert data to container

  bool status;
  status = (this->insert(interpolation)).second;

  return status;
}

/// vector perm defines the permutation by mapping i -> perm(i)
template<class DataType>
void DofInterpolation<DataType>::apply_permutation(const std::vector< DofID > &perm) {
  // backup and clear the container

  DofInterpolation<DataType> ic_old(*this);
  this->clear();

  // now fill ...

  const_iterator first = ic_old.begin();
  const_iterator last = ic_old.end();

  const_iterator tmp_first = ic_old.begin();
  std::vector< DofID > old_inter;
  while (tmp_first != last) {
    old_inter.push_back(tmp_first->first);
    ++tmp_first;
  }

  LOG_DEBUG(2, "Old DofInterpolation<DataType>: " << string_from_range(old_inter.begin(),
                                                             old_inter.end()));

  LOG_DEBUG(2, "Permutation: " << string_from_range(perm.begin(), perm.end()));
  while (first != last) {
    // New key
    assert(first->first < static_cast< int >(perm.size()) && first->first >= 0);
    DofID new_key = perm[static_cast< size_t >(first->first)];

    LOG_DEBUG(2, " permutation size " << perm.size() << " key " << new_key
                                      << " hanging dof global id "
                                      << first->first);

    assert(new_key >= 0);
    //         assert(new_key < static_cast<int>(perm.size()));

    // New data
    typedef typename std::vector< std::pair< DofID, DataType > > ContainerType;
    ContainerType new_data;
    typename ContainerType::const_iterator first_data_ic;
    typename ContainerType::const_iterator last_data_ic;

    first_data_ic = (first->second).begin();
    last_data_ic = (first->second).end();

    new_data.resize((first->second).size());

    while (first_data_ic != last_data_ic) {
      // get index in old data

      DofID k = first_data_ic - (first->second).begin();

      DofID ind_old = first_data_ic->first;
      DataType val = first_data_ic->second;

      assert((ind_old >= 0) && (ind_old < static_cast< int >(perm.size())));
      assert((k >= 0) && (k < static_cast< int >(new_data.size())));

      // fill new_data vector

      (new_data[k]).first = perm[ind_old];
      (new_data[k]).second = val;

      ++first_data_ic;
    }

    // Insert

    this->insert(std::pair< DofID, ContainerType >(new_key, new_data));

    // Next element

    ++first;
  }
  tmp_first = this->begin();
  std::vector< DofID > new_inter;
  while (tmp_first != this->end()) {
    new_inter.push_back(tmp_first->first);
    ++tmp_first;
  }

  LOG_DEBUG(2, "New DofInterpolation<DataType>: " << string_from_range(new_inter.begin(),
                                                             new_inter.end()));
}

template<class DataType>
void DofInterpolation<DataType>::clear_entries() {
  // clear interpolation information
  this->clear();

  // clear identification information
  dof_identification_list_.clear();
  dof_identification_factors_.clear();
}

template<class DataType>
void DofInterpolation<DataType>::backup(const std::string &file_name) const {
  // open

  std::ofstream file_stream(file_name.c_str());
  assert(file_stream);

  // backup

  backup(file_stream);

  // close

  file_stream.close();
}

template<class DataType>
void DofInterpolation<DataType>::backup(std::ostream &os) const {
  os.precision(22);

  os << "# DofInterpolation<DataType>" << std::endl;
  os << "# -------------------------" << std::endl;
  os << "# number of entries: " << this->size() << std::endl;
  os << "# " << std::endl;

  const_iterator first = this->begin();
  const_iterator last = this->end();

  while (first != last) {
    os << first->first << std::endl;
    os << (first->second).size() << std::endl;

    for (size_t k = 0, e_k = (first->second).size(); k != e_k; ++k) {
      os << ((first->second)[k]).first << "\t" << ((first->second)[k]).second
         << std::endl;
    }
    ++first;
  }
}

template<class DataType>
void DofInterpolation<DataType>::restore(const std::string &file_name) {
  // clear

  this->clear();

  // open

  std::ifstream file_stream(file_name.c_str());
  assert(file_stream);

  // backup

  restore(file_stream);

  // close

  file_stream.close();
}

template<class DataType>
void DofInterpolation<DataType>::restore(std::istream &is) {
  // clear

  this->clear();

  // check for validity and reach the data block

  std::string str_line;
  bool file_is_valid = false;

  while ((str_line.empty()) || (str_line[0] != '#')) {
    if (str_line.find("# DofInterpolation<DataType>") != std::string::npos) {
      file_is_valid = true;
    }

    interminable_assert(!is.eof());
    getline(is, str_line);
  }

  int nb_elem;

  assert(file_is_valid);
  assert(!is.eof());
  getline(is, str_line);
  std::istringstream ist_1(str_line);
  ist_1 >> nb_elem;

  for (int k = 0; k != nb_elem; ++k) {
    // Load key

    DofID key;
    assert(!is.eof());
    getline(is, str_line);
    std::istringstream ist_2(str_line);
    ist_2 >> key;

    // Load vector

    std::vector< std::pair< DofID, DataType > > data;
    int data_size;
    assert(!is.eof());
    getline(is, str_line);
    std::istringstream ist_3(str_line);
    ist_3 >> data_size;

    data.resize(data_size);

    for (int k1 = 0; k1 != data_size; ++k1) {
      assert(!is.eof());
      getline(is, str_line);
      std::istringstream ist_4(str_line);

      DofID index;
      DataType weight;

      ist_4 >> index >> weight;

      data[k1] = std::pair< DofID, DataType >(index, weight);
    }

    // add element

    assert(this->find(key) == this->end());

    bool status;
    status = (this->insert(std::pair< DofID, std::vector< std::pair< DofID, DataType > > >(key, data))).second;
    assert(status);
  }
}

template<class DataType>
bool DofInterpolation<DataType>::is_constrained(DofID node) const {
  return this->find(node) != this->end();
}

template<class DataType>
struct CmpFirst {

  CmpFirst(DofID id) : id_(id) {}

  bool operator()(const std::pair< DofID, DataType > &p) const {
    return p.first == id_;
  }

private:
  DofID id_;
};

/// Adds a dependency from node to dep with the given weight
template<class DataType>
void DofInterpolation<DataType>::add_dependency(DofID node, DofID dep, DataType weight) {
  iterator it = this->find(node);
  if (it == this->end()) 
  {
    // If node does not already exist ->
    push_back(node, std::vector< DofID >(1, dep), std::vector< DataType >(1, weight));
  } 
  else 
  {
    typedef typename std::vector< std::pair< DofID, DataType > >::iterator DependencyIterator;
    DependencyIterator dep_it = std::find_if(it->second.begin(), it->second.end(), CmpFirst<DataType>(dep));
    if (dep_it != it->second.end()) 
    {
      dep_it->second += weight;
    } 
    else 
    {
      it->second.push_back(std::make_pair(dep, weight));
    }
  }
}

template<class DataType>
void DofInterpolation<DataType>::remove_dependency(DofID node, DofID dep) {
  iterator it = this->find(node);
  if (it != this->end()) 
  {
    typedef typename std::vector< std::pair< DofID, DataType > >::iterator DependencyIterator;
    DependencyIterator dep_it = std::find_if(it->second.begin(), it->second.end(), CmpFirst<DataType>(dep));
    if (dep_it != it->second.end()) 
    {
      it->second.erase(dep_it);
    }
  }
}

template<class DataType>
bool DofInterpolation<DataType>::has_dependency(DofID node, DofID dep) const {
  const_iterator it = this->find(node);
  if (it != this->end()) 
  {
    typedef typename std::vector< std::pair< DofID, DataType > >::const_iterator DependencyIterator;
    DependencyIterator dep_it = find_if(it->second.begin(), it->second.end(), CmpFirst<DataType>(dep));
    return dep_it != it->second.end();
  }
  return false;
}

template<class DataType>
DataType DofInterpolation<DataType>::get_weight(DofID node, DofID dep) const {
  const_iterator it = this->find(node);
  if (it != this->end()) {
    typedef typename std::vector< std::pair< DofID, DataType > >::const_iterator
        DependencyIterator;
    DependencyIterator dep_it =
        find_if(it->second.begin(), it->second.end(), CmpFirst<DataType>(dep));
    if (dep_it != it->second.end()) {
      return dep_it->second;
    }
  }
  return 0.;
}

template<class DataType>
DataType DofInterpolation<DataType>::sum_weights(DofID node) const {
  DataType s = 0.;
  const_iterator it = this->find(node);
  if (it != this->end()) {
    typedef typename std::vector< std::pair< DofID, DataType > >::const_iterator DependencyIterator;
    for (DependencyIterator dep_it = it->second.begin(),
                            e_dep_it = it->second.end();
         dep_it != e_dep_it; ++dep_it) 
    {
      s += dep_it->second;
    }
  }
  return s;
}

template<class DataType>
void DofInterpolation<DataType>::reduce_from_above() {
  typedef typename std::vector< std::pair< DofID, DataType > >::iterator
      DependencyIterator;
  LOG_DEBUG(2, "Interpolation before reducing from above:\n" << *this);

  std::vector< DofID > resolved_deps;

  // loop over interpolation entries from highest to lowest
  for (reverse_iterator it = this->rbegin(), it_end = this->rend(); it != it_end; ++it) {
    // stack to store constrained dofs during search
    std::stack< std::pair< DofID, DataType > > s;
    const DofID root = it->first;

    LOG_DEBUG(3, "Root = " << root);

    resolved_deps.clear();

    // push all higher-numbered constrained dependencies of root on stack
    for (DependencyIterator d_it = it->second.begin(), d_end = it->second.end();
         d_it != d_end; ++d_it) {
      assert(root != d_it->first); // should not depend on itself at beginning
      if (d_it->first > root && is_constrained(d_it->first)) {
        s.push(*d_it);

        // store to remove afterwards
        resolved_deps.push_back(d_it->first);
      }
    }

    LOG_DEBUG(3, "Deps = " << string_from_range(resolved_deps.begin(),
                                                resolved_deps.end()));

    // remove dependencies that will be treated below
    for (std::vector< DofID >::const_iterator d_it = resolved_deps.begin(),
                                              e_d_it = resolved_deps.end();
         d_it != e_d_it; ++d_it) {
      remove_dependency(root, *d_it);
    }

    // invariant: s contains higher numbered
    // dependencies on constrained dofs that must still be
    // treated.
    while (!s.empty()) {
      const std::pair< DofID, DataType > curr = s.top();
      s.pop();

      LOG_DEBUG(3, "Treating dep " << curr.first);

      iterator curr_it = this->find(curr.first);
      assert(curr_it != this->end());
      for (DependencyIterator d_it = curr_it->second.begin(),
                              d_end = curr_it->second.end();
           d_it != d_end; ++d_it) {
        const DofID dep_id = d_it->first;
        if (dep_id <= root || !is_constrained(dep_id)) {
          // add weight to existing dependent dof in root, or create new entry
          // if needed
          add_dependency(root, dep_id, d_it->second * curr.second);
        } else {
          s.push(std::make_pair(dep_id, d_it->second * curr.second));
        }
      }

      remove_dependency(root, curr.first);

      LOG_DEBUG(3, "New interpolation:\n" << *this);
    }

    if (has_dependency(root, root)) {
      LOG_DEBUG(3, "Removing self-dependency");
      iterator root_it = this->find(root);
      const DataType weight = 1. - get_weight(root, root);

      // check that we did not eliminate the dof (can this happen?)
      if (std::abs(weight) < 1.e-9) {
        std::cerr << "Circular dependency with weight " << weight << "!\n";
        quit_program();
      }

      // eliminate root - root dependency
      for (DependencyIterator d_it = root_it->second.begin(),
                              d_end = root_it->second.end();
           d_it != d_end; ++d_it) {
        d_it->second /= weight;
      }

      remove_dependency(root, root);
      LOG_DEBUG(3, "New interpolation:\n" << *this);
    }
  }
}

template<class DataType>
void DofInterpolation<DataType>::reduce_from_below() {
  typedef typename std::vector< std::pair< DofID, DataType > >::iterator
      DependencyIterator;
  LOG_DEBUG(2, "Interpolation before reducing from below:\n" << *this);

  std::vector< DofID > resolved_deps;

  // loop over interpolation entries from lowest to highest
  for (iterator it = this->begin(), it_end = this->end(); it != it_end; ++it) {

    // stack to store constrained dofs during search
    std::stack< std::pair< DofID, DataType > > s;
    const DofID root = it->first;

    LOG_DEBUG(3, "Root = " << root);

    resolved_deps.clear();

    // push all constrained dependencies of root on stack
    for (DependencyIterator d_it = it->second.begin(), d_end = it->second.end();
         d_it != d_end; ++d_it) {
      if (is_constrained(d_it->first)) {
        // constrained dependent dofs should now be strictly higher than root
        assert(d_it->first < root);
        s.push(*d_it);

        // store to remove afterwards
        resolved_deps.push_back(d_it->first);
      }
    }
    LOG_DEBUG(3, "Deps = " << string_from_range(resolved_deps.begin(),
                                                resolved_deps.end()));

    for (std::vector< DofID >::const_iterator d_it = resolved_deps.begin(),
                                              e_d_it = resolved_deps.end();
         d_it != e_d_it; ++d_it) {
      remove_dependency(root, *d_it);
    }

    // invariant: s contains dependencies on constrained dofs
    // (all lower numbered) that must still be treated.
    while (!s.empty()) {
      const std::pair< DofID, DataType > curr = s.top();
      s.pop();

      LOG_DEBUG(3, "Treating dep " << curr.first);

      iterator curr_it = this->find(curr.first);
      assert(curr_it != this->end());
      for (DependencyIterator d_it = curr_it->second.begin(),
                              d_end = curr_it->second.end();
           d_it != d_end; ++d_it) {
        const DofID dep_id = d_it->first;
        assert(dep_id < root);
        if (!is_constrained(dep_id)) {
          // add weight to existing dependent dof in root, or create new entry
          // if needed
          add_dependency(root, dep_id, d_it->second * curr.second);
        } else {
          s.push(std::make_pair(dep_id, d_it->second * curr.second));
        }
      }

      remove_dependency(root, curr.first);

      LOG_DEBUG(3, "New interpolation:\n" << *this);
    }

    // there should be no self-dependency
    assert(!has_dependency(root, root));
  }
}

template<class DataType>
void DofInterpolation<DataType>::reduce() {

  // typedef std::vector< std::pair<DofID,DataType> >::iterator
  // DependencyIterator;

  // The reduction of dofs is performed in two phases, to avoid
  // problems with circular dependencies.

  // 1. Reduce the constraints to a form where each constrained
  // dof only depends on unconstrained dofs and constrained dofs
  // with strictly lower id.
  reduce_from_above();

  // 2. Given the form of the constraints above, reduce the
  // constraints to a form where each constrained dof only
  // depends on unconstrained dofs.
  reduce_from_below();

  LOG_DEBUG(2, "Final reduced interpolation:\n" << *this);
}

template<class DataType>
void DofInterpolation<DataType>::sort() {
  std::sort(this->dof_identification_list_.begin(),
            this->dof_identification_list_.end());
            
  std::sort(this->dof_identification_factors_.begin(),
            this->dof_identification_factors_.end());
}

template<class DataType>
std::ostream &operator<<(std::ostream &s, const DofInterpolation<DataType> &ic) {
  // Interpolation information

  typename DofInterpolation<DataType>::const_iterator first_ic;
  typename DofInterpolation<DataType>::const_iterator last_ic;

  first_ic = ic.begin();
  last_ic = ic.end();

  while (first_ic != last_ic) {
    s << "\t" << (*first_ic).first << " ->";
    typename std::vector< std::pair< DofID, DataType > >::const_iterator itv;
    itv = ((*first_ic).second).begin();

    while (itv != ((*first_ic).second).end()) {
      s << "\t (" << (*itv).first << " " << (*itv).second << ")";
      ++itv;
    }
    s << std::endl;

    ++first_ic;
  }

  // Identification information

  std::vector< std::pair< DofID, DofID > >::const_iterator it =
      ic.dof_identification_list().begin();
  while (it != ic.dof_identification_list().end()) {
    s << "\t" << it->first << "\t <->\t " << it->second << std::endl;
    ++it;
  }

  return s;
}

template class DofInterpolation<double>;
template class DofInterpolation<float>;

} // namespace doffem
} // namespace hiflow
