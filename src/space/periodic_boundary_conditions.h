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

#ifndef _PERIODIC_BOUNDARY_CONDITIONS_H_
#define _PERIODIC_BOUNDARY_CONDITIONS_H_

#include <map>
#include <mpi.h>
#include <string>
#include <vector>

#include "common/macros.h"
#include "dof/dof_partition_local.h"
#include "dof/dof_fem_types.h"
#include "space/vector_space.h"

namespace hiflow {

///
/// @class PeriodicBoundaryConditions periodic_boundary_conditions.h
/// @brief Handling of periodic boundary conditions.
/// @author Martin Baumann
///

/**
Periodic boundary conditions are applied by identification of degrees of
freedom, hence no interpolation or iterative procedure is needed.
<H2>Example:</H2>
<img src="../images/periodic_boundary.png" alt="DoF identification in case of
periodic boundaries"> This class serves as abstract class, as the member
function compute_conditions() is pure abstract. This function calculates the
corresponding degrees of freedom. By the material number of boundaries, the
periodic boundary condition can be set. A doubly periodic domain can be set by
adding two boundary tuples, f.e.. The colours of the DoF points on the
boundaries indicate unique DoFs, i.e. the green DoFs on the left-hand side are
unified with the ones on the right-hand side. Due to the doubly periodicity
condition, the four blue DoFs are unified to one single DoF. \code VectorSpace
space;
  // ...
  PeriodicBoundaryConditionsCartesian periodic_boundaries;
  periodic_boundaries.add_boundary_tuple(10, 12);
  periodic_boundaries.add_boundary_tuple(11, 13);
  periodic_boundaries.apply_boundary_conditions(space);
  std::cout << periodic_boundaries << std::endl;
\endcode
\todo Should there be a base class for general boundary condition
      that can iterate over boundaries of the real domain?
 **/

template < class DataType, int DIM > class PeriodicBoundaryConditions {
public:
  typedef mesh::MaterialNumber MaterialNumber;

protected:
  /// maps a 'real' dof to vector of corresponding dofs, which will be
  /// eliminated
  std::map< doffem::DofID, std::vector< doffem::DofID > > corresponding_dof_;

  /// mapping (mat,var) -> dof_id
  std::map< std::pair< MaterialNumber, int >, std::vector< doffem::DofID > >
      dof_id_map_;

  /// find corresponding DoFs if there are more than one
  void handle_multiple_correspondences();

  /// description of boundary correspondence, f.e. mat 12 <-> mat 13
  std::map< MaterialNumber, MaterialNumber > boundary_descriptor_;

  /// returns true if entry was added; returns false if entry already existed.
  bool insert_dof_id_entry(MaterialNumber mat, int var, doffem::DofID dof_id);

  /// fills the list of corresponding DoFs (corresponding_dof_)
  virtual void compute_conditions(VectorSpace< DataType, DIM > &) = 0;

  /// permutes DoFs such that number of DoFs is reduced
  void change_dofs(VectorSpace< DataType, DIM > &space);

public:
  PeriodicBoundaryConditions() {}

  virtual ~PeriodicBoundaryConditions() {}

  /// add boundary tuple to boundary_descriptor_
  void add_boundary_tuple(MaterialNumber, MaterialNumber);

  /// calculates corresponding DoFs and performs DoF identification
  void apply_boundary_conditions(VectorSpace< DataType, DIM > &);

  /// checks if a boundary has periodic boundary condition
  bool is_periodic_boundary(MaterialNumber) const;
};

template < class DataType, int DIM >
void PeriodicBoundaryConditions< DataType, DIM >::handle_multiple_correspondences() {
  bool debug_print = false;

  std::map< doffem::DofID, std::vector< doffem::DofID > >::iterator it;
  it = corresponding_dof_.begin();
  while (it != corresponding_dof_.end()) {
    std::map< doffem::DofID, std::vector< doffem::DofID > >::iterator it2;
    it2 = corresponding_dof_.begin();
    while (it2 != corresponding_dof_.end()) {
      if (it2 != it) {
        bool remove_it2 = false;

        // key value of it2 matches key value of it
        if (it2->first == it->first) {
          for (size_t i = 0; i < it2->second.size(); ++i) {
            it->second.push_back(it2->second[i]);
          }
          remove_it2 = true;
          if (debug_print) {
            std::cout << "MODE1: " << it->first << "\t" << it2->first
                      << std::endl;
          }
        }

        // one of the values in the vector of it2 matches key value of it
        bool found_matching = false;
        for (size_t i = 0; i < it2->second.size(); ++i) {
          if (it2->second[i] == it->first) {
            found_matching = true;
            break;
          }
          if (debug_print) {
            std::cout << "MODE2: " << it->first << "\t" << it2->second.at(0)
                      << ", SIZE = " << it2->second.size() << std::endl;
          }
        }
        if (found_matching) {
          it->second.push_back(it2->first);
          for (size_t i = 0; i < it2->second.size(); ++i) {
            if (it2->second[i] != it->first) {
              it->second.push_back(it2->second[i]);
            }
          }
          remove_it2 = true;
        }

        // key value of it2 matches one of the values in it
        for (size_t i = 0; i < it->second.size(); ++i) {
          if (it2->first == it->second[i]) {
            for (size_t j = 0; j < it2->second.size(); ++j) {
              it->second.push_back(it2->second[j]);
            }
            remove_it2 = true;
            if (debug_print) {
              std::cout << "MODE3" << std::endl;
            }
            break;
          }
        }

        // value in it2 matches value in it
        for (size_t i = 0; i < it->second.size(); ++i) {
          bool found_matching = false;
          for (size_t j = 0; j < it2->second.size(); ++j) {
            if (it->second[i] == it2->second[j]) {
              found_matching = true;
            }
          }
          if (found_matching) {
            it->second.push_back(it2->first);
            for (size_t j = 0; j < it2->second.size(); ++j) {
              if (it2->second[j] != it->second[i]) {
                it->second.push_back(it2->second[j]);
              }
            }
            remove_it2 = true;
            if (debug_print) {
              std::cout << "MODE4" << std::endl;
            }
            break;
          }
        }

        // remove it2

        if (remove_it2) {
          corresponding_dof_.erase(it2);
        } else if (debug_print) {
          std::cout << "--------- NOTHING ------- " << std::endl;
        }
      }
      ++it2;
    }
    ++it;
  }

  // cleaning

  it = corresponding_dof_.begin();
  while (it != corresponding_dof_.end()) {
    // erase entry if DoF in vector is same as DoF in key
    std::vector< doffem::DofID >::iterator values;
    for (values = it->second.begin(); values != it->second.end(); ++values) {
      if (*values == it->first) {
        values = it->second.erase(values);
      }
    }

    // erase DoF in vector if it occurs multiply
    for (values = it->second.begin(); values != it->second.end(); ++values) {
      // first sort vector than erase double entries
      sort(it->second.begin(), it->second.end());
      std::vector< doffem::DofID >::iterator last_unique;
      last_unique = std::unique(it->second.begin(), it->second.end());
      it->second.resize(last_unique - it->second.begin());
    }

    ++it;
  }

  // print correspondence list

  if (debug_print) {
    std::cout << "PeriodicBoundary Correspondence List" << std::endl;
    std::map< doffem::DofID, std::vector< doffem::DofID > >::const_iterator it;
    for (it = corresponding_dof_.begin(); it != corresponding_dof_.end();
         ++it) {
      std::cout << "  " << it->first << "\t <-> \t";
      std::vector< doffem::DofID >::const_iterator itc;
      for (itc = it->second.begin(); itc != it->second.end(); ++itc) {
        std::cout << *itc << "\t";
      }
      std::cout << std::endl;
    }
  }
}

template < class DataType, int DIM >
bool PeriodicBoundaryConditions< DataType, DIM >::insert_dof_id_entry(
    MaterialNumber mat, int var, doffem::DofID dof_id) {
  std::map< std::pair< MaterialNumber, int >,
            std::vector< doffem::DofID > >::iterator it;
  it = dof_id_map_.find(std::make_pair(mat, var));

  // insert new map entry for tuple (mat,var) if none existing
  if (it == dof_id_map_.end()) {
    std::vector< doffem::DofID > ids;
    dof_id_map_.insert(std::make_pair(std::make_pair(mat, var), ids));
    it = dof_id_map_.find(std::make_pair(mat, var));
  }

  // is dof_id already in vector of DoFs?
  std::vector< doffem::DofID > &dof_vector = it->second;
  if (find(dof_vector.begin(), dof_vector.end(), dof_id) == dof_vector.end()) {
    dof_id_map_.find(std::make_pair(mat, var))->second.push_back(dof_id);
    return true;
  }

  // std::cout << "ALREADY INSERTED: " << mat << "\t" << var << "\t" << dof_id
  // << std::endl;

  return false;
}

template < class DataType, int DIM >
void PeriodicBoundaryConditions< DataType, DIM >::apply_boundary_conditions(
    VectorSpace< DataType, DIM > &space) {
  // calculates corresponding DoFs
  compute_conditions(space);

  // adapt DoFs
  change_dofs(space);
}

template < class DataType, int DIM >
void PeriodicBoundaryConditions< DataType, DIM >::add_boundary_tuple(
    MaterialNumber a, MaterialNumber b) {
  boundary_descriptor_.insert(std::make_pair(a, b));
  boundary_descriptor_.insert(std::make_pair(b, a));
}

/// there are boundary conditions for a boundary with material number 'mat' if
/// there is an entry in boundary_descriptor_

template < class DataType, int DIM >
bool PeriodicBoundaryConditions< DataType, DIM >::is_periodic_boundary(
    MaterialNumber mat) const {
  std::map< MaterialNumber, MaterialNumber >::const_iterator it;
  it = boundary_descriptor_.find(mat);
  return (it != boundary_descriptor_.end());
}

template < class DataType, int DIM >
void PeriodicBoundaryConditions< DataType, DIM >::change_dofs(
    VectorSpace< DataType, DIM > &space) {
  // Calculate permutation: Blind DoFs should be replaced by real DoFs

  std::vector< int > perm;
  perm.resize(space.dof().get_nb_dofs());
  for (size_t i = 0; i < perm.size(); ++i) {
    perm[i] = i;
  }

  std::map< doffem::DofID, std::vector< doffem::DofID > > &cd =
      corresponding_dof_;
  std::map< doffem::DofID, std::vector< doffem::DofID > >::const_iterator
      cd_it = cd.begin();
  while (cd_it != cd.end()) {
    // cd.first  -> master dof
    // cd.second -> vector of slave dofs
    for (size_t slave = 0; slave < cd_it->second.size(); ++slave) {
      assert(cd_it->second.at(slave) < perm.size());
      perm[cd_it->second[slave]] = cd_it->first;
    }
    ++cd_it;
  }

  // Make permutation consecutive

  std::vector< bool > is_used(space.dof().get_nb_dofs(), true);
  cd_it = cd.begin();
  while (cd_it != cd.end()) {
    for (size_t slave = 0; slave < cd_it->second.size(); ++slave) {
      is_used[cd_it->second[slave]] = false;
    }
    ++cd_it;
  }
  std::vector< int > consecutive_perm;
  consecutive_perm.resize(is_used.size(), -1);
  int counter = -1;
  for (size_t i = 0; i < is_used.size(); ++i) {
    if (is_used[i]) {
      ++counter;
    }
    consecutive_perm[i] = counter;
  }

  // Mix permutations together

  for (size_t i = 0; i < perm.size(); ++i) {
    perm[i] = consecutive_perm[perm[i]];
  }

  // Apply permutation

  space.dof().apply_permutation(perm, "PeriodicBoundaryConditions");
}

template < class DataType, int DIM >
std::ostream &operator<<(std::ostream &s,
                         const PeriodicBoundaryConditions< DataType, DIM > &cond) {
  s << "PeriodicBoundaryConditions" << std::endl;
  s << "  Corresponding Boundaries:" << std::endl;
  typename std::map<
      typename PeriodicBoundaryConditions< DataType, DIM >::MaterialNumber,
      typename PeriodicBoundaryConditions< DataType, DIM >::MaterialNumber >::
      const_iterator it2;
  it2 = cond.boundary_descriptor_.begin();
  while (it2 != cond.boundary_descriptor_.end()) {
    s << "    " << it2->first << " <-> " << it2->second << std::endl;
    ++it2;
  }
  s << "  Boundary DoFs on (mat, var):" << std::endl;
  typename std::map< std::pair< typename PeriodicBoundaryConditions<
                                    DataType, DIM >::MaterialNumber, int >,
                     std::vector< doffem::DofID > >::const_iterator it =
      cond.dof_id_map_.begin();
  while (it != cond.dof_id_map_.end()) {
    s << "    (" << it->first.first << ", " << it->first.second
      << "): " << it->second.size() << std::endl;
    ++it;
  }
  s << "  # corresponding DoFs: " << cond.corresponding_dof_.size()
    << std::endl;

  return s;
}

} // namespace hiflow

#endif
