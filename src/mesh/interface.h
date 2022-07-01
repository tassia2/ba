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

/// \author Staffan Ronnas

#ifndef HIFLOW_MESH_INTERFACE_H
#define HIFLOW_MESH_INTERFACE_H

#include <iosfwd>
#include <vector>

#include "mesh/types.h"

namespace hiflow {
namespace mesh {

enum class InterfaceType 
{
  NOT_SET = -1,
  INTERIOR = 0,
  OUTER_BDY = 1,
  GHOST_BDY = 2
};

/// \brief Information about an interface between a group of cells in the mesh.

class Interface {
public:
  typedef std::vector< EntityNumber >::const_iterator const_iterator;

  /// \brief Constructor.
  /// \param master_index  cell index of the master cell.
  Interface(EntityNumber master_index, int master_facet_number)
   : master_index_(master_index), master_facet_number_(master_facet_number) 
   {}

  /// \return the cell index of the master cell.
  inline EntityNumber master_index() const
  { 
    return master_index_; 
  }

  /// \return the interface facet number in the master cell
  inline int master_facet_number() const
  { 
    return master_facet_number_; 
  }

  /// \return the interface facet number in slave cell i
  inline int slave_facet_number(int i) const
  {
    range_check(slave_facet_numbers_, i);
    return slave_facet_numbers_[i];
  }

  /// \return the number of slave cells of the Interface.
  inline int num_slaves() const
  { 
    return slave_indices_.size(); 
  }

  /// \param i  index of the slave cell.
  /// \pre 0 <= i < Interface::num_slaves() .
  /// \return the cell index of the i:th slave cell.
  inline EntityNumber slave_index(int i) const
  {
    range_check(slave_indices_, i);
    return slave_indices_[i];
  }

  /// \return iterator to the first slave cell index.
  inline const_iterator begin() const
  {
    return slave_indices_.begin();
  }

  /// \return iterator to the position just past the last slave cell index.
  inline const_iterator end() const
  {
    return slave_indices_.end();
  }

  /// \brief Add a slave cell.
  ///
  /// \param slave   cell index of the new slave cell.
  inline void add_slave(EntityNumber slave, int slave_facet_number)
  {
    slave_indices_.push_back(slave);
    slave_facet_numbers_.push_back(slave_facet_number);
  }

  inline InterfaceType type() const 
  {
    return this->type_;
  }

  inline InterfaceType& type() 
  {
    return this->type_;
  }

private:
  EntityNumber master_index_;
  int master_facet_number_;
  InterfaceType type_ = InterfaceType::NOT_SET;
  std::vector< EntityNumber > slave_indices_;
  std::vector< int > slave_facet_numbers_;
};

/// \brief Output of Interface objects.
std::ostream &operator<<(std::ostream &os, const Interface &interface);

/// \brief Container of Interface objects for a mesh.

class InterfaceList {
  
public:
  typedef std::vector< Interface >::const_iterator const_iterator;
  typedef std::vector< Interface >::iterator iterator;

  InterfaceList() { ; }

  /// \brief Constructor to be called from InterfaceList::create() .
  ///
  /// \param mesh  pointer to the mesh for which the InterfaceList will be
  /// computed.
  InterfaceList(ConstMeshPtr& mesh)
  : mesh_(mesh) 
  {}

  /// \brief Construct the InterfaceList for a mesh.
  ///
  /// \param mesh   pointer to the mesh.
  /// \return  InterfaceList object for mesh.
  static InterfaceList create(ConstMeshPtr mesh);

  void determine_interface_types(ConstMeshPtr mesh);

  /// \return iterator to the first Interface object.
  inline const_iterator begin() const
  {
    return interfaces_.begin();
  }

  /// \return iterator to the position just past the last Interface object.
  inline const_iterator end() const
  {
    return interfaces_.end();
  }

  /// \return pointer to the mesh to which the InterfaceList belongs.
  inline ConstMeshPtr mesh() const
  { 
    return mesh_; 
  }

  /// \return number of Interface objects in the container.
  inline int size() const
  { 
    return interfaces_.size(); 
  }

  inline void clear()
  { 
    this->interfaces_.clear();
  }

  inline void add_interface (const Interface& interface)
  {
    interfaces_.push_back(interface);
  }

  inline Interface get_interface(int i) const
  {
    range_check(interfaces_, i);
    return interfaces_[i];
  }

/// \brief Add new Interface object to list.
  ///
  /// \param master_index   cell index of the master cell.
  /// \return reference to added Interface object, that is
  /// valid until the next call to add_interface.
  inline Interface &add_interface(EntityNumber master_index, int master_facet_number)
  {
    interfaces_.push_back(Interface(master_index, master_facet_number));
    return interfaces_.back();
  }
  
private:

  // \brief Access an entry in the interfaces_ vector.
  inline Interface &get_interface(int i)
  {
    range_check(interfaces_, i);
    return interfaces_[i];
  }

  std::vector< Interface > interfaces_;
  ConstMeshPtr mesh_ = 0;
};

/// \brief Output of InterfaceList objects.
std::ostream &operator<<(std::ostream &os, const InterfaceList &interface_list);

/// \brief Description of the constellation of an Interface.

class InterfacePattern {
public:
  typedef std::vector< int >::const_iterator const_iterator;

  /// \brief Constructor.
  ///
  /// \details Creates invalid InterfacePattern
  /// (master_facet_pattern() == orientation() == -1). These
  /// fields must be set with the corresponding set_* member
  /// functions.
  InterfacePattern();

  /// \return  the facet number of the facet in the master cell.
  int master_facet_number() const;

  /// \return the number of slaves.
  int num_slaves() const;

  /// \return the facet number of the i:th slave facet in the slave cell.
  int slave_facet_number(int i) const;

  /// \return the facet number of the i:th slave facet in the
  /// parent of the slave cell, or the slave cell itself if the interface is
  /// regular.
  int slave_facet_number_in_parent(int i) const;

  /// \return iterator to the first slave facet number.
  const_iterator begin() const;

  /// \return iterator to the position just past the last slave facet number.
  const_iterator end() const;

  /// \return the orientation flag.
  int orientation() const;

  /// \return true if the facet number in the master cell, the
  /// facet numbers in all slave cells, and the orientation flags in this and p
  /// are equal (and in the same order).
  bool operator==(const InterfacePattern &p) const;

  bool operator!=(const InterfacePattern &p) const;

  bool operator<(const InterfacePattern &p) const;

  /// \brief Set the master facet number.
  /// \param master_facet   the new facet number in the master slave.
  void set_master_facet_number(int master_facet);

  /// \brief Add a regular slave facet.
  /// \param slave_facet    the facet number in the cell type of the slave
  void add_regular_slave_facet(int slave_facet_number);

  /// \brief Add an irregular slave facet.
  /// \param slave_facet_in_slave    the facet number in the slave cell itself.
  /// \param slave_facet_in_parent   the facet number in the parent neighboring
  /// the master cell.
  void add_irregular_slave_facet(int slave_facet_in_slave,
                                 int slave_facet_in_parent);

  /// \brief Set the orientation flag.
  ///
  /// \param orientation   the vertex number in the parent
  /// neighboring the master cell of the vertex that is common
  /// with the master cell, and has the lowest vertex number in the master cell.
  void set_orientation(int orientation);

private:
  int master_facet_number_;
  std::vector< int > slave_facet_numbers_;

  // slave_facet_numbers_in_parents contains the facet numbers
  // relative to the parent cell type is used. For regular
  // cells, this array contains the same as
  // slave_facet_numbers_.
  std::vector< int > slave_facet_numbers_in_parent_;
  int orientation_;
};

/// \brief Output of InterfacePattern objects.
std::ostream &operator<<(std::ostream &os,
                         const InterfacePattern &interface_pattern);

/// \brief Computes the InterfacePattern object corresponding to an Interface.
InterfacePattern compute_interface_pattern(ConstMeshPtr mesh,
                                           const Interface &interface);

} // namespace mesh
} // namespace hiflow
#endif /* _INTERFACE_H_ */
