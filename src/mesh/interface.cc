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

#include "interface.h"

#include <iostream>
// Check if C++11 or newer is supported
#if __cplusplus > 199711L
#include <unordered_map>
#define UMAP_NAMESPACE std
#else
#include <boost/unordered_map.hpp>
#define UMAP_NAMESPACE boost::unordered
#endif
#include <cmath>

#include "common/log.h"

#include "cell_type.h"
#include "entity.h"
#include "iterator.h"
#include "mesh.h"
#include "mpi.h"

namespace hiflow {
namespace mesh {

//////////////// Begin Interface implementation ////////////////

std::ostream &operator<<(std::ostream &os, const Interface &interface) {
  os << "Interface {\n";
  os << "\tmaster cell index = " << interface.master_index() << "\n";
  os << "\tmaster facet number = " << interface.master_facet_number() << "\n";
  os << "\tslave cell indices = " << string_from_range(interface.begin(), interface.end()) << "\n";
  switch (interface.type())
  {
    case InterfaceType::NOT_SET:
      os << "\ttype = NOT_SET \n";
      break;
    case InterfaceType::INTERIOR:
      os << "\ttype = INTERIOR \n";
      break;
    case InterfaceType::OUTER_BDY:
      os << "\ttype = OUTER_BDY \n";
      break;
    case InterfaceType::GHOST_BDY:
      os << "\ttype = GHOST_BDY \n";
      break;
    default:
      os << "\ttype = unknown \n";
      break;      
  }
  os << "}\n";
  return os;
}

//////////////// End Interface implementation ////////////////

//////////////// Begin InterfaceList implementation ////////////////

namespace {
/// \brief Search for facet with index facet_index in the
/// given cell.
///
/// \returns true if the facet was found. The local
/// number of the facet is returned in facet_number.

bool find_facet_by_index(const Entity &cell, EntityNumber facet_index,
                         int &facet_number) {
  const TDim facet_tdim = cell.tdim() - 1;
  IncidentEntityIterator facet = cell.begin_incident(facet_tdim);
  const IncidentEntityIterator end = cell.end_incident(facet_tdim);

  facet_number = 0;
  while (facet != end && facet->index() != facet_index) {
    ++facet_number;
    ++facet;
  }

  return facet != end;
}

/// \brief Steps a forward iterator forward num_steps, by calling ++iterator.
/// \details No checking of the validity of the iterator is performed.

template < class IteratorT >
void fast_forward_iterator(IteratorT &iterator, int num_steps) {
  for (int k = 0; k < num_steps; ++k) {
    ++iterator;
  }
}

/// \brief Finds the local facet number in parent_cell of facet number
/// child_facet_number in child_cell. \pre parent_cell is the direct parent of
/// child_cell.

int facet_number_in_parent(const Entity &child_cell, const Entity &parent_cell,
                           int child_facet_number) {
  assert(child_cell.parent().index() == parent_cell.index());
  assert(child_cell.tdim() == parent_cell.tdim());

  const TDim facet_tdim = child_cell.tdim() - 1;

  // get sub-cell number of child cell
  int sub_cell_number;
  child_cell.get(std::string("__sub_cell_number__"), &sub_cell_number);
  assert(sub_cell_number > 0);

  // get cell type of parent
  const CellType &parent_cell_type = parent_cell.cell_type();

  // get map (sub_cell_number, local_facet_number_in_sub_cell) ->
  // local_facet_number_in_parent_cell
  const std::vector< int > &sub_facets_of_slave =
      parent_cell_type.sub_entities_of_cell(facet_tdim, sub_cell_number);

  assert(static_cast< int >(sub_facets_of_slave.size()) ==
         child_cell.cell_type().num_regular_entities(facet_tdim));

  range_check(sub_facets_of_slave, child_facet_number);
  return sub_facets_of_slave[child_facet_number];
}

/// \brief Finds the local number in parent_cell of the parent facet of facet
/// child_facet_number in child_cell. \pre parent_cell is the direct parent of
/// child_cell.

int parent_facet_number(const Entity &child_cell, const Entity &parent_cell,
                        int child_facet_number) {
  const TDim facet_tdim = parent_cell.tdim() - 1;
  const int facet_number =
      facet_number_in_parent(child_cell, parent_cell, child_facet_number);
  return parent_cell.cell_type().regular_parent(facet_tdim, facet_number);
}

/// \brief Return the first cell incident to a facet.

Entity incident_cell(const Entity &facet) {
  return *(facet.begin_incident(facet.tdim() + 1));
}

/// \brief Searches for a parent of a facet in the same mesh.

bool find_parent_facet_in_mesh(const Entity &facet, ConstMeshPtr mesh,
                               EntityNumber *parent_facet_index) {
  Entity cell = incident_cell(facet);
  LOG_DEBUG(1, "Cell with index " << cell.index() << " and id " << cell.id()
                                  << " has parent cell? " << cell.has_parent());

  if (!cell.has_parent()) {
    return false;
  }

  // get facet number of facet in the cell
  int facet_number;
  const bool found_facet =
      find_facet_by_index(cell, facet.index(), facet_number);
  assert(found_facet);

  // go up in ancestor tree of cell
  while (cell.has_parent()) {
    Entity parent_cell = cell.parent();
    LOG_DEBUG(1, "Ancestor Cell with index "
                     << parent_cell.index() << " and id " << parent_cell.id()
                     << " has parent cell? " << cell.has_parent());

    // get local number of parent facet
    facet_number = parent_facet_number(cell, parent_cell, facet_number);

    // check if facet really has parent (could also be interior)
    if (facet_number == -1) {
      return false;
    }

    IncidentEntityIterator parent_facet_it =
        parent_cell.begin_incident(facet.tdim());
    fast_forward_iterator(parent_facet_it, facet_number);

    if (mesh->find_entity(facet.tdim(), parent_facet_it->id(),
                          parent_facet_index)) {
      assert(parent_facet_it->id() != facet.id());
      return true;
    }
    cell = parent_cell;
  }

  return false;
}
} // namespace

// determine master cell of interface by comparing respective midpoints
int get_master_cell (const Entity& cell_0, const Entity& cell_1)
{
  const double eps=1e-12;
  
  std::vector<double> midpoint_0;
  std::vector<double> midpoint_1;
  cell_0.get_midpoint(midpoint_0);
  cell_1.get_midpoint(midpoint_1);
  const int gdim = midpoint_0.size();
  
  //std::cout << " ---- " << std::endl;
  //std::cout << "midpoint 0 " << midpoint_0[0] << " , " << midpoint_0[1] << std::endl;
  //std::cout << "midpoint 1 " << midpoint_1[0] << " , " << midpoint_1[1] << std::endl;
  
  for (int d=0; d<gdim; ++d)
  {
    if (midpoint_0[d] < midpoint_1[d] - eps)
    {
      return 0;
    }
    if (midpoint_1[d] < midpoint_0[d] - eps)
    {
      return 1;
    }
  }
  assert (false);
  return -1;
}

InterfaceList InterfaceList::create(ConstMeshPtr mesh) {
  // -- Algorithm -----
  // loop facets of mesh
  //   if facet not visited
  //     if facet has two cell neighbors
  //       add interface with master cell and slave cell
  //     else
  //       if facet has no parent in the mesh
  //         add interface with cell of facet as master
  //         add entry in irregular_facet->interface map
  //       else
  //         if parent does not exist in irregular_facet->interface map
  //            add interface with cell of parent as master
  //            add entry in irregular_facet->interface map
  //            mark parent
  //         endif
  //         add cell of facet to interface in irregular_facet->interface map
  //       endif
  //      endif
  //     endif
  //     mark facet
  //    endif
  // endloop

  const TDim cell_tdim = mesh->tdim();

  //    if (cell_tdim == 1) return InterfaceList(mesh);

  const TDim facet_tdim = cell_tdim - 1;

  // flags for visited facets
  std::vector< bool > visited_facets(mesh->num_entities(facet_tdim), false);

  // map of irregular master facets -> interfaces
  UMAP_NAMESPACE::unordered_map< EntityNumber, int > irregular_interfaces;

  // interface list to return
  InterfaceList interfaces(mesh);

  // loop over facets of mesh
  for (EntityIterator facet_it = mesh->begin(facet_tdim),
                      facet_end = mesh->end(facet_tdim);
       facet_it != facet_end; ++facet_it) {
    // have we already visited this facet?
    if (visited_facets[facet_it->index()]) {
      continue;
    }

    LOG_DEBUG(1, "Facet index: " << facet_it->index());
    const EntityCount num_cell_neighbors =
        facet_it->num_incident_entities(cell_tdim);

    assert(num_cell_neighbors == 1 || num_cell_neighbors == 2);

    if (num_cell_neighbors == 2) {
      // regular facet
      int facet_numbers[2];
      EntityNumber cell_indices[2];

      IncidentEntityIterator cell_it = facet_it->begin_incident(cell_tdim);
      for (int k = 0; k < 2; ++k) {
        cell_indices[k] = cell_it->index();
        const bool found_facet =
            find_facet_by_index(*cell_it, facet_it->index(), facet_numbers[k]);
        assert(found_facet);

        ++cell_it;
      }
      assert(cell_it == facet_it->end_incident(cell_tdim));

      // Determine master and slave. Master has lower facet
      // number. If both have same facet number, the
      // InterfacePatterns corresponding to both choices
      // will be identical, and so there is no need to
      // impose any further order.

      // const int master = ( facet_numbers[0] < facet_numbers[1] ) ? 0 : 1;

      // note(Philipp G): old master identification
      // const int master = (cell_indices[0] <= cell_indices[1]) ? 0 : 1;
      
      // note (Philipp G): new master identification, based on coordinates of midpoints
      // In this way, a consistent master/cell choice is ensured among those interfcaes,
      // that are shared by multiple process. 
      // This is important in the context of mixed finite elements, were degrees of freedom 
      // involve normal and tangent vectors on the cell boundaries; thus becoming orientation dependend.
      
      
      Entity cell_0 = mesh->get_entity(cell_tdim, cell_indices[0]);
      Entity cell_1 = mesh->get_entity(cell_tdim, cell_indices[1]);
      const int master = get_master_cell(cell_0, cell_1);
      const int slave = 1 - master;
      assert (master >= 0);
      
      /*
      std::cout << " cell 0 " << cell_indices[0] << std::endl;
      std::cout << " cell 1 " << cell_indices[1] << std::endl;
      std::cout << " master k " << master << std::endl;
      std::cout << " master index " << cell_indices[master] << "    , " << facet_numbers[master] << std::endl;
      std::cout << " slave index " << cell_indices[slave] << "    , " << facet_numbers[slave] << std::endl;
      */
      
      Interface &interface =
          interfaces.add_interface(cell_indices[master], facet_numbers[master]);
      interface.add_slave(cell_indices[slave], facet_numbers[slave]);
      LOG_DEBUG(1, "(Regular facet) Master cell index: "
                       << cell_indices[master] << " with local facet number "
                       << facet_numbers[master]);
      LOG_DEBUG(1, "(Regular facet) Slave cell index: "
                       << cell_indices[slave] << " with local facet number "
                       << facet_numbers[slave]);

    } 
    else 
    {
      assert(num_cell_neighbors == 1);

      // irregular or boundary facet
      EntityNumber parent_facet_index;
      bool facet_has_parent;
      if (facet_tdim == 0) {
        facet_has_parent = false;
      } else {
        facet_has_parent =
            find_parent_facet_in_mesh(*facet_it, mesh, &parent_facet_index);
      }

      if (!facet_has_parent) {
        // This facet is a "master" facet.
        Entity master_cell = incident_cell(*facet_it);

        EntityNumber facet_index = facet_it->index();

        int master_facet_number;
        const bool master_facet_found =
            find_facet_by_index(master_cell, facet_index, master_facet_number);
        assert(master_facet_found);

        const int interface_number = interfaces.size();

        // create interface
        interfaces.add_interface(master_cell.index(), master_facet_number);

        // update book-keeping
        irregular_interfaces[facet_index] = interface_number;

        LOG_DEBUG(1, "(Irregular Master facet) Master cell index: "
                         << master_cell.index() << " with local facet number "
                         << master_facet_number);
      } 
      else {
        // This facet is a "slave" facet.
        LOG_DEBUG(1, "On irregular slave facet, slave cell: "
                         << incident_cell(*facet_it).index() << " with id "
                         << incident_cell(*facet_it).id()
                         << " paretnfacet index " << parent_facet_index
                         << ", parent facet visited?: "
                         << visited_facets[parent_facet_index]);

        // Add parent interface if it has not yet been visited.
        if (!visited_facets[parent_facet_index]) {
          assert(irregular_interfaces.find(parent_facet_index) ==
                 irregular_interfaces.end());

          // extract info about master cell & facet
          Entity master_facet =
              mesh->get_entity(facet_tdim, parent_facet_index);
          Entity master_cell = incident_cell(master_facet);

          int master_facet_number;
          const bool master_facet_found = find_facet_by_index(
              master_cell, parent_facet_index, master_facet_number);
          assert(master_facet_found);

          const int interface_number = interfaces.size();

          // create interface
          interfaces.add_interface(master_cell.index(), master_facet_number);

          // update book-keeping
          visited_facets[parent_facet_index] = true;
          irregular_interfaces[parent_facet_index] = interface_number;
          LOG_DEBUG(1, "(Irregular Slave facet) Master cell index: "
                           << master_cell.index() << " with local facet number "
                           << master_facet_number);
        }

        assert(irregular_interfaces.find(parent_facet_index) !=
               irregular_interfaces.end());

        const int interface_pos =
            irregular_interfaces.find(parent_facet_index)->second;

        // add cell of current facet as slave
        const Entity slave_cell = incident_cell(*facet_it);
        LOG_DEBUG(1, "(Irregular Slave facet) Slave cell index: "
                         << slave_cell.index());
        int slave_facet_number = -1;
        const bool found = find_facet_by_index(slave_cell, facet_it->index(),
                                               slave_facet_number);
        assert(found);
        interfaces.get_interface(interface_pos)
            .add_slave(slave_cell.index(), slave_facet_number);

        LOG_DEBUG(1, "(Irregular Slave facet) Slave cell index: "
                         << slave_cell.index() << " with local facet number "
                         << slave_facet_number);

      } // master / slave facet
    }   // regular / irregular facet

    // mark facet as visited
    visited_facets[facet_it->index()] = true;
  } // end loop facets

  interfaces.determine_interface_types(mesh);
  return interfaces;
}

void InterfaceList::determine_interface_types(ConstMeshPtr mesh) 
{
  assert (this->mesh_ == 0 || mesh == this->mesh_);
  const int tdim = mesh->tdim();
  for (auto it = this->interfaces_.begin(),
       end_it = this->interfaces_.end();
       it != end_it; ++it) 
  {
    int remote_index_master = -10;
    if (mesh->has_attribute("_remote_index_", tdim))
    {
      mesh->get_attribute_value("_remote_index_", tdim, it->master_index(), &remote_index_master);
    }
    else 
    {
      remote_index_master = -1;
    }
    const int num_slaves = it->num_slaves(); 

    // subdom              = localdom + ghostdom
    // num_slaves = 0      -> bdy of subdom
    // num_slaves > 0      -> interior of subdom
    // remote_index == -1  -> cell \in localdom
    // remote_index >= 0   -> cell \in ghostdom
    // num_slaves = 0 / remote_index_master == -1 -> bdy of localdom = physical bdy
    // num_slaves = 0 / remote_index_master >= 0  -> bdy of ghost    = physical interior
    
    if (num_slaves == 0)
    {
      if (remote_index_master == -1) 
      {
        it->type() = InterfaceType::OUTER_BDY;
      }
      else if (remote_index_master >= 0)
      {
        it->type() = InterfaceType::GHOST_BDY;
      }
    }
    else 
    {
      it->type() = InterfaceType::INTERIOR;
    }
  }
}

std::ostream &operator<<(std::ostream &os,
                         const InterfaceList &interface_list) {
  os << "=== InterfaceList Begin "
        "============================================\n";
  for (InterfaceList::const_iterator it = interface_list.begin();
       it != interface_list.end(); ++it) 
  {
    int master_index = it->master_index();
    int num_slaves = it->num_slaves();
    int slave_index = -1;
    if (num_slaves == 1)
    {
      slave_index = it->slave_index(0);
    }
    
    int tdim = interface_list.mesh()->tdim();
    int remote_master, remote_slave;
    int subdom_master, subdom_slave;
    interface_list.mesh()->get_attribute_value("_remote_index_", tdim, master_index, &remote_master);
    interface_list.mesh()->get_attribute_value("_sub_domain_", tdim, master_index, &subdom_master);
    
    if (slave_index >= 0)
    {
      interface_list.mesh()->get_attribute_value("_remote_index_", tdim, slave_index, &remote_slave);
      interface_list.mesh()->get_attribute_value("_sub_domain_", tdim, slave_index, &subdom_slave);
    }
    else
    {
      remote_slave = -99;
      subdom_slave = -1;
    }
//    if ( slave_index >= 0
//      && (remote_master >= 0 || remote_slave >= 0) )
//    {
    os << *it
       << "master remote " << remote_master << " master subdom " << subdom_master << std::endl
       << "slave remote " << remote_slave << " slave subdom " << subdom_slave << std::endl
       << "--------" << std::endl;
//    }
  }
  os << "=== InterfaceList End "
        "==============================================\n";
  return os;
}

//////////////// End InterfaceList implementation ////////////////

//////////////// Begin InterfacePattern implementation ////////////////

InterfacePattern::InterfacePattern()
    : master_facet_number_(-1), orientation_(-1) {}

int InterfacePattern::master_facet_number() const {
  return master_facet_number_;
}

int InterfacePattern::num_slaves() const { return slave_facet_numbers_.size(); }

int InterfacePattern::slave_facet_number(int i) const {
  range_check(slave_facet_numbers_, i);
  return slave_facet_numbers_[i];
}

InterfacePattern::const_iterator InterfacePattern::begin() const {
  return slave_facet_numbers_.begin();
}

InterfacePattern::const_iterator InterfacePattern::end() const {
  return slave_facet_numbers_.end();
}

int InterfacePattern::orientation() const { return orientation_; }

bool InterfacePattern::operator==(const InterfacePattern &p) const {
  // NB vector comparison requires correct order
  return master_facet_number_ == p.master_facet_number_ &&
         orientation_ == p.orientation_ &&
         slave_facet_numbers_in_parent_ == p.slave_facet_numbers_in_parent_;
}

bool InterfacePattern::operator!=(const InterfacePattern &p) const {
  return !(operator==(p));
}

bool InterfacePattern::operator<(const InterfacePattern &p) const {
  if (master_facet_number_ < p.master_facet_number_) {
    return true;
  } else if (master_facet_number_ == p.master_facet_number_) {
    if (orientation_ < p.orientation_) {
      return true;
    } else if (orientation_ == p.orientation_) {
      return slave_facet_numbers_in_parent_ < p.slave_facet_numbers_in_parent_;
    }
  }
  return false;
}

std::ostream &operator<<(std::ostream &os,
                         const InterfacePattern &interface_pattern) {
  os << "InterfacePattern {\n";
  os << "\tmaster facet number = " << interface_pattern.master_facet_number()
     << "\n";
  os << "\tslave facet numbers = "
     << string_from_range(interface_pattern.begin(), interface_pattern.end())
     << "\n";
  os << "\torientation = " << interface_pattern.orientation() << "\n";
  os << "}\n";
  return os;
}

void InterfacePattern::set_master_facet_number(int master_facet) {
  master_facet_number_ = master_facet;
}

void InterfacePattern::add_regular_slave_facet(int slave_facet) {
  slave_facet_numbers_.push_back(slave_facet);
  slave_facet_numbers_in_parent_.push_back(slave_facet);
}

void InterfacePattern::add_irregular_slave_facet(int slave_facet_in_slave,
                                                 int slave_facet_in_parent) {
  slave_facet_numbers_.push_back(slave_facet_in_slave);
  slave_facet_numbers_in_parent_.push_back(slave_facet_in_parent);
}

void InterfacePattern::set_orientation(int orientation) {
  orientation_ = orientation;
}

// Helper functions for compute_interface_pattern
namespace {
/// \brief Search for facet with Id facet_id in the
/// given cell.
///
/// \returns true if the facet was found. The local
/// number of the facet is returned in facet_number.

bool find_facet_by_id(const Entity &cell, Id facet_id, int &facet_number) {
  const TDim facet_tdim = cell.tdim() - 1;
  IncidentEntityIterator facet = cell.begin_incident(facet_tdim);
  const IncidentEntityIterator end = cell.end_incident(facet_tdim);

  facet_number = 0;
  while (facet != end && facet->id() != facet_id) {
    ++facet_number;
    ++facet;
  }

  return facet != end;
}

int compute_orientation(const Entity &master_cell, int master_facet_number,
                        const Entity &slave_cell) {
  const TDim facet_tdim = master_cell.tdim() - 1;

  // find id of first vertex in master facet
  const std::vector< int > &master_facet_vertex_numbers =
      master_cell.cell_type().local_vertices_of_entity(facet_tdim,
                                                       master_facet_number);
  const int first_vertex_number = *std::min_element(
      master_facet_vertex_numbers.begin(), master_facet_vertex_numbers.end());
  assert(first_vertex_number >= 0);
  assert(first_vertex_number < master_cell.num_vertices());
  const Id first_vertex_id = master_cell.vertex_id(first_vertex_number);

  // search for vertex in slave cell
  for (int k = 0; k < slave_cell.num_vertices(); ++k) {
    if (slave_cell.vertex_id(k) == first_vertex_id) {
      return k;
    }
  }
  return -1;
}
} // namespace

/// \param mesh       pointer to the mesh to which the interface belongs.
/// \param interface  interface object.
/// \returns    the InterfacePattern corresponding to the interface object in
/// mesh.

InterfacePattern compute_interface_pattern(ConstMeshPtr mesh,
                                           const Interface &interface) {
  const TDim cell_tdim = mesh->tdim();
  const TDim facet_tdim = cell_tdim - 1;

  // object to initialize
  InterfacePattern pattern;

  // get facet number in master cell from interface object
  const int master_facet_number = interface.master_facet_number();
  pattern.set_master_facet_number(master_facet_number);

  // get entity for the interface (big) facet
  const EntityNumber master_cell_index = interface.master_index();
  const Entity master_cell = mesh->get_entity(cell_tdim, master_cell_index);
  assert(master_facet_number >= 0);
  assert(master_facet_number < master_cell.num_incident_entities(facet_tdim));

  IncidentEntityIterator master_facet_iterator =
      master_cell.begin_incident(facet_tdim);
  for (int k = 0; k < master_facet_number; ++k) {
    ++master_facet_iterator;
  }

  const Entity interface_facet = *master_facet_iterator;
  assert(interface_facet.num_incident_entities(cell_tdim) == 1 ||
         interface_facet.num_incident_entities(cell_tdim) == 2);

  if (interface_facet.num_incident_entities(cell_tdim) == 2) {
    // Case 1: regular interface between two cells -> no hanging facets
    assert(interface.num_slaves() == 1);

    // find slave facet number
    const EntityNumber slave_cell_index = interface.slave_index(0);
    const Entity slave_cell = mesh->get_entity(cell_tdim, slave_cell_index);

    int slave_facet_number;
    const bool found_facet = find_facet_by_index(
        slave_cell, interface_facet.index(), slave_facet_number);

    assert(found_facet);
    pattern.add_regular_slave_facet(slave_facet_number);

    // compute orientation
    IncidentEntityIterator slave_facet_iterator =
        slave_cell.begin_incident(facet_tdim);
    fast_forward_iterator(slave_facet_iterator, slave_facet_number);

    assert(slave_facet_iterator != slave_cell.end_incident(facet_tdim));

    const int orientation =
        compute_orientation(master_cell, master_facet_number, slave_cell);
    assert(orientation >= 0);
    pattern.set_orientation(orientation);
  } else { // interface_facet.num_incident_entities() == 1
    if (interface.num_slaves() == 0) {
      // Case 2: a boundary interface -> no need to look for slave facets.

      // the orientation is meaningless in this case, and hence set to -1
      pattern.set_orientation(-1);
    } else {

      if (DEBUG_LEVEL >= 2) {
        std::vector< int > slave_indices;
        std::vector< int > incident_cells;
        slave_indices.reserve(interface.num_slaves());
        for (int l = 0; l < interface.num_slaves(); ++l) {
          slave_indices.push_back(interface.slave_index(l));
        }
        for (IncidentEntityIterator cell_it =
                 interface_facet.begin_incident(cell_tdim);
             cell_it != interface_facet.end_incident(cell_tdim); ++cell_it) {
          incident_cells.push_back(cell_it->index());
        }

        LOG_DEBUG(2, "Irregular facet with master index "
                         << master_cell_index << " num incident cells "
                         << interface_facet.num_incident_entities(cell_tdim)
                         << " slave indices: "
                         << string_from_range(slave_indices.begin(),
                                              slave_indices.end())
                         << " incident cells: "
                         << string_from_range(incident_cells.begin(),
                                              incident_cells.end()));
      }

      // Case 3: irregular refinement with hanging facets
      //              assert(interface.num_slaves() > 1);

      // Find master facet in ancestor of first slave cell
      Entity parent_cell =
          mesh->get_entity(cell_tdim, interface.slave_index(0));
      int parent_facet_number;
      do {
        // Entity::parent() asserts that parent exists
        parent_cell = parent_cell.parent();
      } while (!find_facet_by_id(parent_cell, interface_facet.id(),
                                 parent_facet_number));
      assert(parent_facet_number >= 0);
      assert(parent_facet_number <
             parent_cell.num_incident_entities(facet_tdim));

      IncidentEntityIterator parent_facet_iterator =
          parent_cell.begin_incident(facet_tdim);
      fast_forward_iterator(parent_facet_iterator, parent_facet_number);

      Entity parent_facet = *parent_facet_iterator;

      const int orientation =
          compute_orientation(master_cell, master_facet_number, parent_cell);
      assert(orientation >= 0);
      pattern.set_orientation(orientation);

      // Find the facet numbers for all slave cells
      const int num_slave_cells = interface.num_slaves();
      for (int s = 0; s < num_slave_cells; ++s) {
        // find ancestor of slave that has parent_cell as parent.
        const EntityNumber slave_cell_index = interface.slave_index(s);
        Entity slave_cell = mesh->get_entity(cell_tdim, slave_cell_index);
        Entity parent_of_slave = slave_cell.parent();
        while (parent_of_slave.id() != parent_cell.id()) {
          slave_cell = slave_cell.parent();
          parent_of_slave = slave_cell.parent();
        }

        // Find local facet number in slave cell such that
        // its parent facet in the CellType of parent_cell is
        // parent_facet_number.
        const CellType &parent_cell_type = parent_cell.cell_type();
        int sub_cell_number;
        slave_cell.get(std::string("__sub_cell_number__"), &sub_cell_number);

        assert(sub_cell_number >= 0);
        assert(sub_cell_number < parent_cell_type.num_entities(cell_tdim));

        const std::vector< int > &sub_facets_of_slave =
            parent_cell_type.sub_entities_of_cell(facet_tdim, sub_cell_number);
        assert(static_cast< int >(sub_facets_of_slave.size()) ==
               slave_cell.cell_type().num_regular_entities(facet_tdim));

        const int check_num_slaves = pattern.num_slaves();

        for (size_t k = 0; k != sub_facets_of_slave.size(); ++k) {
          const int slave_facet_number_in_parent = sub_facets_of_slave[k];
          if (parent_cell_type.regular_parent(facet_tdim,
                                              slave_facet_number_in_parent) ==
              parent_facet_number) {
            pattern.add_irregular_slave_facet(k, slave_facet_number_in_parent);
            break;
          }
        }

        // check that a facet number was added
        assert(pattern.num_slaves() == check_num_slaves + 1);
      }
    }
  }

  return pattern;
}

//////////////// End InterfacePattern implementation ////////////////

} // namespace mesh
} // namespace hiflow
