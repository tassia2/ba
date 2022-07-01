// Copyright (C) 2011-2017 Vincent Heuveline
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

/// \author Staffan Ronnas, Simon Gawlok

#include "assembly/quadrature_selection.h"

#include <vector>
#include <numeric>
#include <cmath>
#include <boost/function.hpp>

#include "assembly/assembly_utils.h"
#include "fem/fe_reference.h"
#include "fem/reference_cell.h"
#include "space/element.h"
#include "quadrature/quadrature.h"

namespace hiflow {

template < class DataType, int DIM > 
DefaultQuadratureSelection<DataType, DIM>::DefaultQuadratureSelection()
//: last_ref_cell_type_(doffem::RefCellType::NOT_SET), last_order_(-1) 
{
}

template < class DataType, int DIM >
void DefaultQuadratureSelection<DataType, DIM>::operator()(const Element< DataType, DIM > &elem,
                                                           Quadrature< DataType > &quadrature) 
  {
    // we assume that all FE in current element live on the same reference cell
    //const doffem::RefCellType ref_cell_type = elem.ref_cell()->type();
    const mesh::CellType::Tag cur_tag = elem.ref_cell()->tag();
    size_t fe_deg = 0;

    // compute maxmimum FE degree for all variables
    for (size_t v = 0, end_v = elem.nb_fe(); v < end_v; ++v) 
    {
      fe_deg = std::max(fe_deg, elem.get_fe(v)->max_deg());
    }

    const int desired_order = 3 * fe_deg;

/*
    std::cout << " Default " <<  this 
              << ":: old order " << quadrature.order()
              << " , des order " << desired_order << " , " 
              << " old cell " << quadrature.get_cell_tag() 
              << " , cur cell " << cur_tag
              << " size " << quadrature.size() << std::endl;
*/
               
    // Return early if we already have the desired quadrature.
    // This is a very important optimization, since setting
    // the quadrature is quite expensive. The elements are
    // typically sorted before traversal, which means that we
    // can minimize the number of quadrature switches through this.
    if (quadrature.size() > 0 && 
        cur_tag == quadrature.get_cell_tag() &&
        desired_order == quadrature.order()) 
    {
      return;
    }

    QuadString quad_name = elem.ref_cell()->get_quad_name_cell_gauss (false); // false = not economical
    quadrature.set_cell_tag(cur_tag);
    quadrature.set_quadrature_by_order(quad_name, desired_order);

    //last_ref_cell_type_ = ref_cell_type;
    //last_order_ = desired_order;
  }

template < class DataType, int DIM > 
DefaultFacetQuadratureSelection <DataType, DIM>::DefaultFacetQuadratureSelection()
//      : last_ref_cell_type_(doffem::RefCellType::NOT_SET), last_order_(-1) 
{}

template < class DataType, int DIM > 
void DefaultFacetQuadratureSelection <DataType, DIM>::operator()(const Element< DataType, DIM > &elem,
                                                                 Quadrature< DataType > &quadrature, 
                                                                 int facet_number) 
{
    // we assume that all FE in current element live on the same reference cell  
    //const doffem::RefCellType ref_cell_type = elem.ref_cell()->type();
    const mesh::CellType::Tag cur_tag = elem.ref_cell()->tag();

    size_t fe_deg = 0;

    // compute maxmimum FE degree for all variables
    for (size_t v = 0, end_v = elem.nb_fe(); v < end_v; ++v) 
    {
      fe_deg = std::max(fe_deg, elem.get_fe(v)->max_deg());
    }

    const int desired_order = 3 * fe_deg;

    
    // Only set base quadrature if it changed.
    // This is a very important optimization, since setting
    // the quadrature is quite expensive. The elements are
    // typically sorted before traversal, which means that we
    // can minimize the number of quadrature switches through this.
    if (!(base_quadrature_.size() > 0 
       && cur_tag == quadrature.get_cell_tag() 
       && desired_order == quadrature.order()) ||
        cur_tag == mesh::CellType::PYRAMID ) 
    {
      QuadString quad_name = elem.ref_cell()->get_quad_name_facet_gauss (facet_number, false); // false = not economical
      base_quadrature_.set_cell_tag(elem.ref_cell()->facet_tag(facet_number));
      base_quadrature_.set_quadrature_by_order(quad_name, desired_order);

      //last_ref_cell_type_ = ref_cell_type;
      //last_order_ = desired_order;
    }
    quadrature.set_facet_quadrature(base_quadrature_, cur_tag, facet_number);
  }

template < class DataType, int DIM > 
DefaultInterfaceQuadratureSelection <DataType, DIM>::DefaultInterfaceQuadratureSelection()
//      : last_ref_cell_type_(doffem::RefCellType::NOT_SET), last_order_(-1) 
{}

template < class DataType, int DIM > 
void DefaultInterfaceQuadratureSelection<DataType, DIM>::operator()(const Element< DataType, DIM > &master_elem,
                                                                    const Element< DataType, DIM > &slave_elem,
                                                                    int master_facet_number, int slave_facet_number,
                                                                    Quadrature< DataType > &master_quad,
                                                                    Quadrature< DataType > &slave_quad) 
{
  // we assume that all FE in current element live on the same reference cell
  assert (slave_elem.ref_cell()->type() == master_elem.ref_cell()->type());
  
  //const doffem::RefCellType ref_cell_type = slave_elem.ref_cell()->type();
  const mesh::CellType::Tag cur_tag = slave_elem.ref_cell()->facet_tag(slave_facet_number);
  
  size_t fe_deg = 0;

  // compute maxmimum FE degree for all variables
  for (size_t v = 0, end_v = slave_elem.nb_fe(); v < end_v; ++v) 
  {
    fe_deg = std::max(fe_deg, slave_elem.get_fe(v)->max_deg());
  }
  for (size_t v = 0, end_v = master_elem.nb_fe(); v < end_v; ++v) 
  {
    fe_deg = std::max(fe_deg, master_elem.get_fe(v)->max_deg());
  }

  const int desired_order = 3 * fe_deg;
             
  if (!(base_quad_.size() > 0 
     && cur_tag == base_quad_.get_cell_tag() 
     && desired_order == base_quad_.order())) 
  {
    QuadString quad_name = slave_elem.ref_cell()->get_quad_name_facet_gauss (slave_facet_number, true); // true = economical
    base_quad_.set_cell_tag(cur_tag);
    base_quad_.set_quadrature_by_order(quad_name, desired_order);

    //last_ref_cell_type_ = ref_cell_type;
    //last_order_ = desired_order;
  }
  slave_quad.set_facet_quadrature(base_quad_, slave_elem.ref_cell()->tag(), slave_facet_number);
  init_master_quadrature(slave_elem, master_elem, slave_quad, master_quad);
}

template class DefaultQuadratureSelection<double, 3>;
template class DefaultQuadratureSelection<double, 2>;
template class DefaultQuadratureSelection<double, 1>;
template class DefaultQuadratureSelection<float, 3>;
template class DefaultQuadratureSelection<float, 2>;
template class DefaultQuadratureSelection<float, 1>;


template class DefaultFacetQuadratureSelection<double, 3>;
template class DefaultFacetQuadratureSelection<double, 2>;
template class DefaultFacetQuadratureSelection<double, 1>;
template class DefaultFacetQuadratureSelection<float, 3>;
template class DefaultFacetQuadratureSelection<float, 2>;
template class DefaultFacetQuadratureSelection<float, 1>;

template class DefaultInterfaceQuadratureSelection<double, 3>;
template class DefaultInterfaceQuadratureSelection<double, 2>;
template class DefaultInterfaceQuadratureSelection<double, 1>;
template class DefaultInterfaceQuadratureSelection<float, 3>;
template class DefaultInterfaceQuadratureSelection<float, 2>;
template class DefaultInterfaceQuadratureSelection<float, 1>;

} // namespace hiflow
