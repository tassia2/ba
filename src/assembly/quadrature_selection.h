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

#ifndef _QUAD_SELECTION_H_
#define _QUAD_SELECTION_H_


#include "dof/dof_fem_types.h"

/// \file assembly.h
/// \brief Assembly functions.
///
/// \author Staffan Ronnas, Simon Gawlok
///

namespace hiflow {

template <class DataType, int DIM> class Element;
template <class DataType > class Quadrature;

/// The default quadrature selection chooses a quadrature rule that is accurate
/// to 3 * max(fe_degree).

template < class DataType, int DIM > 
class DefaultQuadratureSelection 
{
public:
  DefaultQuadratureSelection();
  ~DefaultQuadratureSelection()
  {}

  void operator()(const Element< DataType, DIM > &elem,
                  Quadrature< DataType > &quadrature); 
  
private:
  //doffem::RefCellType last_ref_cell_type_;
  //int last_order_;
};

/// The default facet quadrature selection chooses a facet quadrature rule that
/// is accurate to 2 * max(fe_degree).

template < class DataType, int DIM > 
class DefaultFacetQuadratureSelection 
{
public:
  DefaultFacetQuadratureSelection();
  ~DefaultFacetQuadratureSelection() {}

  void operator()(const Element< DataType, DIM > &elem,
                  Quadrature< DataType > &quadrature, int facet_number);

private:
  //doffem::RefCellType last_ref_cell_type_;
  //int last_order_;
  Quadrature< DataType > base_quadrature_;
};

template < class DataType, int DIM > 
class DefaultInterfaceQuadratureSelection 
{

public:
  DefaultInterfaceQuadratureSelection();
  ~DefaultInterfaceQuadratureSelection() {}
  
  void operator()(const Element< DataType, DIM > &master_elem,
                  const Element< DataType, DIM > &slave_elem,
                  int master_facet_number, int slave_facet_number,
                  Quadrature< DataType > &master_quad,
                  Quadrature< DataType > &slave_quad);

private:
  Quadrature< DataType > base_quad_;
  //doffem::RefCellType last_ref_cell_type_;
  //int last_order_;
};

} // namespace hiflow


#endif /* _ASSEMBLY_H_ */
