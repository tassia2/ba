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

#ifndef __QUADRATURE_QUADRATURE_H_
#define __QUADRATURE_QUADRATURE_H_

#include <cassert>
#include <iostream>
#include <vector>

#include "common/static_string.h"
#include "common/vector_algebra.h"
#include "quadrature/quadraturetype.h"
#include "mesh/cell_type.h"

namespace hiflow {

///
/// \class Quadrature quadrature.h
/// \brief Holds all necessary information about the desired quadrature rule
///        and is templated by a DataType (e.g. double)
/// \author Michael Schick
///

using QuadString = StaticString<100>;

template < class DataType > 
class Quadrature 
{
public:
  /// Default constructor (setting all pointers to NULL)
  Quadrature();
  /// Default destructor (clearing all pointers)
  ~Quadrature();
  /// Copy constructor (deep copying object)
  Quadrature(const Quadrature< DataType > &q);
  /// Assignment operator (deep copying object)
  const Quadrature< DataType > &operator=(const Quadrature< DataType > &q);

  /// Returns the size (number of quadrature points) of the quadrature

  int size() const { return size_; }
  int order() const { return order_; }
  
  /// Returns the name of the used quadrature

  const QuadString &name() const { return name_; }

  /// Setting up which quadrature rule should be used, e.g. GaussHexahedron
  /// and setting up the desired size
  void set_quadrature_by_size(const QuadString &name, int size);

  /// \brief Sets the quadrature rule by the desired order of accuracy.
  void set_quadrature_by_order(const QuadString &name, int order);

  /// Set up facet quadrature rule by mapping a lower-dimensional
  /// quadrature rule to a higher-dimensional cell.
  void set_facet_quadrature(const Quadrature< DataType > &base_quadrature,
                            mesh::CellType::Tag cell_tag, 
                            int facet_number);

  void set_custom_quadrature(const QuadString& name,
                             int order,
                             mesh::CellType::Tag cell_tag,
                             const std::vector< DataType > &x_coords,
                             const std::vector< DataType > &y_coords,
                             const std::vector< DataType > &z_coords,
                             const std::vector< DataType > &weights)
  {
    this->set_custom_quadrature(order, cell_tag, x_coords, y_coords, z_coords, weights);
    this->name_ = name;
  }
                             
  void set_custom_quadrature(int order,
                             mesh::CellType::Tag cell_tag,
                             const std::vector< DataType > &x_coords,
                             const std::vector< DataType > &y_coords,
                             const std::vector< DataType > &z_coords,
                             const std::vector< DataType > &weights);
  /// Set cell type
  inline void set_cell_tag(mesh::CellType::Tag cell_tag);

  // Get cell type
  mesh::CellType::Tag get_cell_tag() const { return cell_tag_; }
  
  inline int order_2_size (int order) const
  {
    return this->quad_->size_for_order(order);
  }

  inline int size_2_order (int size) const
  {
    return this->quad_->order_for_size(size);
  }
    
  /// Get the x value of the quadrature point with number qpt_index
  inline DataType x(int qpt_index) const;
  /// Get the y value of the quadrature point with number qpt_index
  inline DataType y(int qpt_index) const;
  /// Get the z value of the quadrature point with number qpt_index
  inline DataType z(int qpt_index) const;
  /// Get the weight value of the quadrature point with number qpt_index
  inline DataType w(int qpt_index) const;

  /// Print some status information about the used quadrature
  void print_status() const;

  inline const std::vector< DataType >& qpts_x(int size) const;
  inline const std::vector< DataType >& qpts_y(int size) const;
  inline const std::vector< DataType >& qpts_z(int size) const;
  inline const std::vector< DataType >& weights(int size) const;
  
protected:
  /// Reseting all quadrature pointers to NULL
  void clear();

  /// Name of quadrature
  QuadString name_;
  /// Size of quadrature, i.e. number of quadrature points
  int size_;

  int order_;
  
  /// Cell Type of the quadrature
  mesh::CellType::Tag cell_tag_;

  /// Holds the information about the used quadrature
  const QuadratureType< DataType > *quad_;

  /// Pointer to the x - values of the quadrature points
  const std::vector< DataType > *qpts_x_;
  /// Pointer to the y - values of the quadrature points
  const std::vector< DataType > *qpts_y_;
  /// Pointer to the z - values of the quadrature points
  const std::vector< DataType > *qpts_z_;
  /// Pointer to the weight values of the quadrature points
  const std::vector< DataType > *weights_;
};

///
/// \class SingleQuadrature quadrature.h
/// \brief Holds all necessary information about the desired quadrature rule
///        and is templated by a DataType (e.g. double)
///        Difference to Quadrature: SimpleQuadrature only holds quad points and weights 
///        for a single order 
/// \author Philipp Gerstner
///

template < class DataType > 
class SingleQuadrature 
{
public:
  /// Default constructor (setting all pointers to NULL)
  SingleQuadrature();
  /// Default destructor (clearing all pointers)
  ~SingleQuadrature();
  /// Copy constructor (deep copying object)
  SingleQuadrature(const SingleQuadrature< DataType > &q);
  
  /// Assignment operator (deep copying object)
  void extract_from(const Quadrature< DataType > &q);
  void copy_from_by_size(const Quadrature< DataType > &q, int size);
  void copy_from_by_order(const Quadrature< DataType > &q, int order);
  
  const SingleQuadrature< DataType > &operator=(const SingleQuadrature< DataType > &q);
  
  bool operator==(const SingleQuadrature< DataType > &q) const;
  bool operator==(const Quadrature< DataType > &q) const;

  bool operator!=(const SingleQuadrature< DataType > &q) const
  {
    return !(this->operator==(q));
  }
  bool operator!=(const Quadrature< DataType > &q) const
  {
    return !(this->operator==(q));
  }

  void set_custom_quadrature(int order,
                             mesh::CellType::Tag cell_tag,
                             const std::vector< DataType > &x_coords,
                             const std::vector< DataType > &y_coords,
                             const std::vector< DataType > &z_coords,
                             const std::vector< DataType > &weights);
                             
  /// Returns the size (number of quadrature points) of the quadrature
  int size() const 
  { 
    return size_; 
  }
  
  int order() const 
  { 
    return order_; 
  }
  
  /// Returns the name of the used quadrature
  const QuadString &name() const 
  { 
    return name_; 
  }

  // Get cell type
  mesh::CellType::Tag get_cell_tag() const 
  { 
    return cell_tag_; 
  }

  /// Get the x value of the quadrature point with number qpt_index
  inline DataType x(int qpt_index) const;
  /// Get the y value of the quadrature point with number qpt_index
  inline DataType y(int qpt_index) const;
  /// Get the z value of the quadrature point with number qpt_index
  inline DataType z(int qpt_index) const;
  /// Get the weight value of the quadrature point with number qpt_index
  inline DataType w(int qpt_index) const;

  inline void get_point(int qpt_index, Vec<1, DataType>& pt) const;
  inline void get_point(int qpt_index, Vec<2, DataType>& pt) const;
  inline void get_point(int qpt_index, Vec<3, DataType>& pt) const;

protected:
  /// Reseting all quadrature pointers to NULL
  void clear();

  /// Name of quadrature
  QuadString name_;
  /// Size of quadrature, i.e. number of quadrature points
  int size_;

  int order_;
  
  /// Cell Type of the quadrature
  mesh::CellType::Tag cell_tag_;

  /// Pointer to the x - values of the quadrature points
  std::vector< DataType > qpts_x_;
  /// Pointer to the y - values of the quadrature points
  std::vector< DataType > qpts_y_;
  /// Pointer to the z - values of the quadrature points
  std::vector< DataType > qpts_z_;
  /// Pointer to the weight values of the quadrature points
  std::vector< DataType > weights_;
  
  DataType eps_;
};

// INLINE FUNCTIONS FOR QUADRATURE

template < class DataType >
void Quadrature< DataType >::set_cell_tag(mesh::CellType::Tag cell_tag) {
  cell_tag_ = cell_tag;
}

template < class DataType >
DataType Quadrature< DataType >::x(int qpt_index) const {
  return (*qpts_x_)[qpt_index];
}

template < class DataType >
DataType Quadrature< DataType >::y(int qpt_index) const {
  return (*qpts_y_)[qpt_index];
}

template < class DataType >
DataType Quadrature< DataType >::z(int qpt_index) const {
  return (*qpts_z_)[qpt_index];
}

template < class DataType >
DataType Quadrature< DataType >::w(int qpt_index) const {
  return (*weights_)[qpt_index];
}

template < class DataType >
const std::vector<DataType>& Quadrature< DataType >::qpts_x(int size) const {
  return *(quad_->x(size));
}

template < class DataType >
const std::vector<DataType>& Quadrature< DataType >::qpts_y(int size) const {
  return *(quad_->y(size));
}

template < class DataType >
const std::vector<DataType>& Quadrature< DataType >::qpts_z(int size) const {
  return *(quad_->z(size));
}

template < class DataType >
const std::vector<DataType>& Quadrature< DataType >::weights(int size) const {
  return *(quad_->w(size));
}

// INLINE FUNCTIONS FOR SINGLEQUADRATURE
template < class DataType >
DataType SingleQuadrature< DataType >::x(int qpt_index) const {
  return qpts_x_[qpt_index];
}

template < class DataType >
DataType SingleQuadrature< DataType >::y(int qpt_index) const {
  return qpts_y_[qpt_index];
}

template < class DataType >
DataType SingleQuadrature< DataType >::z(int qpt_index) const {
  return qpts_z_[qpt_index];
}

template < class DataType >
DataType SingleQuadrature< DataType >::w(int qpt_index) const {
  return weights_[qpt_index];
}

template <class DataType >
void SingleQuadrature< DataType >::get_point(int qpt_index, Vec<1,DataType>& pt) const {
  pt.set(0, qpts_x_[qpt_index]);
}

template <class DataType >
void SingleQuadrature< DataType >::get_point(int qpt_index, Vec<2,DataType>& pt) const {
  pt.set(0, qpts_x_[qpt_index]);
  pt.set(1, qpts_y_[qpt_index]);
}

template <class DataType >
void SingleQuadrature< DataType >::get_point(int qpt_index, Vec<3,DataType>& pt) const {
  pt.set(0, qpts_x_[qpt_index]);
  pt.set(1, qpts_y_[qpt_index]);
  pt.set(2, qpts_z_[qpt_index]);
}

} // namespace hiflow

#endif
