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

#include "quadrature/quadrature.h"
#include "common/array_tools.h"
#include "quadrature/custom_quadrature_type.h"
#include "quadrature/mapped_quadrature_type.h"
#include "quadrature/qeconomicalgausshexahedron.h"
#include "quadrature/qeconomicalgaussquadrilateral.h"
#include "quadrature/qgausshexahedron.h"
#include "quadrature/qgaussline.h"
#include "quadrature/qgausspyramid.h"
#include "quadrature/qgaussquadrilateral.h"
#include "quadrature/qgausstetrahedron.h"
#include "quadrature/qgausstriangle.h"

#include <sstream>

namespace hiflow {

/// Factory function for quadrature types.
template < class DataType >
QuadratureType< DataType > *create_quadrature_type(const QuadString &name) {
  if (name == "GaussTetrahedron") {
    return new QuadratureGaussTetrahedron< DataType >;
  }
  if (name == "GaussHexahedron") {
    return new QuadratureGaussHexahedron< DataType >;
  }
  if (name == "EconomicalGaussHexahedron") {
    return new QuadratureEconomicalGaussHexahedron< DataType >;
  }
  if (name == "GaussTriangle") {
    return new QuadratureGaussTriangle< DataType >;
  }
  if (name == "GaussQuadrilateral") {
    return new QuadratureGaussQuadrilateral< DataType >;
  }
  if (name == "EconomicalGaussQuadrilateral") {
    return new QuadratureEconomicalGaussQuadrilateral< DataType >;
  }
  if (name == "GaussLine") {
    return new QuadratureGaussLine< DataType >;
  }
  if (name == "GaussPyramid") {
    return new QuadratureGaussPyramid< DataType >;
  }
  assert(0);

  return 0;
}

mesh::CellType::Tag create_cell_tag(const QuadString &name) {
  if (name == "GaussTetrahedron") {
    return mesh::CellType::TETRAHEDRON;
  }
  if (name == "GaussHexahedron") {
    return mesh::CellType::HEXAHEDRON;
  }
  if (name == "EconomicalGaussHexahedron") {
    return mesh::CellType::HEXAHEDRON;
  }
  if (name == "GaussTriangle") {
    return mesh::CellType::TRIANGLE;
  }
  if (name == "GaussQuadrilateral") {
    return mesh::CellType::QUADRILATERAL;
  }
  if (name == "EconomicalGaussQuadrilateral") {
    return mesh::CellType::QUADRILATERAL;
  }
  if (name == "GaussLine") {
    return mesh::CellType::LINE;
  }
  if (name == "GaussPyramid") {
    return mesh::CellType::PYRAMID;
  }
  return mesh::CellType::NOT_SET;
}

template < class DataType >
Quadrature< DataType >::Quadrature()
    : name_(""), size_(0), order_(-1), cell_tag_(mesh::CellType::NOT_SET), quad_(0), qpts_x_(0), qpts_y_(0),
      qpts_z_(0), weights_(0) {}

template < class DataType > Quadrature< DataType >::~Quadrature() { clear(); }

template < class DataType >
Quadrature< DataType >::Quadrature(const Quadrature< DataType > &q)
    : name_(q.name()), size_(q.size()), quad_(0), order_(q.order()),  cell_tag_(q.get_cell_tag())
{
  if (q.quad_) {
    quad_ = q.quad_->clone();

    qpts_x_ = quad_->x(size_);
    qpts_y_ = quad_->y(size_);
    qpts_z_ = quad_->z(size_);
    weights_ = quad_->w(size_);
  }
}

template < class DataType >
const Quadrature< DataType > &Quadrature< DataType >::
operator=(const Quadrature< DataType > &q) {
  if (&q != this) {
    clear();

    size_ = q.size();
    name_ = q.name();

    quad_ = q.quad_->clone();

    qpts_x_ = quad_->x(size_);
    qpts_y_ = quad_->y(size_);
    qpts_z_ = quad_->z(size_);
    weights_ = quad_->w(size_);
  }
  return *this;
}

template < class DataType > void Quadrature< DataType >::clear() {

  delete quad_;

  quad_ = 0;
  qpts_x_ = 0;
  qpts_y_ = 0;
  qpts_z_ = 0;
  weights_ = 0;
  order_ = -1;
  size_ = 0;
  cell_tag_ = mesh::CellType::NOT_SET;
}

template < class DataType >
void Quadrature< DataType >::set_quadrature_by_size(const QuadString &name, int size) {
  clear();
  quad_ = create_quadrature_type<DataType>(name);
  this->cell_tag_ = create_cell_tag(name);
  
  assert(quad_ != 0);

  qpts_x_ = quad_->x(size);
  qpts_y_ = quad_->y(size);
  qpts_z_ = quad_->z(size);
  weights_ = quad_->w(size);

  assert(qpts_x_);
  assert(qpts_y_);
  assert(qpts_z_);
  assert(weights_);

  size_ = size;
  name_ = name;
  order_ = quad_->order_for_size(size);
}

template < class DataType >
void Quadrature< DataType >::set_quadrature_by_order(const QuadString &name, int order) {
  clear();

  quad_ = create_quadrature_type<DataType>(name);
  cell_tag_ = create_cell_tag(name);
  
  assert(quad_ != 0);

  const int size = quad_->size_for_order(order);

  qpts_x_ = quad_->x(size);
  qpts_y_ = quad_->y(size);
  qpts_z_ = quad_->z(size);
  weights_ = quad_->w(size);

  assert(qpts_x_);
  assert(qpts_y_);
  assert(qpts_z_);
  assert(weights_);

  size_ = size;
  name_ = name;
  order_ = order;
}

template < class DataType >
void Quadrature< DataType >::set_facet_quadrature(const Quadrature< DataType > &base_quadrature, 
                                                  mesh::CellType::Tag cell_tag,
                                                  int facet_number) 
{
  clear();

  const QuadString &base_name = base_quadrature.name();
  const int base_size = base_quadrature.size();

  QuadratureMapping< DataType > *mapping = nullptr;

  switch (cell_tag) {
  case mesh::CellType::TRIANGLE: // Triangle
    mapping = new TriangleQuadratureMapping< DataType >(facet_number);
    break;
  case mesh::CellType::QUADRILATERAL: // Quadrilateral
    mapping = new QuadrilateralQuadratureMapping< DataType >(facet_number);
    break;
  case mesh::CellType::TETRAHEDRON: // Tetrahedron
    mapping = new TetrahedralQuadratureMapping< DataType >(facet_number);
    break;
  case mesh::CellType::HEXAHEDRON: // Hexahedron
    mapping = new HexahedralQuadratureMapping< DataType >(facet_number);
    break;
  case mesh::CellType::PYRAMID: // Pyramid
    mapping = new PyramidQuadratureMapping< DataType >(facet_number);
    break;
  default:
    // TODO: not yet implemented
    std::cerr << "No mapping implemented for cell type " << cell_tag << ".\n";
    assert(false);
  };

  assert(mapping != nullptr);
  quad_ = new MappedQuadratureType< DataType >(base_quadrature, *mapping);
  size_ = base_size;
  order_ = base_quadrature.order();
  cell_tag_ = base_quadrature.get_cell_tag();
  qpts_x_ = quad_->x(size_);
  qpts_y_ = quad_->y(size_);
  qpts_z_ = quad_->z(size_);
  weights_ = quad_->w(size_);
  assert(qpts_x_ != 0);
  assert(qpts_y_ != 0);
  assert(qpts_z_ != 0);
  assert(weights_ != 0);

  // create unique name for quadrature
  //name_stream << base_name << "_" << base_size << "_Mapped_" << cell_tag << "_Facet_" << facet_number;

  name_ = base_name; 
  name_.append("_");
  name_.append_value(base_size);
  name_.append("_Mapped_");
  name_.append_value(as_integer(cell_tag));
  name_.append("_Facet_");
  name_.append_value(facet_number);

  delete mapping;
}

template < class DataType >
void Quadrature< DataType >::set_custom_quadrature(
    int order,
    mesh::CellType::Tag cell_tag,
    const std::vector< DataType > &x_coords,
    const std::vector< DataType > &y_coords,
    const std::vector< DataType > &z_coords,
    const std::vector< DataType > &weights) {
  clear();

  size_ = weights.size();

  assert(x_coords.size() == size_);
  assert(y_coords.size() == size_);
  assert(z_coords.size() == size_);

  quad_ = new CustomQuadratureType< DataType >(x_coords, y_coords, z_coords, weights);

  qpts_x_ = quad_->x(size_);
  qpts_y_ = quad_->y(size_);
  qpts_z_ = quad_->z(size_);
  weights_ = quad_->w(size_);

  // TODO: compute order
  order_ = order;
  cell_tag_ = cell_tag;
  
  assert(qpts_x_ != 0);
  assert(qpts_y_ != 0);
  assert(qpts_z_ != 0);
  assert(weights_ != 0);

  name_ = "Custom";
}

template < class DataType > void Quadrature< DataType >::print_status() const {
  std::cout << "Quadrature points for class Quadrature" << std::endl;

  std::cout << "Name of quadrature:\t" << name_ << std::endl;
  std::cout << "Size of quadrature:\t" << size_ << std::endl;

  if (qpts_x_ != 0) {
    std::cout << "x values:" << std::endl;
    for (int j = 0; j < size_; ++j) {
      std::cout << qpts_x_->at(j) << "\t";
    }
    std::cout << std::endl;
    std::cout << "-------------------------------------" << std::endl;
  } else {
    std::cout << "No x values defined for this quadrature!" << std::endl;
  }

  if (qpts_y_ != 0) {
    std::cout << "y values:" << std::endl;
    for (int j = 0; j < size_; ++j) {
      std::cout << qpts_y_->at(j) << "\t";
    }
    std::cout << std::endl;
    std::cout << "-------------------------------------" << std::endl;
  } else {
    std::cout << "No y values defined for this quadrature!" << std::endl;
  }

  if (qpts_z_ != 0) {
    std::cout << "z values:" << std::endl;
    for (int j = 0; j < size_; ++j) {
      std::cout << qpts_z_->at(j) << "\t";
    }
    std::cout << std::endl;
    std::cout << "-------------------------------------" << std::endl;
  } else {
    std::cout << "No z values defined for this quadrature!" << std::endl;
  }

  if (weights_ != 0) {
    std::cout << "weight values:" << std::endl;
    for (int j = 0; j < size_; ++j) {
      std::cout << weights_->at(j) << "\t";
    }
    std::cout << std::endl;
    std::cout << "-------------------------------------" << std::endl;
  } else {
    std::cout << "WARNING: No weights defined for this quadrature!"
              << std::endl;
  }
}

// template instanciation
template class Quadrature< double >;
template class Quadrature< float >;



template < class DataType >
SingleQuadrature< DataType >::SingleQuadrature()
    : name_(""), size_(0), order_(-1), cell_tag_(mesh::CellType::NOT_SET), eps_(1e-10) 
{}

template < class DataType > 
SingleQuadrature< DataType >::~SingleQuadrature() 
{ 
  this->clear(); 
}

template < class DataType >
SingleQuadrature< DataType >::SingleQuadrature(const SingleQuadrature< DataType > &q)
    : name_(q.name()), size_(q.size()), order_(q.order()), cell_tag_(q.get_cell_tag()), eps_(1e-10)
{
  this->qpts_x_ = q.qpts_x_;
  this->qpts_y_ = q.qpts_y_;
  this->qpts_z_ = q.qpts_z_;
  this->weights_ = q.weights_;
}

template < class DataType >
const SingleQuadrature< DataType > &SingleQuadrature< DataType >::
operator=(const SingleQuadrature< DataType > &q) {
  if (&q != this) 
  {
    clear();

    order_ = q.order();
    size_ = q.size();
    name_ = q.name();
    cell_tag_ = q.get_cell_tag();

    qpts_x_ = q.qpts_x_;
    qpts_y_ = q.qpts_y_;
    qpts_z_ = q.qpts_z_;
    weights_ = q.weights_;
  }
  return *this;
}

template < class DataType >
bool SingleQuadrature< DataType >::
operator==(const SingleQuadrature< DataType > &q) const
{
  if (this->size_ != q.size())
  {
    return false;
  }
  if (!vectors_are_equal(this->qpts_x_, q.qpts_x_, eps_))
  {
    return false;
  }
  if (!vectors_are_equal(this->qpts_y_, q.qpts_y_, eps_))
  {
    return false;
  }
  if (!vectors_are_equal(this->qpts_z_, q.qpts_z_, eps_))
  {
    return false;
  }
  if (!vectors_are_equal(this->weights_, q.weights_, eps_))
  {
    return false;
  }
  return true;
}

template < class DataType >
bool SingleQuadrature< DataType >::
operator==(const Quadrature< DataType > &q) const
{
  if (this->size_ != q.size())
  {
    return false;
  }
  const int e_i = q.size();
  
  if (!vectors_are_equal(this->qpts_x_, q.qpts_x(e_i), eps_))
  {
    return false;
  }
  if (!vectors_are_equal(this->qpts_y_, q.qpts_y(e_i), eps_))
  {
    return false;
  }
  if (!vectors_are_equal(this->qpts_z_, q.qpts_z(e_i), eps_))
  {
    return false;
  }
  if (!vectors_are_equal(this->weights_, q.weights(e_i), eps_))
  {
    return false;
  }
  return true;
}

template < class DataType > 
void SingleQuadrature< DataType >::clear() 
{
  qpts_x_.clear();
  qpts_y_.clear();
  qpts_z_.clear();
  weights_.clear();
  order_ = -1;
  size_ = 0;
  cell_tag_ = mesh::CellType::NOT_SET;
}

template < class DataType > 
void SingleQuadrature< DataType >::copy_from_by_size(const Quadrature< DataType > &q, int size)
{
  this->clear();
    
  order_ = q.order();
  size_ = q.size();
  assert(size == size_);
  
  name_ = q.name();
  cell_tag_ = q.get_cell_tag();
  this->qpts_x_ = q.qpts_x(size);
  this->qpts_y_ = q.qpts_y(size);
  this->qpts_z_ = q.qpts_z(size);
  this->weights_ = q.weights(size);  
}

template < class DataType > 
void SingleQuadrature< DataType >::copy_from_by_order(const Quadrature< DataType > &q, int order)
{
  this->clear();
  int size = q.order_2_size(order);
  this->copy_from_by_size(q, size);
}

template < class DataType > 
void SingleQuadrature< DataType >::extract_from(const Quadrature< DataType > &q)
{
  this->clear();
  int size = q.size();
  this->copy_from_by_size(q, size);
}

template < class DataType >
void SingleQuadrature< DataType >::set_custom_quadrature(
    int order,
    mesh::CellType::Tag cell_tag,
    const std::vector< DataType > &x_coords,
    const std::vector< DataType > &y_coords,
    const std::vector< DataType > &z_coords,
    const std::vector< DataType > &weights) {
  clear();

  size_ = weights.size();
  order_ = order;
  cell_tag_ = cell_tag;
  
  assert(x_coords.size() == size_);
  assert(y_coords.size() == size_);
  assert(z_coords.size() == size_);
  
  this->qpts_x_ = x_coords;
  this->qpts_y_ = y_coords;
  this->qpts_z_ = z_coords;
  this->weights_ = weights;
 
  name_ = "Custom";
}

// template instanciation
template class SingleQuadrature< double >;
template class SingleQuadrature< float >;
} // namespace hiflow
