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

#ifndef HIFLOW_COMMON_BBOX_H
#define HIFLOW_COMMON_BBOX_H

/// \author Staffan Ronnas, Jonas Kratzke

#include <iosfwd>
#include <vector>
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>

#include "common/vector_algebra_descriptor.h"

namespace hiflow {
///
/// \brief   A simple rectangular box in any dimension.
///
/// \details This class constructs a bounding box around a set
/// of points. Points can be added by the member function
/// add_point(s). It can be checked, whether two bounding
/// boxes intersect.
///

template < class DataType, int DIM > 
class BBox {
public:
  using Coord = typename StaticLA<DIM, DIM, DataType>::ColVectorType;

  /// \brief   Constructs an empty box
  BBox();

  BBox(int dim);

  /// \brief   Construction by an array of initial extents
  BBox(DataType *extents, size_t dim);

  /// \brief   Construction by a vector of initial extents
  BBox(const std::vector< DataType > &extents);

  /// \brief   Construction by a vector of vertex coordinates
  BBox(const std::vector< DataType > &vertices, size_t num_vert);
  
  DataType min(size_t dir) const;
  DataType max(size_t dir) const;

  /// \brief   Extension to an additional point
  void add_point(const Coord &pt);
  void Aadd_point(const std::vector< DataType > &pt);

  /// \brief   Extension to additional points, given sequentially
  void add_points(const std::vector< Coord > &pts);
  void Aadd_points(const std::vector< DataType > &pts);

  /// \brief   Extension with a constant value in every direction
  void uniform_extension(DataType extension);

  /// \brief   Check if two boxes intersect
  bool intersects(const BBox< DataType, DIM > &other) const;
  bool intersects(const BBox< DataType, DIM > &other, DataType eps) const;

  /// \brief   Check if box contains point
  bool contains(const Coord &pt, DataType eps) const;
  bool contains(const std::vector<DataType> &pt, DataType eps) const;

  /// \brief   Compute the diagonal of the box
  DataType compute_diagonal() const;

  void print(std::ostream &os) const;
  size_t get_dim() const;
  std::vector< DataType > get_extents() const;
  std::vector< Coord > get_vertices() const;

  DataType volume() const;
  
  void reset (size_t dim);

  void reset (const std::vector< DataType > &extents)
  {
    assert(extents.size() % 2 == 0);
    this->extents_ = extents;
  }
    
private:
  std::vector< DataType > extents_;
};

template < class DataType, int DIM > 
BBox< DataType, DIM >::BBox() {
  this->reset(DIM);
}

template < class DataType, int DIM > 
BBox< DataType, DIM >::BBox(int dim) {
  this->reset(dim);
}

template < class DataType, int DIM >
BBox< DataType, DIM >::BBox(DataType *extents, size_t dim) {
  extents_.assign(&extents[0], &extents[2 * dim]);
  assert(extents_.size() == 2 * dim);
}

template < class DataType, int DIM >
BBox< DataType, DIM >::BBox(const std::vector< DataType > &extents)
    : extents_(extents) {
  assert(!extents.empty());
  assert(!(extents.size() % 2));
}

template < class DataType, int DIM >
BBox< DataType, DIM >::BBox(const std::vector< DataType > &vertex_coords, size_t num_vert)
{
  assert (vertex_coords.size() == DIM * num_vert);

  extents_.resize(2 * DIM);
  for (size_t i = 0; i < DIM; ++i) {
    extents_[i * 2] = std::numeric_limits< DataType >::max();
    extents_[i * 2 + 1] = -std::numeric_limits< DataType >::max();
  }
  
  for (size_t i = 0; i != num_vert; ++i) 
  {
    for (size_t d=0; d != DIM; ++d)
    {  
      extents_[2 * d] = std::min(extents_[2 * d], vertex_coords[i*DIM+d]);
      extents_[2 * d + 1] = std::max(extents_[2 * d + 1], vertex_coords[i*DIM+d]);
    }
  }
}


template < class DataType, int DIM > 
void BBox< DataType, DIM >::reset(size_t dim) {
  extents_.clear();
  extents_.resize(2 * dim);
  for (size_t i = 0; i < dim; ++i) {
    extents_[i * 2] = std::numeric_limits< DataType >::max();
    extents_[i * 2 + 1] = -std::numeric_limits< DataType >::max();
  }
}

template < class DataType, int DIM > 
DataType BBox< DataType, DIM >::min(size_t dir) const {
  assert(extents_.size() > 2 * dir + 1);
  return extents_[2 * dir];
}

template < class DataType, int DIM > 
DataType BBox< DataType, DIM >::max(size_t dir) const {
  assert(extents_.size() > 2 * dir + 1);
  return extents_[2 * dir + 1];
}

template < class DataType, int DIM >
void BBox< DataType, DIM >::add_point(const Coord &pt) {
  assert(extents_.size() == 2 * DIM);
  for (size_t i = 0; i < DIM; ++i) {
    extents_[2 * i] = std::min(extents_[2 * i], pt[i]);
    extents_[2 * i + 1] = std::max(extents_[2 * i + 1], pt[i]);
  }
}

template < class DataType, int DIM >
void BBox< DataType, DIM >::Aadd_point(const std::vector<DataType> &pt) {
  assert(extents_.size() == 2 * DIM);
  assert (pt.size() == DIM);
  for (size_t i = 0; i < DIM; ++i) {
    extents_[2 * i] = std::min(extents_[2 * i], pt[i]);
    extents_[2 * i + 1] = std::max(extents_[2 * i + 1], pt[i]);
  }
}

template < class DataType, int DIM >
void BBox< DataType, DIM >::add_points(const std::vector< Coord > &pts) {
 
  size_t num_pts = pts.size();
  for (size_t n = 0; n < num_pts; ++n) {
    for (size_t i = 0; i < DIM; ++i) {
      extents_[2 * i] = std::min(extents_[2 * i], pts[n][i]);
      extents_[2 * i + 1] = std::max(extents_[2 * i + 1], pts[n][i]);
    }
  }
}

template < class DataType, int DIM >
void BBox< DataType, DIM >::Aadd_points(const std::vector< DataType > &pts) {
  assert ( (pts.size()  % DIM ) == 0 );

  size_t num_pts = pts.size() / DIM;
  for (size_t n = 0; n < num_pts; ++n) {
    for (size_t i = 0; i < DIM; ++i) {
      extents_[2 * i] = std::min(extents_[2 * i], pts[n*DIM + i]);
      extents_[2 * i + 1] = std::max(extents_[2 * i + 1], pts[n*DIM + i]);
    }
  }
}

template < class DataType, int DIM >
void BBox< DataType, DIM >::uniform_extension(DataType extension) {
  for (size_t i = 0; i < extents_.size() / 2; ++i) {
    extents_[2 * i] -= extension;
    extents_[2 * i + 1] += extension;
  }
}

template < class DataType, int DIM >
bool BBox< DataType, DIM >::intersects(const BBox< DataType, DIM > &other) const {
  assert(extents_.size() == 2 * other.get_dim());
  for (size_t i = 0; i < extents_.size() / 2; ++i) {
    if ((extents_[2 * i] > other.max(i)) ||
        extents_[2 * i + 1] < other.min(i)) {
      return false;
    }
  }
  return true;
}

template < class DataType, int DIM >
bool BBox< DataType, DIM >::intersects(const BBox< DataType, DIM > &other, DataType eps) const {
  assert(extents_.size() == 2 * other.get_dim());
  for (size_t i = 0; i < extents_.size() / 2; ++i) {
    if ((extents_[2 * i] > other.max(i) + eps) ||
        extents_[2 * i + 1] < other.min(i) - eps) {
      return false;
    }
  }
  return true;
}

template < class DataType, int DIM >
bool BBox< DataType, DIM >::contains(const Coord &pt, DataType eps) const
{
  assert(extents_.size() == 2 * pt.size());
  for (size_t i = 0; i < extents_.size() / 2; ++i) {
    if (extents_[2 * i] - eps > pt[i] || extents_[2 * i + 1] + eps < pt[i])
      return false;
  }
  return true;
}

template < class DataType, int DIM >
bool BBox< DataType, DIM >::contains(const std::vector<DataType> &pt, DataType eps) const
{
  assert(extents_.size() == 2 * pt.size());
  for (size_t i = 0; i < extents_.size() / 2; ++i) {
    if (extents_[2 * i] - eps > pt[i] || extents_[2 * i + 1] + eps < pt[i])
      return false;
  }
  return true;
}

template < class DataType, int DIM >
void BBox< DataType, DIM >::print(std::ostream &os) const {
  os << "[";
  for (size_t i = 0; i < extents_.size() / 2; ++i) {
    os << extents_[2 * i] << " .. " << extents_[2 * i + 1];
    if (i < extents_.size() / 2 - 1) {
      os << " x ";
    }
  }
  os << "]\n";
}

template < class DataType, int DIM > 
size_t BBox< DataType, DIM >::get_dim() const {
  return extents_.size() / 2;
}

template < class DataType, int DIM >
std::vector< DataType > BBox< DataType, DIM >::get_extents() const {
  return extents_;
}

template < class DataType, int DIM >
std::vector< typename BBox< DataType, DIM >::Coord > BBox< DataType, DIM >::get_vertices() const {
  size_t num_vertices = 1;
  for (size_t d = 0; d < DIM; ++d) {
    num_vertices *= 2;
  }
  std::vector< Coord > vertices(num_vertices);
  for (size_t n = 0; n < num_vertices; ++n) {
    size_t fac = 1;
    for (size_t d = 0; d < DIM; ++d) {
      vertices[n].set(d, extents_[2 * d + (n / fac) % 2]);
      fac *= 2;
    }
  }
  return vertices;
}

template < class DataType, int DIM >
DataType BBox< DataType, DIM >::compute_diagonal() const {
  DataType diagonal = 0;
  for (size_t i = 0; i < extents_.size() / 2; ++i) {
    diagonal += (extents_[2 * i + 1] - extents_[2 * i]) *
                (extents_[2 * i + 1] - extents_[2 * i]);
  }
  diagonal = std::sqrt(diagonal);

  return diagonal;
}

template < class DataType, int DIM >
DataType BBox< DataType, DIM >::volume() const {
  DataType result = 1.;
  for (size_t i = 0; i < extents_.size() / 2; ++i) 
  {
    result *= std::abs((extents_[2 * i + 1] - extents_[2 * i]));
  }
  return result;
}

} // namespace hiflow

#endif
