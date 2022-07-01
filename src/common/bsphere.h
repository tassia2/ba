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

#ifndef HIFLOW_COMMON_BSPHERE_H
#define HIFLOW_COMMON_BSPHERE_H

/// \author Jonas Kratzke

#include "common/vector_algebra_descriptor.h"
#include "common/bbox.h"
#include <iosfwd>
#include <vector>
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>

namespace hiflow {
///
/// \brief   A simple sphere in any dimension.
///
/// \details This class constructs a bounding sphere around a set
/// of points. Points can be added by the member function
/// add_point(s). One can choose, whether the origin of the sphere
/// is to be kept or a simple algorithm by Jack Ritter[1990] should
/// be applied to approximately get the smallest bounding sphere.
/// It can be checked, whether two bounding spheres intersect.
///

template < class DataType, int DIM > 
class BSphere {
public:
  using Coord = typename StaticLA<DIM, DIM, DataType>::ColVectorType;

  /// \brief   Construction by an origin and radius
  BSphere(const Coord &origin, DataType radius = DataType(0));

  /// \brief   Construction by a set of points
  BSphere(size_t dim, const std::vector< Coord > &pts);

  /// \brief   Extension to additional points, given sequentially
  void add_points(const std::vector< Coord > &pts, bool fixed_origin = true);

  /// \brief   Radial extension
  void radial_extension(DataType extension);

  /// \brief   Check if two spheres intersect
  bool intersects(const BSphere< DataType, DIM > &other) const;

  /// \brief   Check if a box intersects the sphere
  bool intersects(const BBox< DataType, DIM > &box) const;

  /// \brief   Check if a box is completely contained in the sphere
  bool contains(const BBox< DataType, DIM > &box) const;

  /// \brief   Compute bounding box
  BBox< DataType, DIM > bbox() const;

  size_t get_dim() const;
  Coord get_origin() const;
  DataType get_radius() const;
  DataType get_diameter() const;

  void print(std::ostream &os) const;

private:
  Coord origin_;
  DataType radius_;
};

template < class DataType, int DIM >
BSphere< DataType, DIM >::BSphere(const Coord &origin,
                             DataType radius)
    : origin_(origin), radius_(radius) {
}

template < class DataType, int DIM >
BSphere< DataType, DIM >::BSphere(size_t dim, const std::vector< Coord > &pts) {
  assert(pts.size() > 0 );
  
  size_t num_pts = pts.size();
  // Jack Ritter's algorithm [1990] to approximate the smallest bounding sphere

  // pick some point
  Coord point_a = pts[0];

  // determine the point with biggest distance
  Coord point_b;
  DataType dist = 0;
  for (size_t n = 1; n < num_pts; ++n) {
    DataType dist_temp = norm(point_a - pts[n]);
    if (dist_temp > dist) {
      point_b = pts[n];
      dist = dist_temp;
    }
  }
  // determine the point with biggest distance to that point
  for (size_t n = 0; n < num_pts; ++n) {
    DataType dist_temp = norm(point_b - pts[n]);
    
    if (dist_temp > dist) {
      point_a = pts[n];
      dist = dist_temp;
    }
  }
  // now we have determined the two points of the set with biggest distance
  // initialize the bounding sphere with the midpoint as origin
  origin_ = point_a;
  for (size_t d = 0; d < DIM; ++d) {
    origin_.set(d, 0.5 * (origin_[d] + point_b[d]));
  }
  // and the radius is half the value of the distance
  radius_ = 0.5 * dist;
  // all other points can just be added with non-fixed origin
  add_points(pts, false);
}

template < class DataType, int DIM >
void BSphere< DataType, DIM >::add_points(const std::vector< Coord > &pts,
                                     bool fixed_origin) {
  assert( pts.size() > 0);
  size_t num_pts = pts.size();

  for (size_t n = 0; n < num_pts; ++n) {
    DataType dist = norm(origin_ - pts[n]);

    if (fixed_origin) {
      radius_ = std::max(radius_, dist);
    } else {
      if (dist > radius_) {
        radius_ = 0.5 * (dist + radius_);
        DataType factor = radius_ / dist;
        for (size_t d = 0; d < DIM; ++d) {
          origin_.set(d, factor * origin_[d] + (1 - factor) * pts[n][d]);
        }
      }
    }
  }
}

template < class DataType, int DIM >
void BSphere< DataType, DIM >::radial_extension(DataType extension) {
  radius_ += extension;
  assert(radius_ >= 0.);
}

template < class DataType, int DIM >
bool BSphere< DataType, DIM >::intersects(const BSphere< DataType, DIM > &other) const {
  assert(origin_.size() == other.get_dim());

  Coord other_origin = other.get_origin();
  DataType dist = norm(origin_ - other_origin);

  return (dist < radius_ + other.get_radius());
}

template < class DataType, int DIM >
bool BSphere< DataType, DIM >::intersects(const BBox< DataType, DIM > &box) const {

  Coord nearest_box_point;

  // do a projection on the box
  for (size_t d = 0; d < DIM; d++) {
    if (origin_[d] <= box.min(d)) {
      nearest_box_point.set(d, box.min(d));
    } else if (origin_[d] >= box.max(d)) {
      nearest_box_point.set(d, box.max(d));
    } else {
      nearest_box_point.set(d, origin_[d]);
    }
  }
  DataType dist = norm(origin_ - nearest_box_point);
  return (dist <= radius_);
}

template < class DataType, int DIM >
bool BSphere< DataType, DIM >::contains(const BBox< DataType, DIM > &box) const {

  bool contains = true;
  std::vector< Coord > box_vertices = box.get_vertices();

  assert( box_vertices.size() > 0);

  for (size_t n = 0; n < box_vertices.size(); ++n) {
    DataType dist = norm(origin_ - box_vertices[n]);
    if (dist > radius_) {
      return false;
    }
  }
  return contains;
}

template < class DataType, int DIM > BBox< DataType, DIM > BSphere< DataType, DIM >::bbox() const {
  BBox< DataType, DIM > bbox(DIM);
  bbox.add_point(origin_);
  bbox.uniform_extension(radius_);
  return bbox;
}

template < class DataType, int DIM > size_t BSphere< DataType, DIM >::get_dim() const {
  return origin_.size();
}

template < class DataType, int DIM >
typename BSphere< DataType, DIM >::Coord BSphere< DataType, DIM >::get_origin() const {
  return origin_;
}

template < class DataType, int DIM > DataType BSphere< DataType, DIM >::get_radius() const {
  return radius_;
}

template < class DataType, int DIM > DataType BSphere< DataType, DIM >::get_diameter() const {
  return radius_ * 2;
}

template < class DataType, int DIM >
void BSphere< DataType, DIM >::print(std::ostream &os) const {
  os << "[(";
  for (size_t i = 0; i < DIM; ++i) {
    os << origin_[i];
    if (i < DIM - 1) {
      os << ", ";
    }
  }
  os << "), " << radius_ << "]\n";
}

} // namespace hiflow

#endif
