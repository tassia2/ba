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

#ifndef HIFLOW_COMMON_GRID_H
#define HIFLOW_COMMON_GRID_H

/// \author Jonas Kratzke, Jonathan Schwegler, Philipp Gerstner

#include "common/vector_algebra_descriptor.h"
#include "common/log.h"
#include "common/bbox.h"
#include "common/bsphere.h"
#include "common/macros.h"
#include "common/sorted_array.h"
#include "mesh/cell_type.h"

#include <iosfwd>
#include <vector>
#include <cmath>
#include <iostream>

namespace hiflow {

/// \brief Description of a regular 2d or 3d grid.
///
/// \details Given a bounding box, this class constructs a
/// regular grid in its extensions. In general, the grid
/// can be of the form of all available cell_tags.
///
/// Currently full functionality is only given for rectangular
/// 2d and 3d grids.
///

template < class DataType, int DIM > 
class Grid {
public:
  using Coord = typename StaticLA<DIM, DIM, DataType>::ColVectorType;

  /// \brief   Constructs a rectangular grid with a BBox and a number of
  /// intervals on each axis.
  Grid();

  Grid(const std::vector< DataType > &grid_width, 
       const BBox< DataType, DIM > bbox);

  Grid(const std::vector< DataType > &grid_width, 
       const BBox< DataType, DIM > bbox,
       DataType eps);

  Grid(const std::vector< int > &num_intervals, 
       const BBox< DataType, DIM > bbox);

  Grid(const std::vector< int > &num_intervals, 
       const BBox< DataType, DIM > bbox,
       DataType eps);

  Grid(const mesh::CellType::Tag cell_tag,
       const std::vector< int > &num_intervals, 
       const BBox< DataType, DIM > bbox);

  Grid(const mesh::CellType::Tag cell_tag,
       const std::vector< int > &num_intervals, 
       const BBox< DataType, DIM > bbox,
       DataType eps);

  void init (const std::vector< int > &num_intervals, 
             const BBox< DataType, DIM > bbox,
             DataType eps);
  
  void init (const mesh::CellType::Tag cell_tag,
             const std::vector< int > &num_intervals, 
             const BBox< DataType, DIM > bbox,
             DataType eps);

  size_t get_gdim() const;
  std::vector< int > get_num_intervals() const;
  size_t get_num_points() const;
  size_t get_num_cells() const;
  BBox< DataType, DIM > get_bbox() const;

  /// \brief   Spacing on the respective axis.
  DataType delta(size_t i) const;

  /// \brief   Get the ids of the vertices of cell i
  const std::vector< int > &vertices_of_cell(size_t i);

  /// \brief   Get the coordinates of the grid
  const std::vector< Coord > &coords();

  /// \brief   Get the coordinates of a vertex
  Coord coords(std::vector< int > &indices) const;

  /// \brief   Get a rectangular grid cell as a box
  BBox< DataType, DIM > box(std::vector< int > &indices) const;

  /// \brief   Find cells that intersect a given box
  void intersect(const BBox< DataType, DIM > &bbox, std::vector< int > &cells) const;

  /// \brief   Find cells that intersect a given sphere
  void intersect(const BSphere< DataType, DIM > &sphere,
                 std::vector< int > &cells) const;

  /// \brief   Find the cell containing the point pt
  int cell_with_point(const Coord &pt) const;

  void get_neighbor_cells(int cell_index, std::vector<int>& ngh_indices) const;

  void build_point_map (const std::vector<Coord >& point_coords,
                        std::vector< int >& point_2_grid_map,
                        std::vector< SortedArray< int > >& grid_2_point_map,
                        std::vector< SortedArray< int > >& grid_2_point_ngh_map) const;

private:
  void init();
  void compute_coords_cells();
  void compute1d_Line();
  void compute2d_Quad();
  void compute3d_Hex();
  void compute2d_Tri();
  void compute3d_Tet();
  void compute3d_Pyr();
  int vertex_index_Line(int i) const;
  int vertex_index_Quad(int i, int j) const;
  int vertex_index_Tri(int i, int j) const;
  int vertex_index_Hex(int i, int j, int k) const;
  int vertex_index_Tet(int i, int j, int k) const;
  int vertex_index_Pyr(int i, int j, int k) const;
  int cell_index_Line(int i) const;
  int cell_index_Quad(int i, int j) const;
  int cell_index_Tri(int i, int j) const;
  int cell_index_Hex(int i, int j, int k) const;
  int cell_index_Tet(int i, int j, int k) const;
  int cell_index_Pyr(int i, int j, int k) const;
  void inverse_cell_index_Line(int cell_index, int& i) const;
  void inverse_cell_index_Quad(int cell_index, int& i, int& j) const;
  void inverse_cell_index_Hex(int cell_index, int& i, int& j, int& k) const;
  
  inline DataType lower (int d) const 
  {
    assert (d >= 0);
    assert (d < gdim_);

    return this->bbox_.min(d) - this->eps_;
  }
  
  inline DataType upper (int d) const 
  {
    assert (d >= 0);
    assert (d < gdim_);

    return this->bbox_.max(d) + this->eps_;
  }
  
  size_t gdim_;
  std::vector< int > num_intervals_;
  size_t num_points_;
  size_t num_cells_;
  BBox< DataType, DIM > bbox_;
  std::vector< std::vector< int > > cells_;
  std::vector< Coord > coords_;
  mesh::CellType::Tag tag_;
  DataType eps_;
};

//////////////// Grid ///////////////////////////////////////////////
template < class DataType, int DIM >
Grid< DataType, DIM >::Grid()
    : gdim_(-1), 
      tag_(mesh::CellType::NOT_SET),
      eps_(0.) 
{
  num_intervals_.clear();
}

template < class DataType, int DIM >
Grid< DataType, DIM >::Grid(const std::vector< int > &num_intervals,
                            BBox< DataType, DIM > bbox,
                            DataType eps)
    : gdim_(num_intervals.size()), num_intervals_(num_intervals), bbox_(bbox),
      tag_(num_intervals.size() == 2 ? mesh::CellType::QUADRILATERAL
                                     : mesh::CellType::HEXAHEDRON),
      eps_(eps) 
{
  init();
}

template < class DataType, int DIM >
Grid< DataType, DIM >::Grid(const std::vector< int > &num_intervals,
                            BBox< DataType, DIM > bbox)
    : Grid<DataType, DIM>(num_intervals, bbox, 0.)
{
}

template < class DataType, int DIM >
Grid< DataType, DIM >::Grid(const std::vector< DataType > &grid_width,
                            BBox< DataType, DIM > bbox,
                            DataType eps)
    : gdim_(grid_width.size()), bbox_(bbox),
      eps_(eps),
      tag_(grid_width.size() == 2 ? mesh::CellType::QUADRILATERAL
                                     : mesh::CellType::HEXAHEDRON) 
{
  this->num_intervals_.clear();
  this->num_intervals_.resize(DIM,0.);
  for (int d=0; d!=DIM; ++d)
  {
    assert (grid_width[d] > 0.);
    const DataType L = this->upper(d) - this->lower(d);
    const DataType n = L / grid_width[d];

    assert (n > 0);
    this->num_intervals_[d] = static_cast<int> (std::max(static_cast<DataType>(1.), std::floor(n)));
  }

  init();
}

template < class DataType, int DIM >
Grid< DataType, DIM >::Grid(const std::vector< DataType > &grid_width,
                            BBox< DataType, DIM > bbox)
    : Grid<DataType, DIM>(grid_width, bbox, 0.) 
{
}

template < class DataType, int DIM >
Grid< DataType, DIM >::Grid(mesh::CellType::Tag cell_tag,
                            const std::vector< int > &num_intervals,
                            BBox< DataType, DIM > bbox,
                            DataType eps)
    : gdim_(num_intervals.size()), 
      num_intervals_(num_intervals), 
      bbox_(bbox),
      tag_(cell_tag),
      eps_(eps) 
{
  init();
}

template < class DataType, int DIM >
Grid< DataType, DIM >::Grid(mesh::CellType::Tag cell_tag,
                            const std::vector< int > &num_intervals,
                            BBox< DataType, DIM > bbox)
    : Grid<DataType, DIM>(cell_tag, num_intervals, bbox, 0.)
{
}


template < class DataType, int DIM > 
void Grid< DataType, DIM >::init(const mesh::CellType::Tag cell_tag,
                                 const std::vector< int > &num_intervals, 
                                 const BBox< DataType, DIM > bbox,
                                 DataType eps) 
{
  this->bbox_ = bbox;
  this->tag_ = cell_tag;
  this->num_intervals_ = num_intervals;
  this->eps_ = eps;
  this->init();
}

template < class DataType, int DIM > 
void Grid< DataType, DIM >::init(const std::vector< int > &num_intervals, 
                                 const BBox< DataType, DIM > bbox,
                                 DataType eps) 
{
  this->bbox_ = bbox;
  this->num_intervals_ = num_intervals;
  this->eps_ = eps;
  this->tag_ = num_intervals.size() == 2 ? mesh::CellType::QUADRILATERAL
                                         : mesh::CellType::HEXAHEDRON;
  this->init();
}

template < class DataType, int DIM > 
void Grid< DataType, DIM >::init() 
{
  assert(gdim_ == mesh::CellType::get_instance(tag_).tdim());
  assert(gdim_ == bbox_.get_dim());
  for (size_t d = 0; d < gdim_; ++d) {
    assert(num_intervals_[d] > 0);
    assert(lower(d) < upper(d));
  }

  coords_.clear();
  cells_.clear();
  num_cells_ = 1;
  if (tag_ == mesh::CellType::PYRAMID) {
    assert(num_intervals_[0] == num_intervals_[1] &&
           num_intervals_[0] == num_intervals_[2]);
    if (num_intervals_[0] == 1) {
      num_cells_ = 1;
    } else if (num_intervals_[0] == 2) {
      num_cells_ = 10;
    } else {
      std::cerr << "Currently only works for num_intervals_ <= 2" << std::endl;
      quit_program();
    }
  } else {
    for (size_t d = 0; d < gdim_; ++d) {
      num_cells_ *= num_intervals_[d];
    }
  }

  switch (tag_) {
  case mesh::CellType::TRIANGLE:
    assert(num_intervals_[0] == num_intervals_[1]);
    num_points_ = (num_intervals_[0] + 1) * (num_intervals_[1] + 2) / 2;
    break;
  case mesh::CellType::TETRAHEDRON:
    assert(num_intervals_[0] == num_intervals_[1] &&
           num_intervals_[0] == num_intervals_[2]);
    num_points_ = (num_intervals_[0] + 1) * (num_intervals_[1] + 2) *
                  (num_intervals_[2] + 3) / 6;
    break;
  case mesh::CellType::LINE:
  case mesh::CellType::QUADRILATERAL:
  case mesh::CellType::HEXAHEDRON:
    num_points_ = 1;
    for (size_t d = 0; d < gdim_; ++d) {
      num_points_ *= num_intervals_[d] + 1;
    }
    break;
  case mesh::CellType::PYRAMID:
    assert(num_intervals_[0] == num_intervals_[1] &&
           num_intervals_[0] == num_intervals_[2]);
    num_points_ = (num_intervals_[0] + 1) * (num_intervals_[1] + 2) *
                  (2 * (num_intervals_[2] + 1) + 1) / 6;
    break;
  default:
    LOG_ERROR("Cell type " << tag_ << " not supported.");
    NOT_YET_IMPLEMENTED;
    break;
  }
}

template < class DataType, int DIM > 
void Grid< DataType, DIM >::compute_coords_cells() {
  switch (tag_) {
  case mesh::CellType::LINE:
    compute1d_Line();
    break;
  case mesh::CellType::TRIANGLE:
    compute2d_Tri();
    break;
  case mesh::CellType::QUADRILATERAL:
    compute2d_Quad();
    break;
  case mesh::CellType::TETRAHEDRON:
    compute3d_Tet();
    break;
  case mesh::CellType::HEXAHEDRON:
    compute3d_Hex();
    break;
  case mesh::CellType::PYRAMID:
    compute3d_Pyr();
    break;
  default:
    LOG_ERROR("Cell type " << tag_ << " not supported.");
    NOT_YET_IMPLEMENTED;
    break;
  }
}

template < class DataType, int DIM > 
void Grid< DataType, DIM >::compute1d_Line() {
  coords_.resize(num_points_);

  const DataType dx = delta(0);

  for (int i = 0; i < num_intervals_[0] + 1; ++i) {
    const DataType offset_x = i * dx;
    coords_[vertex_index_Line(i)].set(0, lower(0) + offset_x);
  }

  cells_.resize(num_cells_, std::vector< int >(2, -1));
  for (int i = 0; i < num_intervals_[0]; ++i) {
    cells_[i][0] = vertex_index_Line(i);
    cells_[i][1] = vertex_index_Line(i + 1);
  }
}

template < class DataType, int DIM > 
void Grid< DataType, DIM >::compute2d_Quad() {
  coords_.resize(num_points_);

  const DataType dx = delta(0);
  const DataType dy = delta(1);

  for (int i = 0; i < num_intervals_[0] + 1; ++i) {
    const DataType offset_x = i * dx;
    for (int j = 0; j < num_intervals_[1] + 1; ++j) {
      const int offset = vertex_index_Quad(i, j);
      coords_[offset].set(0, lower(0) + offset_x);
      coords_[offset].set(1, lower(1) + j * dy);
    }
  }

  cells_.resize(num_cells_, std::vector< int >(4, -1));
  for (int j = 0; j < num_intervals_[1]; ++j) {
    const int offset = j * num_intervals_[0];
    for (int i = 0; i < num_intervals_[0]; ++i) {
      cells_[offset + i][0] = vertex_index_Quad(i, j);
      cells_[offset + i][1] = vertex_index_Quad(i + 1, j);
      cells_[offset + i][2] = vertex_index_Quad(i + 1, j + 1);
      cells_[offset + i][3] = vertex_index_Quad(i, j + 1);
    }
  }
}

template < class DataType, int DIM > 
void Grid< DataType, DIM >::compute3d_Hex() {
  coords_.resize(num_points_);

  const DataType dx = delta(0);
  const DataType dy = delta(1);
  const DataType dz = delta(2);

  for (int i = 0; i < num_intervals_[0] + 1; ++i) {
    const DataType offset_x = i * dx;
    for (int j = 0; j < num_intervals_[1] + 1; ++j) {
      const DataType offset_y = j * dy;
      for (int k = 0; k < num_intervals_[2] + 1; ++k) {
        const int offset_ind = vertex_index_Hex(i, j, k);
        coords_[offset_ind].set(0, lower(0) + offset_x);
        coords_[offset_ind].set(1, lower(1) + offset_y);
        coords_[offset_ind].set(2, lower(2) + k * dz);
      }
    }
  }

  cells_.resize(num_cells_, std::vector< int >(8, -1));
  for (int i = 0; i < num_intervals_[0]; ++i) {
    for (int j = 0; j < num_intervals_[1]; ++j) {
      for (int k = 0; k < num_intervals_[2]; ++k) {
        const int cell_ind = num_intervals_[0] * num_intervals_[1] * k +
                             num_intervals_[0] * j + i;
        cells_[cell_ind][0] = vertex_index_Hex(i, j, k);
        cells_[cell_ind][1] = vertex_index_Hex(i + 1, j, k);
        cells_[cell_ind][2] = vertex_index_Hex(i + 1, j + 1, k);
        cells_[cell_ind][3] = vertex_index_Hex(i, j + 1, k);
        cells_[cell_ind][4] = vertex_index_Hex(i, j, k + 1);
        cells_[cell_ind][5] = vertex_index_Hex(i + 1, j, k + 1);
        cells_[cell_ind][6] = vertex_index_Hex(i + 1, j + 1, k + 1);
        cells_[cell_ind][7] = vertex_index_Hex(i, j + 1, k + 1);
      }
    }
  }
}

template < class DataType, int DIM > 
void Grid< DataType, DIM >::compute2d_Tri() {
  coords_.resize(num_points_);

  const DataType dx = delta(0);
  const DataType dy = delta(1);

  for (int i = 0; i < num_intervals_[0] + 1; ++i) {
    const DataType origin_i = lower(1) + i * dy;
    for (int j = 0; j < num_intervals_[1] + 1 - i; ++j) {
      const int offset = vertex_index_Tri(i, j);
      coords_[offset].set(0, lower(0) + j * dx);
      coords_[offset].set(1, origin_i);
    }
  }

  cells_.resize(num_cells_, std::vector< int >(3, -1));
  int cell_ind = 0;
  for (int i = 0; i < num_intervals_[0]; ++i) {
    for (int j = 0; j < num_intervals_[1] - i; ++j) {
      assert(cell_ind < cells_.size());
      cells_[cell_ind][0] = vertex_index_Tri(i, j);
      cells_[cell_ind][1] = vertex_index_Tri(i, j + 1);
      cells_[cell_ind][2] = vertex_index_Tri(i + 1, j);
      cell_ind++;
    }
  }
  for (int i = 0; i < num_intervals_[0] - 1; ++i) {
    for (int j = 1; j < num_intervals_[1] - i; ++j) {
      assert(cell_ind < cells_.size());
      cells_[cell_ind][0] = vertex_index_Tri(i, j);
      cells_[cell_ind][1] = vertex_index_Tri(i + 1, j - 1);
      cells_[cell_ind][2] = vertex_index_Tri(i + 1, j);
      cell_ind++;
    }
  }
}

template < class DataType, int DIM > 
void Grid< DataType, DIM >::compute3d_Tet() {
  coords_.resize(num_points_);

  const DataType dx = delta(0);
  const DataType dy = delta(1);
  const DataType dz = delta(2);

  for (int k = 0; k < num_intervals_[2] + 1; ++k) {
    const DataType origin_k = lower(2) + (num_intervals_[2] - k) * dz;
    for (int j = 0; j < k + 1; ++j) {
      const DataType origin_j = lower(1) + j * dy;
      for (int i = 0; i < k - j + 1; ++i) {
        const int offset = vertex_index_Tet(i, j, k);

        coords_[offset].set(0, lower(0) + i * dx);
        coords_[offset].set(1, origin_j);
        coords_[offset].set(2, origin_k);
      }
    }
  }

  cells_.resize(num_cells_, std::vector< int >(4, -1));

  int cell_ind = 0;
  for (int k = 0; k < num_intervals_[2]; ++k) {
    for (int j = 0; j < k + 1; ++j) {
      for (int i = 0; i < k - j + 1; ++i) {
        assert(cell_ind < cells_.size());
        cells_[cell_ind][0] = vertex_index_Tet(i, j, k);
        cells_[cell_ind][1] = vertex_index_Tet(i, j, k + 1);
        cells_[cell_ind][2] = vertex_index_Tet(i + 1, j, k + 1);
        cells_[cell_ind][3] = vertex_index_Tet(i, j + 1, k + 1);
        cell_ind++;
      }
    }
  }
  for (int k = 1; k < num_intervals_[2]; ++k) {
    for (int j = 0; j < k; ++j) {
      for (int i = 0; i < k - j; ++i) {
        assert(cell_ind < cells_.size() - 3);
        cells_[cell_ind][0] = vertex_index_Tet(i, j, k);
        cells_[cell_ind][1] = vertex_index_Tet(i + 1, j, k);
        cells_[cell_ind][2] = vertex_index_Tet(i, j + 1, k);
        cells_[cell_ind][3] = vertex_index_Tet(i + 1, j + 1, k + 1);
        cell_ind++;
        cells_[cell_ind][0] = vertex_index_Tet(i, j, k);
        cells_[cell_ind][1] = vertex_index_Tet(i + 1, j, k + 1);
        cells_[cell_ind][2] = vertex_index_Tet(i, j + 1, k + 1);
        cells_[cell_ind][3] = vertex_index_Tet(i + 1, j + 1, k + 1);
        cell_ind++;
        cells_[cell_ind][0] = vertex_index_Tet(i, j, k);
        cells_[cell_ind][1] = vertex_index_Tet(i + 1, j, k);
        cells_[cell_ind][2] = vertex_index_Tet(i + 1, j, k + 1);
        cells_[cell_ind][3] = vertex_index_Tet(i + 1, j + 1, k + 1);
        cell_ind++;
        cells_[cell_ind][0] = vertex_index_Tet(i, j, k);
        cells_[cell_ind][1] = vertex_index_Tet(i, j + 1, k);
        cells_[cell_ind][2] = vertex_index_Tet(i, j + 1, k + 1);
        cells_[cell_ind][3] = vertex_index_Tet(i + 1, j + 1, k + 1);
        cell_ind++;
      }
    }
  }
  for (int k = 2; k < num_intervals_[2]; ++k) {
    for (int j = 0; j < k - 1; ++j) {
      for (int i = 0; i < k - j - 1; ++i) {
        assert(cell_ind < cells_.size());
        cells_[cell_ind][0] = vertex_index_Tet(i + 1, j, k);
        cells_[cell_ind][1] = vertex_index_Tet(i, j + 1, k);
        cells_[cell_ind][2] = vertex_index_Tet(i + 1, j + 1, k);
        cells_[cell_ind][3] = vertex_index_Tet(i + 1, j + 1, k + 1);
        cell_ind++;
      }
    }
  }
}

template < class DataType, int DIM > 
void Grid< DataType, DIM >::compute3d_Pyr() {
  coords_.resize(num_points_);

  if (num_intervals_[0] > 2) {
    std::cerr << "Currently only works for num_intervals_ <= 2" << std::endl;
    ;
    quit_program();
  }

  const DataType dx = delta(0);
  const DataType dy = delta(1);
  const DataType dz = delta(2);

  for (int k = 0; k < num_intervals_[2] + 1; ++k) {
    for (int j = 0; j < num_intervals_[1] + 1 - k; ++j) {
      for (int i = 0; i < num_intervals_[0] + 1 - k; ++i) {
        const int offset_ind = vertex_index_Pyr(i, j, k);

        coords_[offset_ind].set(0,
            lower(0) +
            k * (upper(0) - lower(0)) / (num_intervals_[2]) * 0.5 +
            i * dx);
        coords_[offset_ind].set(1,
            lower(1) +
            k * (upper(0) - lower(0)) / (num_intervals_[2]) * 0.5 +
            j * dy);
        coords_[offset_ind].set(2, lower(2) + k * dz);
      }
    }
  }

  if (num_intervals_[0] == 1) {
    cells_.clear();

    // only one pyr
    cells_.resize(num_cells_, std::vector< int >(5, -1));

    cells_[0][0] = 0;
    cells_[0][1] = 1;
    cells_[0][2] = 3;
    cells_[0][3] = 2;
    cells_[0][4] = 4;

  } else if (num_intervals_[0] == 2) {
    cells_.clear();

    // 6 pyr + 4 tet
    const int num_pyr = 6;
    const int num_tet = 4;
    std::vector< std::vector< int > > cells_pyr(num_pyr,
                                                std::vector< int >(5, -1));
    std::vector< std::vector< int > > cells_tet(num_tet,
                                                std::vector< int >(4, -1));

    cells_.reserve(num_pyr + num_tet);
    cells_.insert(cells_.end(), cells_pyr.begin(), cells_pyr.end());
    cells_.insert(cells_.end(), cells_tet.begin(), cells_tet.end());

    assert(cells_.size() == num_cells_);

    // first 6 pyr
    int cell_ind = 0;

    const int n = num_intervals_[0];

    for (int k = 0; k < n; ++k) {
      for (int j = 0; j < n - k; ++j) {
        for (int i = 0; i < n - k; ++i) {

          cells_[cell_ind][0] = vertex_index_Pyr(i, j, k);
          cells_[cell_ind][1] = vertex_index_Pyr(i + 1, j, k);
          cells_[cell_ind][2] = vertex_index_Pyr(i + 1, j + 1, k);
          cells_[cell_ind][3] = vertex_index_Pyr(i, j + 1, k);
          cells_[cell_ind][4] = vertex_index_Pyr(i, j, k + 1);

          ++cell_ind;

          if (k == n - 1) {
            cells_[cell_ind][0] = vertex_index_Pyr(i, j, k);
            cells_[cell_ind][1] = vertex_index_Pyr(i, j + 1, k);
            cells_[cell_ind][2] = vertex_index_Pyr(i + 1, j + 1, k);
            cells_[cell_ind][3] = vertex_index_Pyr(i + 1, j, k);
            cells_[cell_ind][4] = vertex_index_Pyr(i + 1, j + 1, k - 1);

            ++cell_ind;
          }
        }
      }
    }

    // 1st tet
    cells_[cell_ind][0] = 1;
    cells_[cell_ind][1] = 9;
    cells_[cell_ind][2] = 10;
    cells_[cell_ind][3] = 4;
    ++cell_ind;

    // 2nd tet
    cells_[cell_ind][0] = 5;
    cells_[cell_ind][1] = 10;
    cells_[cell_ind][2] = 12;
    cells_[cell_ind][3] = 4;
    ++cell_ind;

    // 3rd tet
    cells_[cell_ind][0] = 7;
    cells_[cell_ind][1] = 12;
    cells_[cell_ind][2] = 11;
    cells_[cell_ind][3] = 4;
    ++cell_ind;

    // 4th tet
    cells_[cell_ind][0] = 3;
    cells_[cell_ind][1] = 11;
    cells_[cell_ind][2] = 9;
    cells_[cell_ind][3] = 4;
  }

  // const int n = num_intervals_[0] - 1;
  // const int num_hex = n * (n + 1) * (2 * n + 1) / 6;
  // const int num_pyr = n * (n + 1) + n + 1;
  // const int num_tet = n * (n + 1);

  // cells_.clear();
  // std::vector< std::vector<int> > cell_hex(num_hex, std::vector<int>(8, -1));
  // std::vector< std::vector<int> > cell_pyr(num_pyr, std::vector<int>(5, -1));
  // std::vector< std::vector<int> > cell_tet(num_tet, std::vector<int>(4, -1));

  // cells_.insert(cells_.end(), cell_hex.begin(), cell_hex.end());
  // cells_.insert(cells_.end(), cell_pyr.begin(), cell_pyr.end());
  // cells_.insert(cells_.end(), cell_tet.begin(), cell_tet.end());

  // assert(cells_.size() == num_cells_);

  // int cell_ind = 0;

  // //hex
  // for (int i = 0; i < n; ++i) {
  //   for (int j = 0; j < n; ++j) {
  //     for (int k = 0; k < n; ++k) {
  //       cells_[cell_ind][0] = vertex_index_Pyr(i, j, k);
  //       cells_[cell_ind][1] = vertex_index_Pyr(i + 1, j, k);
  //       cells_[cell_ind][2] = vertex_index_Pyr(i + 1, j + 1, k);
  //       cells_[cell_ind][3] = vertex_index_Pyr(i, j + 1, k);
  //       cells_[cell_ind][4] = vertex_index_Pyr(i, j, k + 1);
  //       cells_[cell_ind][5] = vertex_index_Pyr(i + 1, j, k + 1);
  //       cells_[cell_ind][6] = vertex_index_Pyr(i + 1, j + 1, k + 1);
  //       cells_[cell_ind][7] = vertex_index_Pyr(i, j + 1, k + 1);

  //       ++cell_ind;
  //     }
  //   }

  // }

  // //pyr
  // for (int k = 0; k < n + 1; ++k) {
  //   for (int j = 0; j < n - k; ++j) {
  //     int i = n - k;
  //     cells_[cell_ind][0] = vertex_index_Pyr(i, j, k);
  //     cells_[cell_ind][1] = vertex_index_Pyr(i + 1, j, k);
  //     cells_[cell_ind][2] = vertex_index_Pyr(i + 1, j + 1, k);
  //     cells_[cell_ind][3] = vertex_index_Pyr(i, j + 1, k);
  //     cells_[cell_ind][4] = vertex_index_Pyr(i, j, k + 1);

  //     ++cell_ind;
  //   }

  //   for (int i = 0; i < n - k; ++i) {
  //     int j = n - k;
  //     cells_[cell_ind][0] = vertex_index_Pyr(i, j, k);
  //     cells_[cell_ind][1] = vertex_index_Pyr(i + 1, j, k);
  //     cells_[cell_ind][2] = vertex_index_Pyr(i + 1, j + 1, k);
  //     cells_[cell_ind][3] = vertex_index_Pyr(i, j + 1, k);
  //     cells_[cell_ind][4] = vertex_index_Pyr(i, j, k + 1);

  //     ++cell_ind;
  //   }

  //   cells_[cell_ind][0] = vertex_index_Pyr(n - k, n - k, k);
  //   cells_[cell_ind][1] = vertex_index_Pyr(n - k + 1, n - k, k);
  //   cells_[cell_ind][2] = vertex_index_Pyr(n - k + 1, n - k + 1, k);
  //   cells_[cell_ind][3] = vertex_index_Pyr(n - k, n - k + 1, k);
  //   cells_[cell_ind][4] = vertex_index_Pyr(n, n, k + 1);

  //   // cells_[cell_ind][0] = 0;
  //   // cells_[cell_ind][1] = 1;
  //   // cells_[cell_ind][2] = 3;
  //   // cells_[cell_ind][3] = 2;
  //   // cells_[cell_ind][4] = 4;

  //   ++cell_ind;

  // }

  // //tet
  // for (int k = 0; k < n + 1; ++k) {
  //   for (int j = 0; j < n - k; ++j) {
  //     int i = n - k;
  //     cells_[cell_ind][0] = vertex_index_Pyr(i, j, k + 1);
  //     cells_[cell_ind][1] = vertex_index_Pyr(i, j + 1, k + 1);
  //     cells_[cell_ind][2] = vertex_index_Pyr(i, j + 1, k);
  //     cells_[cell_ind][3] = vertex_index_Pyr(i + 1, j + 1, k);

  //     ++cell_ind;
  //   }

  //   for (int i = 0; i < n - k; ++i) {
  //     int j = n - k;
  //     cells_[cell_ind][0] = vertex_index_Pyr(i, j, k + 1);
  //     cells_[cell_ind][1] = vertex_index_Pyr(i + 1, j, k + 1);
  //     cells_[cell_ind][2] = vertex_index_Pyr(i + 1, j + 1, k);
  //     cells_[cell_ind][3] = vertex_index_Pyr(i, j, k);

  //     ++cell_ind;
  //   }

  // }
}

template < class DataType, int DIM > 
size_t Grid< DataType, DIM >::get_gdim() const {
  return gdim_;
}

template < class DataType, int DIM > 
void Grid< DataType, DIM >::get_neighbor_cells(int cell_index, 
                                               std::vector<int>& ngh_indices) const 
{ 
  ngh_indices.clear();
  
  if (tag_ == mesh::CellType::LINE)
  {
    ngh_indices.reserve(3);
    int i = -1;
    
    this->inverse_cell_index_Line(cell_index, i);

    const int num_x = this->num_intervals_[0];
    
    int lower_i = std::max(0      , i-1);
    int upper_i = std::min(num_x-1, i+1);
    
    for (int ii = lower_i; ii <= upper_i; ++ii)
    {
      ngh_indices.push_back(this->cell_index_Line(ii));
    } 
  }
  else if (tag_ == mesh::CellType::QUADRILATERAL)
  {
    ngh_indices.reserve(9);
    int i = -1;
    int j = -1;
    
    this->inverse_cell_index_Quad(cell_index, i,j);

    const int num_x = this->num_intervals_[0];
    const int num_y = this->num_intervals_[1];
    
    int lower_i = std::max(0      , i-1);
    int upper_i = std::min(num_x-1, i+1);
    int lower_j = std::max(0      , j-1);
    int upper_j = std::min(num_y-1, j+1);
    
    for (int ii = lower_i; ii <= upper_i; ++ii)
    {
      for (int jj = lower_j; jj <= upper_j; ++jj)
      {
        ngh_indices.push_back(this->cell_index_Quad(ii,jj));
      }
    } 
  }
  else if (tag_ == mesh::CellType::HEXAHEDRON)
  {
    ngh_indices.reserve(27);
    int i = -1;
    int j = -1;
    int k = -1;
    
    this->inverse_cell_index_Hex(cell_index, i,j,k);

    const int num_x = this->num_intervals_[0];
    const int num_y = this->num_intervals_[1];
    const int num_z = this->num_intervals_[2];
    
    int lower_i = std::max(0      , i-1);
    int upper_i = std::min(num_x-1, i+1);
    int lower_j = std::max(0      , j-1);
    int upper_j = std::min(num_y-1, j+1);
    int lower_k = std::max(0      , k-1);
    int upper_k = std::min(num_z-1, k+1);
    
    for (int ii = lower_i; ii <= upper_i; ++ii)
    {
      for (int jj = lower_j; jj <= upper_j; ++jj)
      {
        for (int kk = lower_k; kk <= upper_k; ++kk)
        {
          ngh_indices.push_back(this->cell_index_Hex(ii,jj,kk));
        }
      }
    } 
  }
  else 
  {
    NOT_YET_IMPLEMENTED;
  }
}

template < class DataType, int DIM >
std::vector< int > Grid< DataType, DIM >::get_num_intervals() const {
  return num_intervals_;
}

template < class DataType, int DIM > 
size_t Grid< DataType, DIM >::get_num_points() const {
  return num_points_;
}

template < class DataType, int DIM > 
size_t Grid< DataType, DIM >::get_num_cells() const {
  return num_cells_;
}

template < class DataType, int DIM >
BBox< DataType, DIM > Grid< DataType, DIM >::get_bbox() const {
  return bbox_;
}

template < class DataType, int DIM > 
DataType Grid< DataType, DIM >::delta(size_t i) const {
  assert(i < gdim_);
  return (upper(i) - lower(i)) /
         static_cast< DataType >(num_intervals_[i]);
}

template < class DataType, int DIM >
const std::vector< int > &Grid< DataType, DIM >::vertices_of_cell(size_t i) {
  if (cells_.empty()) {
    compute_coords_cells();
  }
  return cells_.at(i);
}

template < class DataType, int DIM >
const std::vector< typename Grid< DataType, DIM >::Coord > &Grid< DataType, DIM >::coords() {
  if (coords_.empty()) {
    compute_coords_cells();
  }
  return coords_;
}

template < class DataType, int DIM >
typename Grid< DataType, DIM >::Coord 
Grid< DataType, DIM >::coords(std::vector< int > &indices) const {
  assert(indices.size() == gdim_);
  Coord coords;
  for (size_t d = 0; d < DIM; ++d) {
    assert(indices[d] >= 0);
    assert(indices[d] < num_intervals_[d] + 1);
    coords.set(d, lower(d) + indices[d] * delta(d));
  }
  return coords;
}

template < class DataType, int DIM >
BBox< DataType, DIM > Grid< DataType, DIM >::box(std::vector< int > &indices) const {
  assert(indices.size() == gdim_);
  std::vector< DataType > extends(2 * gdim_);
  if (tag_ == mesh::CellType::LINE || tag_ == mesh::CellType::QUADRILATERAL ||
      tag_ == mesh::CellType::HEXAHEDRON) {
    for (size_t d = 0; d < gdim_; ++d) {
      assert(indices[d] >= 0);
      assert(indices[d] < num_intervals_[d]);
      extends[2 * d] = lower(d) + indices[d] * delta(d);
      extends[2 * d + 1] = lower(d) + (indices[d] + 1) * delta(d);
    }
  } else {
    NOT_YET_IMPLEMENTED;
  }
  return BBox< DataType, DIM >(extends);
}

template < class DataType, int DIM >
void Grid< DataType, DIM >::intersect(const BBox< DataType, DIM > &box,
                                 std::vector< int > &cells) const {
  assert(gdim_ == box.get_dim());

  cells.clear();
  std::vector< int > min_ind(gdim_), max_ind(gdim_);
  for (size_t c = 0; c < gdim_; ++c) {
    const DataType d = this->delta(c);
    min_ind[c] = std::floor((box.min(c) - lower(c)) / d);
    min_ind[c] = std::max(0, min_ind[c]);

    max_ind[c] = std::ceil((box.max(c) - lower(c)) / d);
    max_ind[c] = std::min(num_intervals_[c], max_ind[c]);
  }

  switch (tag_) {
  case mesh::CellType::LINE:
    for (int i = min_ind[0]; i < max_ind[0]; ++i) {
      cells.push_back(cell_index_Line(i));
    }
    break;
  case mesh::CellType::QUADRILATERAL:
    for (int i = min_ind[0]; i < max_ind[0]; ++i) {
      for (int j = min_ind[1]; j < max_ind[1]; ++j) {
        cells.push_back(cell_index_Quad(i, j));
      }
    }
    break;
  case mesh::CellType::HEXAHEDRON:
    for (int i = min_ind[0]; i < max_ind[0]; ++i) {
      for (int j = min_ind[1]; j < max_ind[1]; ++j) {
        for (int k = min_ind[2]; k < max_ind[2]; ++k) {
          cells.push_back(cell_index_Hex(i, j, k));
        }
      }
    }
    break;
  case mesh::CellType::TRIANGLE:
  case mesh::CellType::TETRAHEDRON:
  case mesh::CellType::PYRAMID:
  default:
    // not implemented for Triangles and Tetrahedrons
    NOT_YET_IMPLEMENTED;
    break;
  }
}

template < class DataType, int DIM >
void Grid< DataType, DIM >::intersect(const BSphere< DataType, DIM > &sphere,
                                 std::vector< int > &cells) const {
  assert(gdim_ == sphere.get_dim());

  BBox< DataType, DIM > outer_box = sphere.bbox();
  cells.clear();
  std::vector< int > min_ind(gdim_), max_ind(gdim_);
  for (size_t c = 0; c < DIM; ++c) {
    const DataType d = this->delta(c);
    min_ind[c] = std::floor((outer_box.min(c) - lower(c)) / d);
    min_ind[c] = std::max(0, min_ind[c]);
    max_ind[c] = std::ceil((outer_box.max(c) - lower(c)) / d);
    max_ind[c] = std::min(num_intervals_[c], max_ind[c]);
  }

  switch (tag_) {
  case mesh::CellType::LINE:
    for (int i = min_ind[0]; i < max_ind[0]; ++i) {
      std::vector< int > indices(1);
      indices[0] = i;
      BBox< DataType, DIM > cell_box = box(indices);
      if (sphere.intersects(cell_box)) {
        cells.push_back(cell_index_Line(i));
      }
    }
    break;
  case mesh::CellType::QUADRILATERAL:
    for (int i = min_ind[0]; i < max_ind[0]; ++i) {
      for (int j = min_ind[1]; j < max_ind[1]; ++j) {
        std::vector< int > indices(2);
        indices[0] = i;
        indices[1] = j;
        BBox< DataType, DIM > cell_box = box(indices);
        if (sphere.intersects(cell_box)) {
          cells.push_back(cell_index_Quad(i, j));
        }
      }
    }
    break;
  case mesh::CellType::HEXAHEDRON:
    for (int i = min_ind[0]; i < max_ind[0]; ++i) {
      for (int j = min_ind[1]; j < max_ind[1]; ++j) {
        for (int k = min_ind[2]; k < max_ind[2]; ++k) {
          std::vector< int > indices(3);
          indices[0] = i;
          indices[1] = j;
          indices[2] = k;
          BBox< DataType, DIM > cell_box = box(indices);
          if (sphere.intersects(cell_box)) {
            cells.push_back(cell_index_Hex(i, j, k));
          }
        }
      }
    }
    break;
  case mesh::CellType::TRIANGLE:
  case mesh::CellType::TETRAHEDRON:
  case mesh::CellType::PYRAMID:
  default:
    // not implemented for Triangles and Tetrahedrons
    NOT_YET_IMPLEMENTED;
    break;
  }
}

template < class DataType, int DIM >
int Grid< DataType, DIM >::cell_with_point(const Coord &pt) const {
  assert(gdim_ == pt.size());

  for (size_t c = 0; c < DIM; ++c) {
    if ((pt[c] < lower(c)) || (pt[c] > upper(c))) {
      // pt is not in grid.
      
      LOG_DEBUG(1, "NOT FOUND IN GRID: c = " << c << ", pt[c] = " << pt[c]
                << ", min= " << lower(c) << ", max= " << upper(c));

      return -1;
    }
  }

  std::vector< int > ind(gdim_);
  int cell_ind;

  switch (tag_) {
  case mesh::CellType::LINE: {
    for (size_t c = 0; c < DIM; ++c) {
      const DataType d = this->delta(c);
      assert(d > 0);
      ind[c] = static_cast< int >(std::abs((pt[c] - lower(c)) / d));
      assert(ind[c] >= 0 && ind[c] <= num_intervals_[c]);
      if (ind[c] == num_intervals_[c]) {
        --ind[c];
      }
    }
    cell_ind = cell_index_Line(ind[0]);
    assert(cell_ind >= 0 && cell_ind < num_cells_);
    return cell_ind;
  } break;
  case mesh::CellType::QUADRILATERAL: {
    for (size_t c = 0; c < DIM; ++c) {
      const DataType d = this->delta(c);
      assert(d > 0);
      ind[c] = static_cast< int >(std::abs((pt[c] - lower(c)) / d));
      assert(ind[c] >= 0 && ind[c] <= num_intervals_[c]);
      if (ind[c] == num_intervals_[c]) {
        --ind[c];
      }
    }
    cell_ind = cell_index_Quad(ind[0], ind[1]);
    assert(cell_ind >= 0 && cell_ind < num_cells_);
    return cell_ind;
  } break;
  case mesh::CellType::HEXAHEDRON: {
    for (size_t c = 0; c < DIM; ++c) {
      const DataType d = this->delta(c);
      assert(d > 0);
      ind[c] = static_cast< int >(std::abs((pt[c] - lower(c)) / d));
      assert(ind[c] >= 0 && ind[c] <= num_intervals_[c]);
      if (ind[c] == num_intervals_[c]) {
        --ind[c];
      }
    }
    cell_ind = cell_index_Hex(ind[0], ind[1], ind[2]);
    assert(cell_ind >= 0 && cell_ind < num_cells_);
    return cell_ind;
  } break;
  case mesh::CellType::TRIANGLE: {
    if ((pt[0] - lower(0)) / (upper(0) - lower(0)) +
            (pt[1] - upper(1)) / (upper(1) - lower(1)) >
        0) {
      // pt is not in grid
      return -1;
    }
    // searching the quadrilateral where the triangular lies in
    // (has "index" (quadx, ind[1]) )
    const DataType dx = this->delta(0);
    const int quadx = std::floor((pt[0] - lower(0)) / dx);
    ind[0] = quadx * 2;
    const DataType dy = this->delta(1);
    ind[1] = std::floor((pt[1] - lower(1)) / dy);
    // check if pt is in an "upper right" or "lower left" triangular
    std::vector< DataType > ref_coords(2, 0.0);
    ref_coords[0] = quadx * dx;
    ref_coords[1] = ind[1] * dy;
    if ((pt[0] - ref_coords[0]) / dx + (pt[1] - (ref_coords[1] + dy)) / dy >
        0) {
      ind[0] += 1;
    }
    cell_ind = cell_index_Tri(ind[0], ind[1]);
    assert(cell_ind >= 0 && cell_ind < num_cells_);
    return cell_ind;
  } break;
  case mesh::CellType::TETRAHEDRON:
  case mesh::CellType::PYRAMID:
  default:
    // not implemented for Tetrahedrons
    NOT_YET_IMPLEMENTED;
    break;
  }

  return -1;
}

template < class DataType, int DIM >
int Grid< DataType, DIM >::vertex_index_Line(int i) const {
  return i;
}

template < class DataType, int DIM >
int Grid< DataType, DIM >::vertex_index_Quad(int i, int j) const {
  return (num_intervals_[0] + 1) * j + i;
}

template < class DataType, int DIM >
int Grid< DataType, DIM >::vertex_index_Tri(int i, int j) const {
  return (num_intervals_[1] + 1) * i - i * (i - 1) / 2 + j;
}

template < class DataType, int DIM >
int Grid< DataType, DIM >::vertex_index_Hex(int i, int j, int k) const {
  return (num_intervals_[0] + 1) * (num_intervals_[1] + 1) * k +
         (num_intervals_[0] + 1) * j + i;
}

template < class DataType, int DIM >
int Grid< DataType, DIM >::vertex_index_Tet(int i, int j, int k) const {
  return k * (k + 1) * (k + 2) / 6 + (k + 1) * j - j * (j - 1) / 2 + i;
}

template < class DataType, int DIM >
int Grid< DataType, DIM >::vertex_index_Pyr(int i, int j, int k) const {
  int number = 0;

  for (int it_k = 0; it_k < k; ++it_k) {
    number += (num_intervals_[0] + 1 - it_k) * (num_intervals_[0] + 1 - it_k);
  }

  number += j * (num_intervals_[1] + 1 - k) + i;

  return number;
}

template < class DataType, int DIM > 
int Grid< DataType, DIM >::cell_index_Line(int i) const {
  return i;
}

template < class DataType, int DIM > 
void Grid< DataType, DIM >::inverse_cell_index_Line(int cell_index, int& i) const {
  i = cell_index;
}

template < class DataType, int DIM >
int Grid< DataType, DIM >::cell_index_Quad(int i, int j) const {
  return num_intervals_[0] * j + i;
}

template < class DataType, int DIM > 
void Grid< DataType, DIM >::inverse_cell_index_Quad(int cell_index, int& i, int& j) const {
  j = cell_index / this->num_intervals_[0];
  i = cell_index % this->num_intervals_[0];
}

template < class DataType, int DIM >
int Grid< DataType, DIM >::cell_index_Tri(int i, int j) const {
  if ((i % 2) == 0) {
    return num_intervals_[0] * j - j * (j - 1) / 2 + i / 2;
  } else {
    return (num_intervals_[0] - 1) * j - j * (j - 1) / 2 + (i - 1) / 2 +
           num_intervals_[0] * (num_intervals_[1] + 1) / 2;
  }
}

template < class DataType, int DIM >
int Grid< DataType, DIM >::cell_index_Hex(int i, int j, int k) const {
  return num_intervals_[0] * num_intervals_[1] * k + num_intervals_[0] * j + i;
}

template < class DataType, int DIM > 
void Grid< DataType, DIM >::inverse_cell_index_Hex(int cell_index, int& i, int& j, int& k) const {
  const int num_x = this->num_intervals_[0];
  const int num_y = this->num_intervals_[1];
  const int num_xy = num_x * num_y;
  k = cell_index / num_xy;
  j = (cell_index - k * num_xy) / num_x;
  i = cell_index - k * num_xy - j * num_x; 
}

template < class DataType, int DIM >
int Grid< DataType, DIM >::cell_index_Tet(int i, int j, int k) const {
  // not yet implemented for Tetrahedrons
  NOT_YET_IMPLEMENTED;
  return -1;
}

template < class DataType, int DIM >
int Grid< DataType, DIM >::cell_index_Pyr(int i, int j, int k) const {
  // not yet implemented for Pyramid
  NOT_YET_IMPLEMENTED;
  return -1;
}

template < class DataType, int DIM >
void Grid< DataType, DIM >::build_point_map (const std::vector<Coord >& point_coords,
                                             std::vector< int >& point_2_grid_map,
                                             std::vector< SortedArray< int > >& grid_2_point_map,
                                             std::vector< SortedArray< int > >& grid_2_point_ngh_map) const
{
  const int num_grid_cells = this->get_num_cells();
  const int num_points = point_coords.size();
  grid_2_point_map.clear();
  grid_2_point_ngh_map.clear();
  point_2_grid_map.clear();

  grid_2_point_map.resize(num_grid_cells);
  grid_2_point_ngh_map.resize(num_grid_cells);
  point_2_grid_map.resize(num_points, -1); 

  std::vector<int> ngh_cells;
  for (int p=0; p!=num_points; ++p)
  {
    // grid cell index containing point
    int cell_index = this->cell_with_point(point_coords[p]);

    if (cell_index < 0)
    {
      continue;
    }

    // point -> cell
    point_2_grid_map[p] = cell_index;

    // cell -> points
    grid_2_point_map[cell_index].find_insert(p);

    // cell -> all points in neighboring cells
    ngh_cells.clear();
    this->get_neighbor_cells(cell_index, ngh_cells);
    for (auto c_it = ngh_cells.begin(), e_it = ngh_cells.end(); c_it != e_it; ++c_it)
    {
      grid_2_point_ngh_map[*c_it].find_insert(p);
    }
  }
}

} // namespace hiflow
#endif
