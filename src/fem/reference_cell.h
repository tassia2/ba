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

#ifndef __FEM_REF_CELL_H_
#define __FEM_REF_CELL_H_

#include <cassert>
#include <iostream>
#include <vector>

#include <cmath>

#include "common/macros.h"
#include "common/vector_algebra_descriptor.h"
#include "common/log.h"
#include "common/static_string.h"
#include "mesh/cell_type.h"
#include "dof/dof_fem_types.h"
#include "quadrature/quadrature.h"

namespace hiflow {
namespace doffem {

///
/// \class ReferenceCell reference_cell.h
/// \brief Ancestor class of different reference cells used for defining a FE
/// \author Philipp Gerstner


template < class DataType, int DIM > 
class RefCell 
{
public:
  using Coord = typename StaticLA<DIM, DIM, DataType>::ColVectorType;
  using SCoord = typename StaticLA<DIM-1, DIM, DataType>::ColVectorType;
  using Mat = typename StaticLA<DIM, DIM-1, DataType>::MatrixType;

  /// Default Constructor
  RefCell()
  : topo_cell_(nullptr), 
    type_(RefCellType::NOT_SET),
    // std::numeric_limits<double>::epsilon ( ) = 2.22045e-16
    // 1e3 is used, because same tolerance is used in function compute_weights(),
    // see numbering_lagrange.cc
    eps_ (1.e3 * std::numeric_limits< DataType >::epsilon()) 
  {}

  /// Default Destructor  
  // TODO: delete cell object?
  virtual ~RefCell() 
  {}

  /// Tpological dimension of cell
  inline size_t tdim() const
  {
    assert (this->topo_cell_ != nullptr);
    return this->topo_cell_->tdim();
  }
  
  inline RefCellType type () const
  {
    return this->type_;
  }

  inline mesh::CellType const * topo_cell () const
  {
    return this->topo_cell_;
  }

  inline mesh::CellType::Tag tag () const
  {
    assert (this->topo_cell_ != nullptr);
    return this->topo_cell_->tag();
  }
  
  inline size_t num_vertices () const 
  {
    return this->coord_vertices_.size();
  }

  inline DataType eps() const
  {
    return this->eps_;
  }

  inline size_t num_entities(int tdim) const 
  {
    return this->topo_cell_->num_entities(tdim);
  }
  
  virtual std::vector< Coord > get_coords () const = 0;

  /// check whether coordinates of underlying reference cell coincide with 
  /// given set of points
  bool ref_coord_match( const std::vector<Coord> & test_coord) const
  {
    std::vector<Coord> my_coord = this->get_coords();
    if (my_coord.size() == 0 || test_coord.size() == 0)
    {
      return false;
    }
    if (my_coord.size() != test_coord.size() )
    {
      return false;
    }
    for (size_t p = 0; p<test_coord.size(); ++p)
    {
      if (norm (my_coord[p] - test_coord[p]) > this->eps_)
      {
        return false;
      }
    }
    return true;
  }

  virtual bool contains_point ( const Coord & pt) const = 0;

  // If pt lies on the boundary of the reference cell, return
  // the reference facet. If pt lies on an edge or is a vertex, 
  // the numbers of all facets containing pt are returned.
  virtual bool get_facet_nr( const Coord & pt, std::vector < int > &facet_nr) const = 0;

  virtual void compute_facet_projection_matrix (int facet_number, Mat& proj) const = 0;

  virtual void compute_facet_normal (int facet_number, Coord& n) const = 0;

  virtual void compute_facet_tangents (int facet_number, Coord& t1, Coord& t2) const = 0;

  /// Map point on reference facet (line, triangle, quad) to coordinates on facet on reference cell with given facet_nr
  virtual Coord Tf (size_t facet_nr, const SCoord & ref_pt) const = 0;

  virtual DataType ds (size_t facet_nr, const SCoord & ref_pt) const = 0;

  virtual QuadString get_quad_name_cell_gauss (bool use_economical) const = 0;
  virtual QuadString get_quad_name_facet_gauss (int facet_nr, bool use_economical) const = 0;
  virtual mesh::CellType::Tag facet_tag(int facet_nr) const = 0;

protected:
  /// Storing an instance of the reference cell
  mesh::CellType const * topo_cell_;

  RefCellType type_;

  /// tolerance for checking whether reference coordinates lie in cell
  DataType eps_;

};

template < class DataType, int DIM > 
class RefCellLineStd final : public RefCell<DataType, DIM>
{
public:
  using Coord = typename StaticLA<DIM, DIM, DataType>::ColVectorType;
  using SCoord = typename StaticLA<DIM-1, DIM, DataType>::ColVectorType;
  using Mat = typename StaticLA<DIM, DIM-1, DataType>::MatrixType;

  /// Default Constructor
  RefCellLineStd()
  {
    this->topo_cell_ = &(mesh::CellType::get_instance(mesh::CellType::LINE));
    this->type_ = RefCellType::LINE_STD;
  }

  std::vector< Coord > get_coords () const
  {
    std::vector<Coord> ref_coord(2);
    ref_coord[1].set(0, 1.);

    return ref_coord;  
  }

  void compute_facet_projection_matrix (int facet_number, Mat& proj) const
  {
    assert (0);
  }

  void compute_facet_normal (int facet_number, Coord& n) const
  {
    assert (0);
  }

  void compute_facet_tangents (int facet_number, Coord& t1, Coord& t2) const
  {
    assert (0);
  }

  inline bool contains_point (const Coord &pt) const 
  {
    return pt[0] >= -this->eps_ && pt[0] <= 1. + this->eps_;
  }

  inline bool get_facet_nr(const Coord &pt, std::vector< int > &facet_nr) const
  {
    return false;
  }

  inline Coord Tf (size_t facet_nr, const SCoord & ref_pt) const
  {
    assert (0);
    Coord tmp;
    return tmp;
  }

  inline DataType ds (size_t facet_nr, const SCoord & ref_pt) const
  {
    assert (0);
    return 0.;
  }

  inline QuadString get_quad_name_cell_gauss (bool use_economical) const 
  {
    return "GaussLine";  
  }

  inline QuadString get_quad_name_facet_gauss (int facet_nr, bool use_economical) const 
  {
    return "";  
  }

  inline mesh::CellType::Tag facet_tag(int facet_nr) const
  {
    return mesh::CellType::POINT;
  }

  /// Default Destructor
  virtual ~RefCellLineStd()
  {
  }
};

template < class DataType, int DIM > 
class RefCellQuadStd final : public RefCell<DataType, DIM>
{
public:
  using Coord = typename StaticLA<DIM, DIM, DataType>::ColVectorType;
  using SCoord = typename StaticLA<DIM-1, DIM, DataType>::ColVectorType;
  using Mat = typename StaticLA<DIM, DIM-1, DataType>::MatrixType;
  /// Default Constructor
  RefCellQuadStd()
  {
    this->topo_cell_ = &(mesh::CellType::get_instance(mesh::CellType::QUADRILATERAL));
    this->type_ = RefCellType::QUAD_STD;
  }

  std::vector< Coord > get_coords () const
  {
    std::vector<Coord> ref_coord(4);
    ref_coord[1].set(0, 1.);
    ref_coord[2].set(0, 1.);
    ref_coord[2].set(1, 1.);
    ref_coord[3].set(1, 1.);

    return ref_coord;  
  }

  void compute_facet_projection_matrix (int facet_number, Mat& proj) const
  {
    assert (DIM > 1);
    proj.Zeros();
    switch (facet_number) 
    {
      case 0: // bottom edge
      case 2: // top edge
        proj.set(0, 0, 1.);
        break;
      case 1: // right edge
      case 3: // left edge
        proj.set(1, 0, 1.);
        break;
      default:
        assert(0);
    }
  }

  void compute_facet_normal (int facet_number, Coord& n) const
  {
    assert (DIM > 1);
    n.Zeros();
    switch (facet_number) 
    {
      case 0: // bottom edge
        n.set(1, -1.);
        break;
      case 1: // right edge
        n.set(0, 1.);
        break;
      case 2: // top edge
        n.set(1, 1.);
        break;
      case 3: // left edge
        n.set(0, -1.);
        break;
      default:
        assert(0);
    }
  }

  void compute_facet_tangents (int facet_number, Coord& t1, Coord& t2) const
  {
    t1.Zeros();
    t2.Zeros();
    NOT_YET_IMPLEMENTED;
  }

  Coord Tf (size_t facet_nr, const SCoord & ref_pt) const
  {
    assert (DIM == 2);
    Coord mapped_pt;
    switch (facet_nr)
    {
      case 0:
        mapped_pt.set(0, ref_pt[0]);
        break;
      case 1:
        mapped_pt.set(0, 1.);
        mapped_pt.set(1, ref_pt[0]);
        break;
      case 2:
        mapped_pt.set(0, ref_pt[0]);
        mapped_pt.set(1, 1.);
        break;
      case 3:
        mapped_pt.set(1, ref_pt[0]);
        break;
      default:
        assert(0);  
        break;
    }
    return mapped_pt;
  }

  inline DataType ds (size_t facet_nr, const SCoord & ref_pt) const
  {
    return 1.;
  }

  inline bool contains_point (const Coord &pt) const 
  {
    return (pt[0] >= -this->eps_) && (pt[0] <= 1. + this->eps_) &&
           (pt[1] >= -this->eps_) && (pt[1] <= 1. + this->eps_);
  }

  bool get_facet_nr(const Coord &pt, std::vector< int > &facet_nr) const
  {
    assert(this->contains_point(pt));
    bool found = false;
    facet_nr.clear();
    if (this->eps_ >= std::abs(pt[0])) // left edge
    {
      facet_nr.push_back(3);
      found = true;
    }
    else if (this->eps_ >= std::abs(1. - pt[0])) // right edge
    {
      facet_nr.push_back(1);
      found = true;
    }
    if (this->eps_ >= std::abs(pt[1])) // bottom edge
    {
      facet_nr.push_back(0);
      found = true;
    }
    else if (this->eps_ >= std::abs(1. - pt[1])) // top edge
    {
      facet_nr.push_back(2);
      found = true;
    }
    return found;
  }

  QuadString get_quad_name_cell_gauss (bool use_economical) const 
  {
    if (use_economical)
    {
      return "EconomicalGaussQuadrilateral";
    }
    else
    {
      return "GaussQuadrilateral";
    }  
  }

  inline QuadString get_quad_name_facet_gauss (int facet_nr, bool use_economical) const 
  {
    return "GaussLine";  
  }

  inline mesh::CellType::Tag facet_tag(int facet_nr) const
  {
    return mesh::CellType::LINE;
  }

  /// Default Destructor
  virtual ~RefCellQuadStd()
  {
  }
};

template < class DataType, int DIM > 
class RefCellTriStd final : public RefCell<DataType, DIM>
{
public:
  using Coord = typename StaticLA<DIM, DIM, DataType>::ColVectorType;
  using SCoord = typename StaticLA<DIM-1, DIM, DataType>::ColVectorType;
  using Mat = typename StaticLA<DIM, DIM-1, DataType>::MatrixType;
  /// Default Constructor
  RefCellTriStd()
  {
    this->topo_cell_ = &(mesh::CellType::get_instance(mesh::CellType::TRIANGLE));
    this->type_ = RefCellType::TRI_STD;
  }

  inline std::vector< Coord > get_coords () const
  {
    std::vector<Coord> ref_coord(3);
    ref_coord[1].set(0, 1.);
    ref_coord[2].set(1, 1.);

    return ref_coord;  
  }

  void compute_facet_projection_matrix (int facet_number, Mat& proj) const
  {
    assert (DIM > 1);
    proj.Zeros();
    switch (facet_number) 
    {
      case 0: // bottom edge
        proj.set(0, 0, 1.);
        break;
      case 1: // diagonal
        proj.set(0, 0, 1.);
        proj.set(1, 0, -1.);
        break;
      case 2: // left edge
        proj.set(1, 0, 1.);
        break;
      default:
        assert(0);
    }
  }

  void compute_facet_normal (int facet_number, Coord& n) const
  {
    assert (DIM > 1);
    n.Zeros();
    switch (facet_number) 
    {
      case 0: // bottom edge
        n.set(1, -1.);
        break;
      case 1: // diagonal
        n.set(0, 1. / std::sqrt(2.));
        n.set(1, 1. / std::sqrt(2.));
        break;
      case 2: // left edge
        n.set(0, -1.);
        break;
      default:
        assert(0);
    }
  }

  void compute_facet_tangents (int facet_number, Coord& t1, Coord& t2) const
  {
    t1.Zeros();
    t2.Zeros();
    NOT_YET_IMPLEMENTED;
  }

  Coord Tf (size_t facet_nr, const SCoord & ref_pt) const
  {
    assert (DIM == 2);
    Coord mapped_pt;
    switch (facet_nr)
    {
      case 0:
        mapped_pt.set(0, ref_pt[0]);
        break;
      case 1:
        mapped_pt.set(0, 1. - ref_pt[0]);
        mapped_pt.set(1, ref_pt[0]);
        break;
      case 2:
        mapped_pt.set(1, 1. - ref_pt[0]);
        break;
      default:
        assert(0);  
        break;
    }
    return mapped_pt;
  }

  DataType ds (size_t facet_nr, const SCoord & ref_pt) const
  {
    switch (facet_nr)
    {
      case 1:
        return std::sqrt(2.);
        break;
      default:
        return 1.;
        break;
    }
  }

  bool contains_point (const Coord &pt) const 
  {
    return pt[0] >= -this->eps_ && pt[0] <= 1. + this->eps_ &&
           pt[1] >= -this->eps_ &&
           pt[1] <= 1. - pt[0] + this->eps_;
  }

  bool get_facet_nr(const Coord &pt, std::vector< int > &facet_nr) const
  {
    assert(this->contains_point(pt));
    bool found = false;
    facet_nr.clear();
    if (this->eps_ >= std::abs(pt[0])) // left edge
    {
      facet_nr.push_back(2);
      found = true;
    }
    else if (this->eps_ >= std::abs(1. - pt[0] - pt[1])) // diagonal edge
    {
      facet_nr.push_back(1);
      found = true;
    }
    if (this->eps_ >= std::abs(pt[1])) // bottom edge
    {
      facet_nr.push_back(0);
      found = true;
    }
    return found;
  }

  inline QuadString get_quad_name_cell_gauss (bool use_economical) const 
  {
    return "GaussTriangle";
  }

  inline QuadString get_quad_name_facet_gauss (int facet_nr, bool use_economical) const 
  {
    return "GaussLine";  
  }

  inline mesh::CellType::Tag facet_tag(int facet_nr) const
  {
    return mesh::CellType::LINE;
  }

  /// Default Destructor
  virtual ~RefCellTriStd()
  {
  }
};

template < class DataType, int DIM >
class RefCellHexStd final : public RefCell<DataType, DIM>
{
 
  //                      f5 (top) 
  //                4 --------------- 7
  //               /|                /|
  //              / | f2 (left)     / |
  //             /  |              /  |                   ^ z
  //            5 --------------- 6   |                   |
  //            |   |             |   |  f4 (back)        | 
  // f1 (front) |   |             |   |                   |
  //            |   0 ------------|-- 3                   0-------->y
  //            |  /              |  /                   /
  //            | /  f3 (right)   | /                   /
  //            |/                |/                   / x
  //            1 --------------- 2                   v
  //                f0 (bottom)

public:
  using Coord = typename StaticLA<DIM, DIM, DataType>::ColVectorType;
  using SCoord = typename StaticLA<DIM-1, DIM, DataType>::ColVectorType;
  using Mat = typename StaticLA<DIM, DIM-1, DataType>::MatrixType;
  /// Default Constructor
  RefCellHexStd()
  {
    this->topo_cell_ =
&(mesh::CellType::get_instance(mesh::CellType::HEXAHEDRON));
    this->type_ = RefCellType::HEX_STD;

  }

  /// Default Destructor
  virtual ~RefCellHexStd()
  {
  }

  std::vector< Coord > get_coords () const
  {
    std::vector<Coord> ref_coord(8);

    ref_coord[1].set(0, 1.);
    ref_coord[2].set(0, 1.);
    ref_coord[2].set(1, 1.);
    ref_coord[3].set(1, 1.);
    ref_coord[4].set(2, 1.);
    ref_coord[5].set(0, 1.);
    ref_coord[5].set(2, 1.);
    ref_coord[6].set(0, 1.);
    ref_coord[6].set(1, 1.);
    ref_coord[6].set(2, 1.);
    ref_coord[7].set(1, 1.);
    ref_coord[7].set(2, 1.);

    return ref_coord; 
  }

  void compute_facet_projection_matrix (int facet_number, Mat& proj) const
  {
    assert (DIM == 3);
    proj.Zeros();
    switch (facet_number)
    {
      case 0: // bottom
      case 5: // top
        proj.set(0, 0, 1.);
        proj.set(1, 0, 0.);
        proj.set(2, 0, 0.);

        proj.set(0, 1, 0.);
        proj.set(1, 1, 1.);
        proj.set(2, 1, 0.);
        break;

      case 1: // front 1
      case 4: // back  4
        proj.set(0, 0, 1.);
        proj.set(1, 0, 0.);
        proj.set(2, 0, 0.);

        proj.set(0, 1, 0.);
        proj.set(1, 1, 0.);
        proj.set(2, 1, 1.);
        break;

      case 2: // left  2
      case 3: // right 3
        proj.set(0, 0, 0.);
        proj.set(1, 0, 1.);
        proj.set(2, 0, 0.);

        proj.set(0, 1, 0.);
        proj.set(1, 1, 0.);
        proj.set(2, 1, 1.);
        break;

      default:
        assert(0);
    }
  }

  void compute_facet_normal (int facet_number, Coord& n) const
  {
    assert (DIM == 3);
    n.Zeros();
    switch (facet_number)
    {
    case 0: 
      n.set(2, -1.);
      break;
    case 1: 
      n.set(1, -1.);
      break;
    case 2: 
      n.set(0, -1.);
      break;
    case 3: 
      n.set(0, 1.);
      break;
    case 4: 
      n.set(1, 1.);
      break;
    case 5: 
      n.set(2, 1.);
      break;
    default:
      assert(0);
    }
  }

  void compute_facet_tangents (int facet_number, Coord&t1, Coord& t2) const
  {
    t1.Zeros();
    t2.Zeros();
    NOT_YET_IMPLEMENTED;
  }

  Coord Tf (size_t facet_nr, const SCoord & ref_pt) const
  {
    assert (DIM == 3);
    Coord mapped_pt;
    switch (facet_nr)
    {
      case 0: // bottom
        mapped_pt.set(0, ref_pt[0]);
        mapped_pt.set(1, ref_pt[1]);
        break;
      case 1:// front
        mapped_pt.set(0, ref_pt[0]);
        mapped_pt.set(1, 0.);
        mapped_pt.set(2, ref_pt[1]);
        break;
      case 2:// left
        mapped_pt.set(0, 0.);
        mapped_pt.set(1, ref_pt[0]);
        mapped_pt.set(2, ref_pt[1]);
        break;
      case 3:// right
        mapped_pt.set(0, 1.);
        mapped_pt.set(1, ref_pt[0]);
        mapped_pt.set(2, ref_pt[1]);
        break;
      case 4:// back
        mapped_pt.set(0, ref_pt[0]);
        mapped_pt.set(1, 1.);
        mapped_pt.set(2, ref_pt[1]);
        break;
      case 5: // top
        mapped_pt.set(0, ref_pt[0]);
        mapped_pt.set(1, ref_pt[1]);
        mapped_pt.set(2, 1.);
        break;
      default:
        assert(0); 
        break;
    }
    return mapped_pt;
  }

  inline DataType ds (size_t facet_nr, const SCoord & ref_pt) const
  {
    return 1.;
  }

  inline bool contains_point (const Coord &pt) const
  {
    return pt[0] >= -this->eps_ && pt[0] <= 1. + this->eps_ &&
           pt[1] >= -this->eps_ && pt[1] <= 1. + this->eps_ &&
           pt[2] >= -this->eps_ && pt[2] <= 1. + this->eps_;
  }

  bool get_facet_nr(const Coord &pt, std::vector< int > &facet_nr) const
  {
    assert(this->contains_point(pt));
    bool found = false;
    facet_nr.clear();
    if (this->eps_ >= std::abs(1. - pt[1]))          // 4: back : y=1
    {
      facet_nr.push_back(4);
      found = true;
    }
    else if (this->eps_ >= std::abs(pt[1]))         // 1: front: y=0
    {
      facet_nr.push_back(1);
      found = true;
    }
    if (this->eps_ >= std::abs(pt[0]))              // 2: left: x=0
    {
      facet_nr.push_back(2);
      found = true;
    }
    else if (this->eps_ >= std::abs(1. - pt[0]))    // 3: right: x=1
    {
      facet_nr.push_back(3);
      found = true;
    }
    if (this->eps_ >= std::abs(pt[2]))             // 0: bottom: z=0
    {
      facet_nr.push_back(0);
      found = true;
    }
    else if (this->eps_ >= std::abs(1. - pt[2]))  // 5: top:  z=1
    {
      facet_nr.push_back(5);
      found = true;
    }
    return found;   
  }

  QuadString get_quad_name_cell_gauss (bool use_economical) const
  {
    if (use_economical)
    {
      return "EconomicalGaussHexahedron";
    }
    else
    {
      return "GaussHexahedron";
    } 
  }

  QuadString get_quad_name_facet_gauss (int facet_nr, bool use_economical) const
  {
    if (use_economical)
    {
      return "EconomicalGaussQuadrilateral";
    }
    else
    {
      return "GaussQuadrilateral";
    }   
  }

  inline mesh::CellType::Tag facet_tag(int facet_nr) const
  {
    return mesh::CellType::QUADRILATERAL;
  }

};

template < class DataType, int DIM > 
class RefCellTetStd final : public RefCell<DataType, DIM>
{
  using Coord = typename StaticLA<DIM, DIM, DataType>::ColVectorType;
  using SCoord = typename StaticLA<DIM-1, DIM, DataType>::ColVectorType;
  using Mat = typename StaticLA<DIM, DIM-1, DataType>::MatrixType;
public:
//     z^
//      |\
//      | \
//      |  \
//      |   \
//   f1 | f2 \
//      |_____\ ->y
//     /
//    /  f0
//   /
//  x

  /// Default Constructor
  RefCellTetStd()
  {
    this->topo_cell_ = &(mesh::CellType::get_instance(mesh::CellType::TETRAHEDRON));
    this->type_ = RefCellType::TET_STD;
  }

  /// Default Destructor
  virtual ~RefCellTetStd()
  {
  }

  inline std::vector< Coord > get_coords () const
  {
    std::vector<Coord> ref_coord(4);

    ref_coord[1].set(0, 1.);
    ref_coord[2].set(1, 1.);
    ref_coord[3].set(2, 1.);

    return ref_coord;  
  }

  void compute_facet_projection_matrix (int facet_number, Mat& proj) const
  {
    assert (DIM == 3);
    proj.Zeros();
    switch (facet_number) 
    {
      case 0: // bottom
        proj.set(0, 0, 1.);
        proj.set(1, 0, 0.);
        proj.set(2, 0, 0.);

        proj.set(0, 1, 0.);
        proj.set(1, 1, 1.);
        proj.set(2, 1, 0.);
        break;

      case 1: // front
        proj.set(0, 0, 1.);
        proj.set(1, 0, 0.);
        proj.set(2, 0, 0.);

        proj.set(0, 1, 0.);
        proj.set(1, 1, 0.);
        proj.set(2, 1, 1.);
        break;

      case 2: // left
        proj.set(0, 0, 0.);
        proj.set(1, 0, 1.);
        proj.set(2, 0, 0.);

        proj.set(0, 1, 0.);
        proj.set(1, 1, 0.);
        proj.set(2, 1, 1.);
        break;

      case 3: // back
        /*proj.set(0, 0, 1.);
        proj.set(1, 0, -1.);
        proj.set(2, 0, 0.);

        proj.set(0, 1, 0.);
        proj.set(1, 1, 1.);
        proj.set(2, 1, -1.);*/
        proj.set(0, 0, 1.);
        proj.set(1, 0, 0.);
        proj.set(2, 0, -1.);

        proj.set(0, 1, 0.);
        proj.set(1, 1, 1.);
        proj.set(2, 1, -1.);
        break;

      default:
        assert(0);
    }
  }

  void compute_facet_normal (int facet_number, Coord& n) const
  {
    assert (DIM == 3);
    n.Zeros();
    switch (facet_number) 
    {
      case 0: // bottom
        n.set(2, -1.);
        break;
      case 1: // front
        n.set(1, -1.);
        break;
      case 2: // left
        n.set(0, -1.);
        break;
      case 3: // back
        n.set(0, 1. / std::sqrt(3.));
        n.set(1, 1. / std::sqrt(3.));
        n.set(2, 1. / std::sqrt(3.));
        break;
      default:
        assert(0);
    }
  }

  void compute_facet_tangents (int facet_number, Coord& t1, Coord& t2) const
  {
    t1.Zeros();
    t2.Zeros();
    NOT_YET_IMPLEMENTED;
  }

  Coord Tf (size_t facet_nr, const SCoord & ref_pt) const
  {
    assert (DIM == 3);
    Coord mapped_pt;
    switch (facet_nr)
    {
      case 0:
        mapped_pt.set(0, ref_pt[0]);
        mapped_pt.set(1, ref_pt[1]);
        break;
      case 1:
        mapped_pt.set(0, ref_pt[0]);
        mapped_pt.set(2, ref_pt[1]);
        break;
      case 2:
        mapped_pt.set(1, ref_pt[0]);
        mapped_pt.set(2, ref_pt[1]);
        break;
      case 3:
        mapped_pt.set(0, ref_pt[0]);
        mapped_pt.set(1, ref_pt[1]);
        mapped_pt.set(2, 1. - ref_pt[0] - ref_pt[1]);
        break;
      default:
        assert(0);  
        break;
    }
    return mapped_pt;
  }

  inline DataType ds (size_t facet_nr, const SCoord & ref_pt) const
  {
    switch (facet_nr)
    {
      case 3:
        return std::sqrt(3.);
        break;
      default:
        return 1.;
        break;
    }
  }

  inline bool contains_point (const Coord &pt) const 
  {
    return pt[0] >= -this->eps_ && pt[0] <= 1. + this->eps_ &&
           pt[1] >= -this->eps_ &&
           pt[1] <= 1. - pt[0] + this->eps_ &&
           pt[2] >= -this->eps_ &&
           pt[2] <= 1. - pt[0] - pt[1] + this->eps_;
  }

  bool get_facet_nr(const Coord &pt, std::vector< int > &facet_nr) const
  {
    assert(this->contains_point(pt));
    bool found = false;
    facet_nr.clear();
    if (this->eps_ >= std::abs(pt[1])) // front
    {
      facet_nr.push_back(1);
      found = true;
    }
    if (this->eps_ >= std::abs(1. - pt[0] - pt[1] - pt[2])) // back
    {
      facet_nr.push_back(3);
      found = true;
    }
    if (this->eps_ >= std::abs(pt[0])) // left edge
    {
      facet_nr.push_back(2);
      found = true;
    }
    if (this->eps_ >= std::abs(pt[2])) // bottom edge
    {
      facet_nr.push_back(0);
      found = true;
    }
    return found;
  }

  inline QuadString get_quad_name_cell_gauss (bool use_economical) const 
  {
    return "GaussTetrahedron";
  }

  inline QuadString get_quad_name_facet_gauss (int facet_nr, bool use_economical) const 
  {
    return "GaussTriangle";  
  }

  inline mesh::CellType::Tag facet_tag(int facet_nr) const
  {
    return mesh::CellType::TRIANGLE;
  }
};

template < class DataType, int DIM > 
class RefCellPyrStd final: public RefCell<DataType, DIM>
{
public:
  using Coord = typename StaticLA<DIM, DIM, DataType>::ColVectorType;
  using SCoord = typename StaticLA<DIM-1, DIM, DataType>::ColVectorType;
  using Mat = typename StaticLA<DIM, DIM-1, DataType>::MatrixType;
  /// Default Constructor
  RefCellPyrStd()
  {
    this->topo_cell_ = &(mesh::CellType::get_instance(mesh::CellType::PYRAMID));
    this->type_ = RefCellType::PYR_STD;
  }

  /// Default Destructor
  virtual ~RefCellPyrStd()
  {
  }

  inline std::vector< Coord > get_coords () const
  {
    std::vector<Coord> ref_coord(5);

    ref_coord[1].set(0, 1.);
    ref_coord[2].set(0, 1.);
    ref_coord[2].set(1, 1.);
    ref_coord[3].set(1, 1.);
    ref_coord[4].set(0, 0.5);
    ref_coord[4].set(1, 0.5);
    ref_coord[4].set(2, 1.);

    return ref_coord;  
  }

  void compute_facet_projection_matrix (int facet_number, Mat& proj) const
  {
    assert (DIM ==3);
    proj.Zeros();
    switch (facet_number) 
    {
      case 0: // bottom
        proj.set(0, 0, 1.);
        proj.set(1, 0, 0.);
        proj.set(2, 0, 0.);

        proj.set(1, 1, 0.);
        proj.set(1, 1, 1.);
        proj.set(2, 1, 0.);
        break;

      case 1: // front
        proj.set(0, 0, 1.);
        proj.set(1, 0, 0.);
        proj.set(2, 0, 0.);

        proj.set(0, 1, 0.5);
        proj.set(1, 1, 0.5);
        proj.set(2, 1, 1.);
        break;

      case 2: // right
        proj.set(0, 0, 0.);
        proj.set(1, 0, 1.);
        proj.set(2, 0, 0.);

        proj.set(0, 1, -0.5);
        proj.set(1, 1, 0.5);
        proj.set(2, 1, 1.);
        break;

      case 3: // back
        proj.set(0, 0, -1.);
        proj.set(1, 0, 0.);
        proj.set(2, 0, 0.);

        proj.set(0, 1, -0.5);
        proj.set(1, 1, -0.);
        proj.set(2, 1, 1.);
        break;

      case 4: // left
        proj.set(0, 0, 0.);
        proj.set(1, 0, -1.);
        proj.set(2, 0, 0.);

        proj.set(0, 1, 0.5);
        proj.set(1, 1, -0.5);
        proj.set(2, 1, 1.);
        break;

      default:
        assert(0);
    }
  }

  void compute_facet_normal (int facet_number, Coord& n) const
  {
    assert (DIM == 3);
    n.Zeros();

    switch (facet_number) 
    {
      case 0: // bottom
        n.set(2, -1.);
        break;
      case 1: // front
        n.set(1, -2. / std::sqrt(3.));
        n.set(2, 1.  / std::sqrt(3.));
        break;
      case 2: // right
        n.set(0, 2. / std::sqrt(3.));
        n.set(2, 1. / std::sqrt(3.));
        break;
      case 3: // back
        n.set(1, 2. / std::sqrt(3.));
        n.set(2, 1. / std::sqrt(3.));
        break;
      case 4: // left
        n.set(0, -2. / std::sqrt(3.));
        n.set(2, 1.  / std::sqrt(3.));
        break;
      default:
        assert(0);
    }
  }

  void compute_facet_tangents (int facet_number, Coord& t1, Coord& t2) const
  {
    t1.Zeros();
    t2.Zeros();

    NOT_YET_IMPLEMENTED;
  }

  Coord Tf (size_t facet_nr, const SCoord & ref_pt) const
  {
    NOT_YET_IMPLEMENTED;
    Coord mapped_pt;
    return mapped_pt;
  }

  DataType ds (size_t facet_nr, const SCoord & ref_pt) const
  {
    NOT_YET_IMPLEMENTED;
    return 0.;
  }

  inline bool contains_point (const Coord &pt) const 
  {
    return pt[0] >= + 0.5 * pt[2] - this->eps_
       &&  pt[0] <= 1. - 0.5 * pt[2] + this->eps_
       &&  pt[1] >= 0. + 0.5 * pt[2] - this->eps_
       &&  pt[1] <= 1. - 0.5 * pt[2] + this->eps_
       &&  pt[2] >= -this->eps_ && pt[2] <= 1. + this->eps_;
  }

  inline bool get_facet_nr(const Coord &pt, std::vector< int > &facet_nr) const
  {
    return false;   
  }
  
  inline QuadString get_quad_name_cell_gauss (bool use_economical) const 
  {
    return "GaussPyramid";
  }

  inline QuadString get_quad_name_facet_gauss (int facet_nr, bool use_economical) const 
  {
    if (facet_nr == 0) 
    {
      if (use_economical)
      {
        return "EconomicalGaussQuadrilateral";
      }
      else
      {
        return "GaussQuadrilateral";
      }
    }
    else 
    {
      return "GaussTriangle";
    }  
  }

  inline mesh::CellType::Tag facet_tag(int facet_nr) const
  {
    if (facet_nr == 0)  
    {
      return mesh::CellType::QUADRILATERAL;
    }
    else
    {
      return mesh::CellType::TRIANGLE;
    }
  }
};


} // namespace doffem
} // namespace hiflow
#endif
