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

#ifndef __FEM_TRILINEAR_HEXAHEDRON_TRANSFORMATION_H_
#define __FEM_TRILINEAR_HEXAHEDRON_TRANSFORMATION_H_

#include "common/log.h"
#include "fem/cell_trafo/cell_transformation.h"
#include "fem/cell_trafo/linear_tetrahedron_transformation.h"
#include <cmath>
#include <iomanip>

namespace hiflow {
namespace doffem {

///
/// \class TriLinearHexahedronTransformation trilinear_hexahedron_transformation.h
/// \brief Trilinear transformation mapping from reference to physical cell for
/// a Hexahedron \author \author Michael Schick<br>Martin Baumann<br>Simon Gawlok<br>Philipp Gerstner
///

template < class DataType, int DIM >
class TriLinearHexahedronTransformation final
    : public CellTransformation< DataType, DIM > {
public:
  using Coord = typename CellTransformation< DataType, DIM >::Coord;
  using mat = typename CellTransformation< DataType, DIM >::mat;

  explicit TriLinearHexahedronTransformation(CRefCellSPtr<DataType, DIM> ref_cell);
  explicit TriLinearHexahedronTransformation(CRefCellSPtr<DataType, DIM> ref_cell, const std::vector< mesh::MasterSlave >& period);

  bool differs_by_translation_from (CCellTrafoSPtr<DataType, DIM> rhs) const;
   
  bool inverse_impl(const Coord& co_phy, Coord &co_ref) const;

  DataType cell_diameter() const override;

  inline int num_sub_decomposition() const 
  {
    return 5;
  }

  inline int get_subtrafo_decomposition(int t, int v) const 
  {
    assert (t>= 0);
    assert (t< 5);
    assert (v >= 0);
    assert (v < 4);
    return this->tetrahedron_decomp_ind_[t][v];
  }

  void decompose_2_ref (int t, const Coord& decomp_co_ref, Coord& co_ref) const 
  {
    switch (t) 
    {
      case 0:
        co_ref.set(0, 1. - decomp_co_ref[0]);
        co_ref.set(1, 1. - decomp_co_ref[1]);
        co_ref.set(2, decomp_co_ref[2]);
        break;
      case 1:
        co_ref.set(0, decomp_co_ref[0]);
        co_ref.set(1, decomp_co_ref[1]);
        co_ref.set(2, decomp_co_ref[2]);
        break;
      case 2:
        co_ref.set(0, 1. - decomp_co_ref[0]);
        co_ref.set(1, decomp_co_ref[1]);
        co_ref.set(2, 1. - decomp_co_ref[2]);
        break;
      case 3:
        co_ref.set(0, decomp_co_ref[0]);
        co_ref.set(1, 1. - decomp_co_ref[1]);
        co_ref.set(2, 1. - decomp_co_ref[2]);
        break;
      case 4:
        co_ref.set(0, 1. - decomp_co_ref[0] - decomp_co_ref[2]);
        co_ref.set(1, 1. - decomp_co_ref[1] - decomp_co_ref[2]);
        co_ref.set(2, 1. - decomp_co_ref[0] - decomp_co_ref[1]);
        break;
    }
  }

  inline DataType x(const Coord &coord_ref) const;
  inline DataType x_x(const Coord &coord_ref) const;
  inline DataType x_y(const Coord &coord_ref) const;
  inline DataType x_z(const Coord &coord_ref) const;
  inline DataType x_xy(const Coord &coord_ref) const;
  inline DataType x_xz(const Coord &coord_ref) const;
  inline DataType x_yz(const Coord &coord_ref) const;
  inline DataType x_xyz(const Coord &coord_ref) const;
  inline DataType y(const Coord &coord_ref) const;
  inline DataType y_x(const Coord &coord_ref) const;
  inline DataType y_y(const Coord &coord_ref) const;
  inline DataType y_z(const Coord &coord_ref) const;
  inline DataType y_xy(const Coord &coord_ref) const;
  inline DataType y_xz(const Coord &coord_ref) const;
  inline DataType y_yz(const Coord &coord_ref) const;
  inline DataType y_xyz(const Coord &coord_ref) const;
  inline DataType z(const Coord &coord_ref) const;
  inline DataType z_x(const Coord &coord_ref) const;
  inline DataType z_y(const Coord &coord_ref) const;
  inline DataType z_z(const Coord &coord_ref) const;
  inline DataType z_xy(const Coord &coord_ref) const;
  inline DataType z_xz(const Coord &coord_ref) const;
  inline DataType z_yz(const Coord &coord_ref) const;
  inline DataType z_xyz(const Coord &coord_ref) const;
  
protected:
  void init();
  bool inverse_by_decomposition(const Coord& co_phy, Coord &co_ref) const;

  int tetrahedron_decomp_ind_[5][4];
};

// Reordering of vertices to make transformation coorespond to mesh
// ordering, with (0,0,0) mapped to vertex 0, and (1,1,1) mapped to vertex 7.

template < class DataType, int DIM >
TriLinearHexahedronTransformation<DataType, DIM >::TriLinearHexahedronTransformation(CRefCellSPtr<DataType, DIM> ref_cell)
    : CellTransformation< DataType, DIM >(ref_cell) 
{
  this->init();
  this->my_valid_dim_ = 3;
}

template < class DataType, int DIM >
TriLinearHexahedronTransformation<DataType, DIM >::TriLinearHexahedronTransformation(CRefCellSPtr<DataType, DIM> ref_cell,
                                                                                     const std::vector< mesh::MasterSlave >& period)
    : CellTransformation< DataType, DIM >(ref_cell, period) 
{
  this->init();
  this->my_valid_dim_ = 3;
  this->my_nb_vertices_ = 8;
}

template < class DataType, int DIM >
DataType TriLinearHexahedronTransformation< DataType, DIM >::cell_diameter () const 
{
  DataType diag_length[4];
  
  diag_length[0] = distance(this->coord_vtx_[0], this->coord_vtx_[6]);
  diag_length[1] = distance(this->coord_vtx_[1], this->coord_vtx_[7]);
  diag_length[2] = distance(this->coord_vtx_[2], this->coord_vtx_[4]);
  diag_length[3] = distance(this->coord_vtx_[3], this->coord_vtx_[5]);
  
  DataType h = 0;
  for (int i=0; i!=4; ++i)
  {
    h = std::max(h, diag_length[i]);
  }
  return h;
}

template < class DataType, int DIM >
void TriLinearHexahedronTransformation< DataType, DIM >::init () 
{
  this->order_ = 3;  
  this->fixed_ref_cell_type_ = RefCellType::HEX_STD;
  this->name_ = "Hex";
  this->my_nb_vertices_ = 8;
   

  // define decomposition of hexahedron into 5 tetrahedrons
  //
  //        4 --------------- 7
  //       /|                /|
  //      / |               / |
  //     /  |z             /  |
  //    5 --------------- 6   |
  //    |   |             |   |
  //    |   |       y     |   |
  //    |   0 ------------|-- 3
  //    |  /              |  /
  //    | /x              | /
  //    |/                |/
  //    1 --------------- 2

  // tetrahedron 0:
  //                      6
  //                      |
  //                      |
  //                     z|   3
  //                      |  /
  //                      | /x
  //             y        |/
  //    1 --------------- 2

  tetrahedron_decomp_ind_[0][0] = 2;
  tetrahedron_decomp_ind_[0][1] = 3;
  tetrahedron_decomp_ind_[0][2] = 1;
  tetrahedron_decomp_ind_[0][3] = 6;

  // tetrahedron 1:
  //        4
  //        |
  //        |
  //        |z
  //        |
  //        |
  //        |
  //        0 --------------- 3
  //       /        y
  //     x/
  //     /
  //    1

  tetrahedron_decomp_ind_[1][0] = 0;
  tetrahedron_decomp_ind_[1][1] = 1;
  tetrahedron_decomp_ind_[1][2] = 3;
  tetrahedron_decomp_ind_[1][3] = 4;

  // tetrahedron 2:
  //        4
  //       /
  //     x/
  //     /       y
  //    5 --------------- 6
  //    |
  //    |
  //    |z
  //    |
  //    |
  //    |
  //    1

  tetrahedron_decomp_ind_[2][0] = 5;
  tetrahedron_decomp_ind_[2][1] = 4;
  tetrahedron_decomp_ind_[2][2] = 6;
  tetrahedron_decomp_ind_[2][3] = 1;

  // tetrahedron 3:
  //        4 --------------- 7
  //                y        /|
  //                       x/ |
  //                       /  |
  //                      6   |z
  //                          |
  //                          |
  //                          3

  tetrahedron_decomp_ind_[3][0] = 7;
  tetrahedron_decomp_ind_[3][1] = 6;
  tetrahedron_decomp_ind_[3][2] = 4;
  tetrahedron_decomp_ind_[3][3] = 3;

  // tetrahedron 4:
  //        4 -
  //             -
  //               z-
  //                   -
  //                      6
  //                   -    \x
  //                -        \
            //              -           3
  //            -y
  //         -
  //       -
  //    1

  tetrahedron_decomp_ind_[4][0] = 6;
  tetrahedron_decomp_ind_[4][1] = 3;
  tetrahedron_decomp_ind_[4][2] = 1;
  tetrahedron_decomp_ind_[4][3] = 4;
}

template < class DataType, int DIM >
bool TriLinearHexahedronTransformation< DataType, DIM >::differs_by_translation_from (CCellTrafoSPtr<DataType, DIM> rhs) const
{
  assert(this->ref_cell_);
  
  if (this->ref_cell_->type() != rhs->get_ref_cell()->type())
  {
    return false;
  }
  
  std::vector< Coord > ref_coords = this->ref_cell_->get_coords();
  assert (ref_coords.size() == 8);
  
  Coord my_p01 = this->transform(ref_coords[0]) - this->transform(ref_coords[1]); 
  Coord rhs_p01 = rhs->transform(ref_coords[0]) - rhs->transform(ref_coords[1]); 
  if (my_p01 != rhs_p01)
  {
    return false;
  }
  Coord my_p12 = this->transform(ref_coords[1]) - this->transform(ref_coords[2]); 
  Coord rhs_p12 = rhs->transform(ref_coords[1]) - rhs->transform(ref_coords[2]); 
  if (my_p12 != rhs_p12)
  {
    return false;
  }
  Coord my_p23 = this->transform(ref_coords[2]) - this->transform(ref_coords[3]); 
  Coord rhs_p23 = rhs->transform(ref_coords[2]) - rhs->transform(ref_coords[3]); 
  if (my_p23 != rhs_p23)
  {
    return false;
  }
  Coord my_p30 = this->transform(ref_coords[3]) - this->transform(ref_coords[0]); 
  Coord rhs_p30 = rhs->transform(ref_coords[3]) - rhs->transform(ref_coords[0]); 
  if (my_p30 != rhs_p30)
  {
    return false;
  }
  Coord my_p45 = this->transform(ref_coords[4]) - this->transform(ref_coords[5]); 
  Coord rhs_p45 = rhs->transform(ref_coords[4]) - rhs->transform(ref_coords[5]); 
  if (my_p45 != rhs_p45)
  {
    return false;
  }
  Coord my_p56 = this->transform(ref_coords[5]) - this->transform(ref_coords[6]); 
  Coord rhs_p56 = rhs->transform(ref_coords[5]) - rhs->transform(ref_coords[6]); 
  if (my_p56 != rhs_p56)
  {
    return false;
  }
  Coord my_p67 = this->transform(ref_coords[6]) - this->transform(ref_coords[7]); 
  Coord rhs_p67 = rhs->transform(ref_coords[6]) - rhs->transform(ref_coords[7]); 
  if (my_p67 != rhs_p67)
  {
    return false;
  }
  Coord my_p74 = this->transform(ref_coords[7]) - this->transform(ref_coords[4]); 
  Coord rhs_p74 = rhs->transform(ref_coords[7]) - rhs->transform(ref_coords[4]); 
  if (my_p74 != rhs_p74)
  {
    return false;
  }
  Coord my_p04 = this->transform(ref_coords[0]) - this->transform(ref_coords[4]); 
  Coord rhs_p04 = rhs->transform(ref_coords[0]) - rhs->transform(ref_coords[4]); 
  if (my_p04 != rhs_p04)
  {
    return false;
  }
  Coord my_p15 = this->transform(ref_coords[1]) - this->transform(ref_coords[5]); 
  Coord rhs_p15 = rhs->transform(ref_coords[1]) - rhs->transform(ref_coords[5]); 
  if (my_p15 != rhs_p15)
  {
    return false;
  }
  Coord my_p26 = this->transform(ref_coords[2]) - this->transform(ref_coords[6]); 
  Coord rhs_p26 = rhs->transform(ref_coords[2]) - rhs->transform(ref_coords[6]); 
  if (my_p26 != rhs_p26)
  {
    return false;
  }
  Coord my_p37 = this->transform(ref_coords[3]) - this->transform(ref_coords[7]); 
  Coord rhs_p37 = rhs->transform(ref_coords[3]) - rhs->transform(ref_coords[7]); 
  if (my_p37 != rhs_p37)
  {
    return false;
  }
  return true;
}

template < class DataType, int DIM >
bool TriLinearHexahedronTransformation< DataType, DIM >::inverse_by_decomposition(const Coord& co_phy, Coord &co_ref) const 
{
  RefCellSPtr<DataType, DIM> ref_cell_tet = RefCellSPtr<DataType, DIM>(new RefCellTetStd<DataType, DIM> );
  LinearTetrahedronTransformation< DataType, DIM > tetra_trafo(ref_cell_tet);

  bool found_pt_in_tetra = inverse_transformation_decomposition
                            <DataType, DIM, DIM, TriLinearHexahedronTransformation< DataType, DIM >, 
                             LinearTetrahedronTransformation< DataType, DIM > >
                              (this, &tetra_trafo, co_phy, co_ref);

#ifdef NEW_INVERSE
  return found_pt_in_tetra;
#else 
  // old impl
  if (found_pt_in_tetra)
  {
    return this->residual_inverse_check(co_phy, co_ref);
  }
  else
  {
    return false;
  }
#endif
}

template < class DataType, int DIM >
bool TriLinearHexahedronTransformation< DataType, DIM >::inverse_impl(const Coord& co_phy, Coord &co_ref) const 
{
  
  // if true: point is contained in sub-tet
  bool found_pt_in_tet = this->inverse_by_decomposition(co_phy, co_ref);

  // if true: reference point obtained by decomposition is sufficiently accurate
  bool passed_residual_check = this->residual_inverse_check(co_phy, co_ref);

  if (found_pt_in_tet && passed_residual_check) 
  {
    return true;
  } 
  else 
  {
#ifdef NEW_INVERSE
    // somehow only works in 2D so far
    // new impl
    if (!found_pt_in_tet)
    {
      // if point is not contained in sub-tet, it is also not contained in quad
      // -> no need to call newton
      return false;
    }
#endif
    // point is contained in quad but
    // reference point obtained by decomposition is not accurate enough
    // -> use as initial value for Newton method
    bool newton_success = inverse_transformation_newton<DataType, DIM, DIM, TriLinearHexahedronTransformation<DataType, DIM> >
                            (this, co_phy, co_ref, co_ref);
    return newton_success;
  }
}

template < class DataType, int DIM >
DataType
TriLinearHexahedronTransformation< DataType, DIM >::x(const Coord &coord_ref) const {
  
  const DataType coord_0 = coord_ref[0];
  const DataType coord_1 = coord_ref[1];
  const DataType coord_2 = coord_ref[2];

  return this->coord_vtx_[0][0] * (1. - coord_0)  * (1. - coord_1)  * (1. - coord_2) +
         this->coord_vtx_[1][0] * coord_0         * (1. - coord_1)  * (1. - coord_2) +
         this->coord_vtx_[2][0] * coord_0         * coord_1         * (1. - coord_2) +
         this->coord_vtx_[3][0] * (1. - coord_0)  * coord_1         * (1. - coord_2) +
         this->coord_vtx_[4][0] * (1. - coord_0)  * (1. - coord_1)  * coord_2 +
         this->coord_vtx_[5][0] * coord_0         * (1. - coord_1)  * coord_2 +
         this->coord_vtx_[6][0] * coord_0         * coord_1         * coord_2 +
         this->coord_vtx_[7][0] * (1. - coord_0)  * coord_1         * coord_2;
}

template < class DataType, int DIM >
DataType TriLinearHexahedronTransformation< DataType, DIM >::x_x( const Coord &coord_ref) const {
  
  const DataType coord_1 = coord_ref[1];
  const DataType coord_2 = coord_ref[2];

  return -this->coord_vtx_[0][0] * (1. - coord_1) * (1. - coord_2) +
          this->coord_vtx_[1][0] * (1. - coord_1) * (1. - coord_2) +
          this->coord_vtx_[2][0] * coord_1        * (1. - coord_2) -
          this->coord_vtx_[3][0] * coord_1        * (1. - coord_2) -
          this->coord_vtx_[4][0] * (1. - coord_1) * coord_2 +
          this->coord_vtx_[5][0] * (1. - coord_1) * coord_2 +
          this->coord_vtx_[6][0] * coord_1        * coord_2 -
          this->coord_vtx_[7][0] * coord_1        * coord_2;
}

template < class DataType, int DIM >
DataType TriLinearHexahedronTransformation< DataType, DIM >::x_y( const Coord &coord_ref) const {
  
  const DataType coord_0 = coord_ref[0];
  const DataType coord_2 = coord_ref[2];

  return -this->coord_vtx_[0][0] * (1. - coord_0) * (1. - coord_2) -
          this->coord_vtx_[1][0] * coord_0        * (1. - coord_2) +
          this->coord_vtx_[2][0] * coord_0        * (1. - coord_2) +
          this->coord_vtx_[3][0] * (1. - coord_0) * (1. - coord_2) -
          this->coord_vtx_[4][0] * (1. - coord_0) * coord_2 -
          this->coord_vtx_[5][0] * coord_0        * coord_2 +
          this->coord_vtx_[6][0] * coord_0        * coord_2 +
          this->coord_vtx_[7][0] * (1. - coord_0) * coord_2;
}

template < class DataType, int DIM >
DataType TriLinearHexahedronTransformation< DataType, DIM >::x_z( const Coord &coord_ref) const {
  
  const DataType coord_0 = coord_ref[0];
  const DataType coord_1 = coord_ref[1];

  return -this->coord_vtx_[0][0] * (1. - coord_0) * (1. - coord_1) -
          this->coord_vtx_[1][0] * coord_0        * (1. - coord_1) -
          this->coord_vtx_[2][0] * coord_0        * coord_1 -
          this->coord_vtx_[3][0] * (1. - coord_0) * coord_1 +
          this->coord_vtx_[4][0] * (1. - coord_0) * (1. - coord_1) +
          this->coord_vtx_[5][0] * coord_0        * (1. - coord_1) +
          this->coord_vtx_[6][0] * coord_0        * coord_1 +
          this->coord_vtx_[7][0] * (1. - coord_0) * coord_1;
}

template < class DataType, int DIM >
DataType TriLinearHexahedronTransformation< DataType, DIM >::x_xy( const Coord &coord_ref) const {
  
  const DataType coord_2 = coord_ref[2];

  return this->coord_vtx_[0][0] * (1. - coord_2) -
         this->coord_vtx_[1][0] * (1. - coord_2) +
         this->coord_vtx_[2][0] * (1. - coord_2) -
         this->coord_vtx_[3][0] * (1. - coord_2) +
         this->coord_vtx_[4][0] * coord_2 -
         this->coord_vtx_[5][0] * coord_2 +
         this->coord_vtx_[6][0] * coord_2 -
         this->coord_vtx_[7][0] * coord_2;
}

template < class DataType, int DIM >
DataType TriLinearHexahedronTransformation< DataType, DIM >::x_xz( const Coord &coord_ref) const {
  
  const DataType coord_1 = coord_ref[1];

  return +this->coord_vtx_[0][0] * (1. - coord_1) -
         this->coord_vtx_[1][0] * (1. - coord_1) -
         this->coord_vtx_[2][0] * coord_1 +
         this->coord_vtx_[3][0] * coord_1 -
         this->coord_vtx_[4][0] * (1. - coord_1) *
             +this->coord_vtx_[5][0] * (1. - coord_1) *
             +this->coord_vtx_[6][0] * coord_1 -
         this->coord_vtx_[7][0] * coord_1;
}

template < class DataType, int DIM >
DataType TriLinearHexahedronTransformation< DataType, DIM >::x_yz( const Coord &coord_ref) const {
  
  const DataType coord_0 = coord_ref[0];

  return +this->coord_vtx_[0][0] * (1. - coord_0) +
         this->coord_vtx_[1][0] * coord_0 -
         this->coord_vtx_[2][0] * coord_0 -
         this->coord_vtx_[3][0] * (1. - coord_0) -
         this->coord_vtx_[4][0] * (1. - coord_0) -
         this->coord_vtx_[5][0] * coord_0 +
         this->coord_vtx_[6][0] * coord_0 +
         this->coord_vtx_[7][0] * (1. - coord_0);
}

template < class DataType, int DIM >
DataType TriLinearHexahedronTransformation< DataType, DIM >::x_xyz( const Coord &coord_ref) const {
  

  return - this->coord_vtx_[0][0]
         + this->coord_vtx_[1][0] 
         - this->coord_vtx_[2][0] 
         + this->coord_vtx_[3][0] 
         + this->coord_vtx_[4][0] 
         - this->coord_vtx_[5][0] 
         + this->coord_vtx_[6][0]
         - this->coord_vtx_[7][0];
}

template < class DataType, int DIM >
DataType
TriLinearHexahedronTransformation< DataType, DIM >::y(const Coord &coord_ref) const {
  
  const DataType coord_0 = coord_ref[0];
  const DataType coord_1 = coord_ref[1];
  const DataType coord_2 = coord_ref[2];

  return this->coord_vtx_[0][1] * (1. - coord_0) * (1. - coord_1) * (1. - coord_2) +
         this->coord_vtx_[1][1] * coord_0        * (1. - coord_1) * (1. - coord_2) +
         this->coord_vtx_[2][1] * coord_0        * coord_1        * (1. - coord_2) +
         this->coord_vtx_[3][1] * (1. - coord_0) * coord_1        * (1. - coord_2) +
         this->coord_vtx_[4][1] * (1. - coord_0) * (1. - coord_1) * coord_2 +
         this->coord_vtx_[5][1] * coord_0        * (1. - coord_1) * coord_2 +
         this->coord_vtx_[6][1] * coord_0        * coord_1        * coord_2 +
         this->coord_vtx_[7][1] * (1. - coord_0) * coord_1        * coord_2;
}

template < class DataType, int DIM >
DataType TriLinearHexahedronTransformation< DataType, DIM >::y_x( const Coord &coord_ref) const {
  
  const DataType coord_1 = coord_ref[1];
  const DataType coord_2 = coord_ref[2];

  return -this->coord_vtx_[0][1] * (1. - coord_1) * (1. - coord_2) +
          this->coord_vtx_[1][1] * (1. - coord_1) * (1. - coord_2) +
          this->coord_vtx_[2][1] * coord_1        * (1. - coord_2) -
          this->coord_vtx_[3][1] * coord_1        * (1. - coord_2) -
          this->coord_vtx_[4][1] * (1. - coord_1) * coord_2 +
          this->coord_vtx_[5][1] * (1. - coord_1) * coord_2 +
          this->coord_vtx_[6][1] * coord_1        * coord_2 -
          this->coord_vtx_[7][1] * coord_1        * coord_2;
}

template < class DataType, int DIM >
DataType TriLinearHexahedronTransformation< DataType, DIM >::y_y( const Coord &coord_ref) const {
  
  const DataType coord_0 = coord_ref[0];
  const DataType coord_2 = coord_ref[2];

  return -this->coord_vtx_[0][1] * (1. - coord_0) * (1. - coord_2) -
          this->coord_vtx_[1][1] * coord_0        * (1. - coord_2) +
          this->coord_vtx_[2][1] * coord_0        * (1. - coord_2) +
          this->coord_vtx_[3][1] * (1. - coord_0) * (1. - coord_2) -
          this->coord_vtx_[4][1] * (1. - coord_0) * coord_2 -
          this->coord_vtx_[5][1] * coord_0        * coord_2 +
          this->coord_vtx_[6][1] * coord_0        * coord_2 +
          this->coord_vtx_[7][1] * (1. - coord_0) * coord_2;
}

template < class DataType, int DIM >
DataType TriLinearHexahedronTransformation< DataType, DIM >::y_z( const Coord &coord_ref) const {
  
  const DataType coord_0 = coord_ref[0];
  const DataType coord_1 = coord_ref[1];

  return -this->coord_vtx_[0][1] * (1. - coord_0) * (1. - coord_1) -
          this->coord_vtx_[1][1] * coord_0        * (1. - coord_1) -
          this->coord_vtx_[2][1] * coord_0        * coord_1 -
          this->coord_vtx_[3][1] * (1. - coord_0) * coord_1 +
          this->coord_vtx_[4][1] * (1. - coord_0) * (1. - coord_1) +
          this->coord_vtx_[5][1] * coord_0        * (1. - coord_1) +
          this->coord_vtx_[6][1] * coord_0        * coord_1 +
          this->coord_vtx_[7][1] * (1. - coord_0) * coord_1;
}

template < class DataType, int DIM >
DataType TriLinearHexahedronTransformation< DataType, DIM >::y_xy( const Coord &coord_ref) const {
  
  const DataType coord_2 = coord_ref[2];

  return +this->coord_vtx_[0][1] * (1. - coord_2) -
         this->coord_vtx_[1][1] * (1. - coord_2) +
         this->coord_vtx_[2][1] * (1. - coord_2) -
         this->coord_vtx_[3][1] * (1. - coord_2) +
         this->coord_vtx_[4][1] * coord_2 -
         this->coord_vtx_[5][1] * coord_2 +
         this->coord_vtx_[6][1] * coord_2 -
         this->coord_vtx_[7][1] * coord_2;
}

template < class DataType, int DIM >
DataType TriLinearHexahedronTransformation< DataType, DIM >::y_xz( const Coord &coord_ref) const {
  
  const DataType coord_1 = coord_ref[1];

  return +this->coord_vtx_[0][1] * (1. - coord_1) -
         this->coord_vtx_[1][1] * (1. - coord_1) -
         this->coord_vtx_[2][1] * coord_1 +
         this->coord_vtx_[3][1] * coord_1 -
         this->coord_vtx_[4][1] * (1. - coord_1) +
         this->coord_vtx_[5][1] * (1. - coord_1) +
         this->coord_vtx_[6][1] * coord_1 -
         this->coord_vtx_[7][1] * coord_1;
}

template < class DataType, int DIM >
DataType TriLinearHexahedronTransformation< DataType, DIM >::y_yz( const Coord &coord_ref) const {
  
  const DataType coord_0 = coord_ref[0];

  return +this->coord_vtx_[0][1] * (1. - coord_0) +
         this->coord_vtx_[1][1] * coord_0 -
         this->coord_vtx_[2][1] * coord_0 -
         this->coord_vtx_[3][1] * (1. - coord_0) -
         this->coord_vtx_[4][1] * (1. - coord_0) -
         this->coord_vtx_[5][1] * coord_0 +
         this->coord_vtx_[6][1] * coord_0 +
         this->coord_vtx_[7][1] * (1. - coord_0);
}

template < class DataType, int DIM >
DataType TriLinearHexahedronTransformation< DataType, DIM >::y_xyz( const Coord &coord_ref) const {
  

  return - this->coord_vtx_[0][1]
         + this->coord_vtx_[1][1] 
         - this->coord_vtx_[2][1] 
         + this->coord_vtx_[3][1] 
         + this->coord_vtx_[4][1] 
         - this->coord_vtx_[5][1] 
         + this->coord_vtx_[6][1]
         - this->coord_vtx_[7][1];
}

template < class DataType, int DIM >
DataType
TriLinearHexahedronTransformation< DataType, DIM >::z(const Coord &coord_ref) const {
  
  const DataType coord_0 = coord_ref[0];
  const DataType coord_1 = coord_ref[1];
  const DataType coord_2 = coord_ref[2];

  return this->coord_vtx_[0][2] * (1. - coord_0)  * (1. - coord_1)  * (1. - coord_2) +
         this->coord_vtx_[1][2] * coord_0         * (1. - coord_1)  * (1. - coord_2) +
         this->coord_vtx_[2][2] * coord_0         * coord_1         * (1. - coord_2) +
         this->coord_vtx_[3][2] * (1. - coord_0)  * coord_1         * (1. - coord_2) +
         this->coord_vtx_[4][2] * (1. - coord_0)  * (1. - coord_1)  * coord_2 +
         this->coord_vtx_[5][2] * coord_0         * (1. - coord_1)  * coord_2 +
         this->coord_vtx_[6][2] * coord_0         * coord_1         * coord_2 +
         this->coord_vtx_[7][2] * (1. - coord_0)  * coord_1         * coord_2;
}

template < class DataType, int DIM >
DataType TriLinearHexahedronTransformation< DataType, DIM >::z_x( const Coord &coord_ref) const {
  
  const DataType coord_1 = coord_ref[1];
  const DataType coord_2 = coord_ref[2];

  return -this->coord_vtx_[0][2] * (1. - coord_1) * (1. - coord_2) +
          this->coord_vtx_[1][2] * (1. - coord_1) * (1. - coord_2) +
          this->coord_vtx_[2][2] * coord_1        * (1. - coord_2) -
          this->coord_vtx_[3][2] * coord_1        * (1. - coord_2) -
          this->coord_vtx_[4][2] * (1. - coord_1) * coord_2 +
          this->coord_vtx_[5][2] * (1. - coord_1) * coord_2 +
          this->coord_vtx_[6][2] * coord_1        * coord_2 -
          this->coord_vtx_[7][2] * coord_1        * coord_2;
}

template < class DataType, int DIM >
DataType TriLinearHexahedronTransformation< DataType, DIM >::z_y(  const Coord &coord_ref) const {
  
  const DataType coord_0 = coord_ref[0];
  const DataType coord_2 = coord_ref[2];

  return -this->coord_vtx_[0][2] * (1. - coord_0) * (1. - coord_2) -
          this->coord_vtx_[1][2] * coord_0        * (1. - coord_2) +
          this->coord_vtx_[2][2] * coord_0        * (1. - coord_2) +
          this->coord_vtx_[3][2] * (1. - coord_0) * (1. - coord_2) -
          this->coord_vtx_[4][2] * (1. - coord_0) * coord_2 -
          this->coord_vtx_[5][2] * coord_0        * coord_2 +
          this->coord_vtx_[6][2] * coord_0        * coord_2 +
          this->coord_vtx_[7][2] * (1. - coord_0) * coord_2;
}

template < class DataType, int DIM >
DataType TriLinearHexahedronTransformation< DataType, DIM >::z_z( const Coord &coord_ref) const {
  
  const DataType coord_0 = coord_ref[0];
  const DataType coord_1 = coord_ref[1];

  return -this->coord_vtx_[0][2] * (1. - coord_0) * (1. - coord_1) -
          this->coord_vtx_[1][2] * coord_0        * (1. - coord_1) -
          this->coord_vtx_[2][2] * coord_0        * coord_1 -
          this->coord_vtx_[3][2] * (1. - coord_0) * coord_1 +
          this->coord_vtx_[4][2] * (1. - coord_0) * (1. - coord_1) +
          this->coord_vtx_[5][2] * coord_0        * (1. - coord_1) +
          this->coord_vtx_[6][2] * coord_0        * coord_1 +
          this->coord_vtx_[7][2] * (1. - coord_0) * coord_1;
}

template < class DataType, int DIM >
DataType TriLinearHexahedronTransformation< DataType, DIM >::z_xy( const Coord &coord_ref) const {
  
  const DataType coord_2 = coord_ref[2];

  return +this->coord_vtx_[0][2] * (1. - coord_2) -
         this->coord_vtx_[1][2] * (1. - coord_2) +
         this->coord_vtx_[2][2] * (1. - coord_2) -
         this->coord_vtx_[3][2] * (1. - coord_2) +
         this->coord_vtx_[4][2] * coord_2 -
         this->coord_vtx_[5][2] * coord_2 +
         this->coord_vtx_[6][2] * coord_2 -
         this->coord_vtx_[7][2] * coord_2;
}

template < class DataType, int DIM >
DataType TriLinearHexahedronTransformation< DataType, DIM >::z_xz( const Coord &coord_ref) const {
  
  const DataType coord_1 = coord_ref[1];

  return +this->coord_vtx_[0][2] * (1. - coord_1) -
         this->coord_vtx_[1][2] * (1. - coord_1) -
         this->coord_vtx_[2][2] * coord_1 +
         this->coord_vtx_[3][2] * coord_1 -
         this->coord_vtx_[4][2] * (1. - coord_1) +
         this->coord_vtx_[5][2] * (1. - coord_1) +
         this->coord_vtx_[6][2] * coord_1 -
         this->coord_vtx_[7][2] * coord_1;
}

template < class DataType, int DIM >
DataType TriLinearHexahedronTransformation< DataType, DIM >::z_yz( const Coord &coord_ref) const {
  
  const DataType coord_0 = coord_ref[0];

  return +this->coord_vtx_[0][2] * (1. - coord_0) +
         this->coord_vtx_[1][2] * coord_0 -
         this->coord_vtx_[2][2] * coord_0 -
         this->coord_vtx_[3][2] * (1. - coord_0) -
         this->coord_vtx_[4][2] * (1. - coord_0) -
         this->coord_vtx_[5][2] * coord_0 +
         this->coord_vtx_[6][2] * coord_0 +
         this->coord_vtx_[7][2] * (1. - coord_0);
}

template < class DataType, int DIM >
DataType TriLinearHexahedronTransformation< DataType, DIM >::z_xyz( const Coord &coord_ref) const {
  

  return - this->coord_vtx_[0][2]
         + this->coord_vtx_[1][2] 
         - this->coord_vtx_[2][2] 
         + this->coord_vtx_[3][2] 
         + this->coord_vtx_[4][2] 
         - this->coord_vtx_[5][2] 
         + this->coord_vtx_[6][2]
         - this->coord_vtx_[7][2];
}

} // namespace doffem
} // namespace hiflow

#endif
