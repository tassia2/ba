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

#ifndef __FEM_CELL_TRANSFORMATION_H_
#define __FEM_CELL_TRANSFORMATION_H_

#include <cassert>
#include <vector>
#include <limits>
#include <cmath>

#include "common/log.h"
#include "common/macros.h"
#include "common/vector_algebra_descriptor.h"
#include "common/static_string.h"
#include "dof/dof_fem_types.h"
#include "fem/reference_cell.h"
#include "fem/cell_trafo/cell_trafo_inverse.h"
#include "mesh/periodicity_tools.h"
#include "mesh/entity.h"
#include "mesh/iterator.h"
#include "mesh/geometric_tools.h"

#define NEW_INVERSE

namespace hiflow {
namespace doffem {

enum CellTrafoType
{
  CELL_TRAFO_NONE = -1,
  CELL_TRAFO_LINEARLINE = 0,
  CELL_TRAFO_LINEARTRI = 1,
  CELL_TRAFO_LINEARTET = 2,
  CELL_TRAFO_LINEARPYR = 3,
  CELL_TRAFO_BILINEARQUAD = 4,
  CELL_TRAFO_TRILINEARHEX = 5,
  CELL_TRAFO_ALIGNEDQUAD = 6,
  CELL_TRAFO_ALIGNEDHEX = 7
};

using TrafoString = StaticString<50>;

///
/// \class CellTransformation cell_transformation.h
/// \brief Ancestor class of all transformation mappings from reference to
/// physical cells 
/// \author Michael Schick<br>Martin Baumann<br>Simon Gawlok<br>Philipp Gerstner
///

template < class DataType, int DIM > 
class CellTransformation 
{
public:
  static constexpr double INVERSE_RESIDUAL_TOL = 1000. * std::numeric_limits< double >::epsilon();
  using Coord = typename StaticLA<DIM, DIM, DataType>::ColVectorType;
  using mat = typename StaticLA<DIM, DIM, DataType>::MatrixType;

  /// Use this constructor which needs the geometrical dimension as input
  explicit CellTransformation(CRefCellSPtr<DataType, DIM> ref_cell);
  
  explicit CellTransformation(CRefCellSPtr<DataType, DIM> ref_cell,
                              const std::vector< mesh::MasterSlave >& period);

  virtual ~CellTransformation() 
  {
  }

  TrafoString name() const 
  {
    return this->name_;
  }
  
  virtual bool differs_by_translation_from (CCellTrafoSPtr<DataType, DIM> rhs) const = 0;
  
  /// Reinitialization of the transformation via coordinates of physical cell
  virtual void reinit(const std::vector<Coord >&coord_vtx);
  virtual void reinit(const std::vector<DataType> &coord_vtx);
  virtual void reinit(const std::vector<DataType> &coord_vtx, const mesh::Entity& cell);

  CRefCellSPtr<DataType, DIM> get_ref_cell() const
  {
    return this->ref_cell_;
  }
  
  std::vector< Coord > get_reference_coordinates() const 
  { 
    assert(this->ref_cell_);
    return this->ref_cell_->get_coords();
  }

  std::vector<Coord> get_coordinates() const 
  { 
    return this->coord_vtx_;
  }

  void get_coordinates(std::vector<Coord>& coords) const 
  { 
    coords = this->coord_vtx_;
  }

  Coord get_coordinate(int vertex_id) const 
  {
    assert (vertex_id >= 0);
    assert (vertex_id < this->coord_vtx_.size());
    return this->coord_vtx_[vertex_id];
  }

  inline int num_vertices() const 
  {
    return my_nb_vertices_;
  }

  void print_vertex_coords() const 
  {
    for (size_t l=0; l<this->coord_vtx_.size(); ++l)
    {
      std::cout << "[" << l << "]: " << this->coord_vtx_[l] << std::endl;
    }
    std::cout << std::endl;
  }

  inline int phys2ref_facet_nr(int i) const
  {
    assert(i >= 0);
    assert(i < phys2ref_facet_nr_.size());
    return phys2ref_facet_nr_[i];
  }

  inline int ref2phys_facet_nr(int i) const
  {
    assert(i >= 0);
    assert(i < ref2phys_facet_nr_.size());
    return ref2phys_facet_nr_[i];
  }

  /// \brief Check whether a given point is contained in the closure of the
  /// cell.
  ///
  /// \param[in] coord_ref    reference coordinates of the point
  /// \returns  True if reference coordinates are contained in the cell.
  inline bool contains_reference_point(const Coord &coord_ref) const
  {
    assert(this->ref_cell_);
    return this->ref_cell_->contains_point (coord_ref);
  }

  /// \brief Check whether a given point is contained in the closure of the
  /// cell.
  /// \param[in]  coord_phys   Physical coordinates of the point.
  /// \param[out] coord_ref    Optional output parameter, where reference
  /// coordinates in the cell are stored, if the function returns true. \returns
  /// True if reference coordinates are contained in the cell.
  virtual bool contains_physical_point(const Coord &coord_phys, Coord &coord_ref) const;

  virtual DataType cell_diameter() const;

  void transform ( const Coord& coord_ref, Coord& coord_mapped ) const;
  
  Coord transform ( const Coord& coord_ref ) const
  {
    Coord coord_mapped;
    this->transform(coord_ref, coord_mapped);
    return coord_mapped;
  }

  bool residual_inverse_check (const Coord &co_phy, const Coord &co_ref ) const;
  DataType residual_inverse (const Coord &co_phy, const Coord &co_ref ) const;
  
  void J (const Coord &coord_ref, mat& J) const;

  void H (const Coord &coord_ref, size_t d, mat& H) const;

  /// \brief compute determinant of Jacobian of cell transformation at given reference coordinates
  DataType detJ (const Coord &coord_ref) const;

  void J_and_detJ (const Coord &coord_ref, mat& J, DataType& detJ) const;
  
  /// \brief compute gradient of determinant of Jacobian of cell transformation at given reference coordinates
  void grad_detJ (const Coord &coord_ref, Coord & grad) const;
  void grad_inv_detJ (const Coord &coord_ref, Coord & grad) const;
  
  DataType detJ_x (const Coord &coord_ref) const;
  DataType detJ_y (const Coord &coord_ref) const;
  DataType detJ_z (const Coord &coord_ref) const;
  
  /// \brief compute hessian ofdeterminant of Jacobian of cell transformation at given reference coordinates
  void hessian_detJ (const Coord &coord_ref, mat & mat) const;
  DataType detJ_xx (const Coord &coord_ref) const;
  DataType detJ_xy (const Coord &coord_ref) const;
  DataType detJ_xz (const Coord &coord_ref) const;
  DataType detJ_yy (const Coord &coord_ref) const;
  DataType detJ_yz (const Coord &coord_ref) const;          
  DataType detJ_zz (const Coord &coord_ref) const;
  
  /// \brief Given physical cell coordinates in 1D,
  ///        this routine computes the corresponding reference cell coordinates
  /// @return true, if inverse computation was successful
  virtual bool inverse_impl(const Coord& co_phy, Coord &co_ref) const = 0;
  bool inverse (const Coord& co_phy, Coord &co_ref) const;

  /// Given reference coordinates, this routine computes the physical x
  /// coordinates
  virtual DataType x(const Coord &coord_ref) const { return 0.; }
  
  /// \brief Given reference coordinates, this routine computes the derivatives
  /// of the mapping (ref_coordinates to physical x value)
  virtual DataType x_x(const Coord &coord_ref) const { return 0.; }
  virtual DataType x_y(const Coord &coord_ref) const { return 0.; }
  virtual DataType x_z(const Coord &coord_ref) const { return 0.; }
  
  /// \brief Given reference coordinates, these routine compute the second
  /// derivatives of the mapping (ref_coordinates to physical x value). 
  virtual DataType x_xx(const Coord &coord_ref) const { return 0.; }
  virtual DataType x_xy(const Coord &coord_ref) const { return 0.; }
  virtual DataType x_xz(const Coord &coord_ref) const { return 0.; }
  virtual DataType x_yy(const Coord &coord_ref) const { return 0.; }
  virtual DataType x_yz(const Coord &coord_ref) const { return 0.; }
  virtual DataType x_zz(const Coord &coord_ref) const { return 0.; }

  /// \brief Given reference coordinates, these routine compute the third
  /// derivatives of the mapping (ref_coordinates to physical x value). 
  virtual DataType x_xxx(const Coord &coord_ref) const { return 0.; }
  virtual DataType x_xxy(const Coord &coord_ref) const { return 0.; }
  virtual DataType x_xxz(const Coord &coord_ref) const { return 0.; }
  virtual DataType x_xyy(const Coord &coord_ref) const { return 0.; }
  virtual DataType x_xyz(const Coord &coord_ref) const { return 0.; }
  virtual DataType x_xzz(const Coord &coord_ref) const { return 0.; }
  virtual DataType x_yyy(const Coord &coord_ref) const { return 0.; }
  virtual DataType x_yyz(const Coord &coord_ref) const { return 0.; }
  virtual DataType x_yzz(const Coord &coord_ref) const { return 0.; }
  virtual DataType x_zzz(const Coord &coord_ref) const { return 0.; }
  
  /// Given reference coordinates, this computes the physical y coordinates
  virtual DataType y(const Coord &coord_ref) const { return 0.; }
  
  /// \brief Given reference coordinates, this routine computes the derivatives
  /// of the mapping (ref_coordinates to physical y value)
  virtual DataType y_x(const Coord &coord_ref) const { return 0.; }
  virtual DataType y_y(const Coord &coord_ref) const { return 0.; }
  virtual DataType y_z(const Coord &coord_ref) const { return 0.; }

  /// \brief Given reference coordinates, this routine computes the second
  /// derivatives of the mapping (ref_coordinates to physical y value)
  virtual DataType y_xx(const Coord &coord_ref) const { return 0.; }
  virtual DataType y_xy(const Coord &coord_ref) const { return 0.; }
  virtual DataType y_xz(const Coord &coord_ref) const { return 0.; }
  virtual DataType y_yy(const Coord &coord_ref) const { return 0.; }
  virtual DataType y_yz(const Coord &coord_ref) const { return 0.; }
  virtual DataType y_zz(const Coord &coord_ref) const { return 0.; }

  /// \brief Given reference coordinates, these routine compute the third
  /// derivatives of the mapping (ref_coordinates to physical y value). 
  virtual DataType y_xxx(const Coord &coord_ref) const { return 0.; }
  virtual DataType y_xxy(const Coord &coord_ref) const { return 0.; }
  virtual DataType y_xxz(const Coord &coord_ref) const { return 0.; }
  virtual DataType y_xyy(const Coord &coord_ref) const { return 0.; }
  virtual DataType y_xyz(const Coord &coord_ref) const { return 0.; }
  virtual DataType y_xzz(const Coord &coord_ref) const { return 0.; }
  virtual DataType y_yyy(const Coord &coord_ref) const { return 0.; }
  virtual DataType y_yyz(const Coord &coord_ref) const { return 0.; }
  virtual DataType y_yzz(const Coord &coord_ref) const { return 0.; }
  virtual DataType y_zzz(const Coord &coord_ref) const { return 0.; }

  /// Given reference coordinates, this computes the physical z coordinates
  virtual DataType z(const Coord &coord_ref) const { return 0.; }
  /// \brief Given reference coordinates, this routine computes the derivatives
  /// in of the mapping (ref_coordinates to physical z value).
  /// Return value is a dummy value for 2D problems, but neccessary for UnitIntegrator

  /// \brief Given reference coordinates, this routine computes the derivatives
  /// of the mapping (ref_coordinates to physical z value).
  /// Return value is a dummy value for 2D problems, but neccessary for
  /// UnitIntegrator
  virtual DataType z_x(const Coord &coord_ref) const { return 0.; }
  virtual DataType z_y(const Coord &coord_ref) const { return 0.; }
  virtual DataType z_z(const Coord &coord_ref) const { return 0.; }

  /// \brief Given reference coordinates, this routine computes the second
  /// derivatives  of the mapping (ref_coordinates to physical z
  /// value). Return value is a dummy value for 2D problems, but
  /// neccessary for UnitIntegrator
  virtual DataType z_xx(const Coord &coord_ref) const { return 0.; }
  virtual DataType z_xy(const Coord &coord_ref) const { return 0.; }
  virtual DataType z_xz(const Coord &coord_ref) const { return 0.; }
  virtual DataType z_yy(const Coord &coord_ref) const { return 0.; }
  virtual DataType z_yz(const Coord &coord_ref) const { return 0.; }
  virtual DataType z_zz(const Coord &coord_ref) const { return 0.; }

  /// \brief Given reference coordinates, these routine compute the third
  /// derivatives of the mapping (ref_coordinates to physical x value). 
  virtual DataType z_xxx(const Coord &coord_ref) const { return 0.; }
  virtual DataType z_xxy(const Coord &coord_ref) const { return 0.; }
  virtual DataType z_xxz(const Coord &coord_ref) const { return 0.; }
  virtual DataType z_xyy(const Coord &coord_ref) const { return 0.; }
  virtual DataType z_xyz(const Coord &coord_ref) const { return 0.; }
  virtual DataType z_xzz(const Coord &coord_ref) const { return 0.; }
  virtual DataType z_yyy(const Coord &coord_ref) const { return 0.; }
  virtual DataType z_yyz(const Coord &coord_ref) const { return 0.; }
  virtual DataType z_yzz(const Coord &coord_ref) const { return 0.; }
  virtual DataType z_zzz(const Coord &coord_ref) const { return 0.; }

protected:
  int my_valid_dim_ = -1;
  int my_nb_vertices_ = -1;

  /// \details The vector index is calculated by an offset of the
  ///          magnitude i * geometrical dimension
  /// \param[in] i index of vertex id
  /// \param[in] j index of coordinate id (0 for x, 1 for y and 2 for z)
  /// \return index for vector of coordinates coord_vtx_

  inline int ij2ind(int i, int j) const 
  { 
    return i * DIM + j; 
  }

  void print_vertices() const;
  void print_ref_vertices() const;
  
  CRefCellSPtr<DataType, DIM> ref_cell_;
  RefCellType fixed_ref_cell_type_;

  CellTrafoType trafo_type_;
  
  /// Vector, which holds the coordinates of every vertex of the physical cell
  std::vector<Coord> coord_vtx_;
  
  /// highest polynomial order 
  int order_;
  
  TrafoString name_;
  
  std::vector< mesh::MasterSlave > period_;  
  
  std::vector< int > phys2ref_facet_nr_;
  std::vector< int > ref2phys_facet_nr_;
};

template < class DataType, int DIM >
CellTransformation< DataType, DIM >::CellTransformation(CRefCellSPtr<DataType, DIM> ref_cell) 
: ref_cell_(ref_cell), 
order_(10),
trafo_type_(CELL_TRAFO_NONE) 
{
  period_.clear();
}

template < class DataType, int DIM >
CellTransformation< DataType, DIM >::CellTransformation(CRefCellSPtr<DataType, DIM> ref_cell,
                                                        const std::vector< mesh::MasterSlave >& period) 
: ref_cell_(ref_cell), 
order_(10),
trafo_type_(CELL_TRAFO_NONE),
period_(period)
{
}

/// \details Given vector of coordinates on physical cell, the are
///          stored the protected member variable coord_vtx_
template < class DataType, int DIM >
void CellTransformation< DataType, DIM >::reinit(const std::vector<Coord >&coord_vtx) 
{  
  assert (this->fixed_ref_cell_type_ == this->ref_cell_->type());
  assert (DIM == this->my_valid_dim_);
  this->coord_vtx_ = coord_vtx;
}

template < class DataType, int DIM >
void CellTransformation< DataType, DIM >::reinit(const std::vector<DataType> &coord_vtx) 
{  
  assert (DIM == this->my_valid_dim_);
  assert (this->fixed_ref_cell_type_ == this->ref_cell_->type());
  
  int num_points = coord_vtx.size() / DIM;
  coord_vtx_.clear();
  coord_vtx_.resize(num_points);
  for (int i=0; i<num_points; ++i) 
  {
    for (int d=0; d<DIM; ++d) 
    {
      coord_vtx_[i].set(d, coord_vtx[i*DIM + d]);
    } 
  }
}

template < class DataType, int DIM >
void CellTransformation< DataType, DIM >::reinit(const std::vector<DataType> &coord_vtx, const mesh::Entity& cell)
{
  assert (DIM == this->my_valid_dim_);
  this->reinit(coord_vtx);

  const mesh::TDim facet_tdim = cell.tdim() - 1;
  const mesh::IncidentEntityIterator facet_begin = cell.begin_incident(facet_tdim);
  const mesh::IncidentEntityIterator facet_end = cell.end_incident(facet_tdim);

  //int phys_facet_nr = 0;
  //int ref_facet_nr = 0;
  phys2ref_facet_nr_.clear();
  phys2ref_facet_nr_.reserve(6);
  ref2phys_facet_nr_.clear();
  std::vector< int > ref_facet_nr;
  int ctr = 0;
  
  for (auto facet_it = facet_begin; facet_it != facet_end; ++facet_it)
  {
    std::vector<DataType> facet_vertex_coords;
    facet_it->get_coordinates(facet_vertex_coords);

    std::vector< DataType > midpoint_coords;
    facet_it->get_midpoint(midpoint_coords);
    Coord facet_midpoint(midpoint_coords);

    Coord ref_facet_midpoint;
    
    // TODO: for nonlinear cell tranformations, facets can be non-planar
    // -> the computed facet_midpoint can be outside the cell
    // -> found1 = false 
    bool found1 = this->inverse(facet_midpoint, ref_facet_midpoint);
    bool found2 = false;
    
    if (found1)
    {
      found2 = this->ref_cell_->get_facet_nr (ref_facet_midpoint, ref_facet_nr);
    }
    
    if (found2)
    {
      phys2ref_facet_nr_.push_back(ref_facet_nr[0]);
    }
    else
    {
      phys2ref_facet_nr_.push_back(ctr);
    }
      
    ctr++;
    ref_facet_nr.clear();
    //phys_facet_nr++;
  }
  
  ref2phys_facet_nr_.resize(phys2ref_facet_nr_.size());
  for(int i = 0; i < phys2ref_facet_nr_.size(); ++i)
  {
    ref2phys_facet_nr_[phys2ref_facet_nr_[i]] = i;
  }
}


template < class DataType, int DIM >
void CellTransformation< DataType, DIM >::print_vertices() const 
{ 
  int num_points = coord_vtx_.size();
  for (int i=0; i<num_points; ++i) 
  {
    LOG_INFO("Vertex " << i << ": ", coord_vtx_[i]);
  }
}

template < class DataType, int DIM >
void CellTransformation< DataType, DIM >::print_ref_vertices() const 
{ 
  std::vector< Coord > ref_coords = this->ref_cell_->get_coords();
  int num_points = ref_coords.size();
  for (int i=0; i<num_points; ++i) 
  {
    LOG_INFO("Ref Vertex " << i << ": ", ref_coords[i]);
  }
}

template < class DataType, int DIM >
DataType CellTransformation< DataType, DIM >::cell_diameter() const 
{  
  DataType h = 0.;
  const int num_points = coord_vtx_.size();

  for (int i = 1; i != num_points; ++i)
  {
    for (int j = 0; j != i; ++j)
    {
      const DataType dist_ij = distance(coord_vtx_[i], coord_vtx_[j]);
      h = std::max(h, dist_ij);
    }
  }
  return h;
}

template < class DataType, int DIM >
DataType CellTransformation< DataType, DIM >::residual_inverse (const Coord &co_phy, const Coord &co_ref ) const
{
  Coord co_test = this->transform(co_ref);
  return norm(co_test - co_phy);
}

template < class DataType, int DIM >
bool CellTransformation< DataType, DIM >::residual_inverse_check (const Coord &co_phy, const Coord &co_ref ) const
{ 
  DataType residual = this->residual_inverse(co_phy, co_ref);
  LOG_DEBUG(2, "residual of inverse compuation " << residual << " vs tolerance " << INVERSE_RESIDUAL_TOL);

  if (residual < INVERSE_RESIDUAL_TOL)
  {
    return true;
  }
  return false;
}

template < class DataType, int DIM >
bool CellTransformation< DataType, DIM >::inverse(const Coord &co_phy, Coord &co_ref ) const
{  
  // compute inverse cell trafo
  // true if inverse compuation was successfull. 
  // However, this does not mean that the co_phy is really conatined in the physical cell

  for (int d=0; d!= DIM; ++d)
    co_ref.set(d, -1.);
    
  //LOG_INFO("coref in ", co_ref);
  bool found_ref_pt = this->inverse_impl(co_phy, co_ref);

  //LOG_INFO("coref inter ", co_ref);
  // avoid rounding errors on ref cell bdy 
  const DataType eps = this->ref_cell_->eps();
  for (int i = 0; i != DIM; ++i)
  {
    if (std::abs(co_ref[i]) < eps) 
    {
      co_ref.set(i, 0.);
    } 
    else if (std::abs(co_ref[i] - 1.) < eps) 
    {
      co_ref.set(i, 1.);
    }
  }
  
  //LOG_INFO("coref out ", co_ref);

#ifndef NDEBUG
  bool res_check = this->residual_inverse_check(co_phy, co_ref);
  if ((!res_check) && found_ref_pt)
  {
    LOG_INFO("trafo type ",  this->name_ << ", inverse residual " << this->residual_inverse(co_phy, co_ref));
    this->print_vertices();
    LOG_INFO("co phy ",  co_phy);
    LOG_INFO("co ref ",  co_ref);
  }
  if (found_ref_pt)
  {
    assert (res_check);
  }

#ifdef NEW_INVERSE
  if ((!found_ref_pt) && (this->period_.size() == 0))
  {
    if (this->contains_reference_point(co_ref))
    {
      LOG_INFO(" found ref point?", found_ref_pt);
      LOG_INFO(" co phys ", co_phy);
      LOG_INFO(" co ref ", co_ref);
      this->print_vertices();
      this->print_ref_vertices();
    }
    assert (!this->contains_reference_point(co_ref));
  }
#endif
#endif

  // check if ref point is in ref cell
  bool found = found_ref_pt && this->contains_reference_point(co_ref);

  if (found)
  {
    return true;
  }

  int num_periods = this->period_.size();
  if (num_periods == 0)
  {
    return found;
  }
  
  // check if physical point is in cell, if periodic boundary is taken into account

  std::vector< mesh::PeriodicBoundaryType > per_type;
  mesh::get_periodicity_type<DataType, DIM>(co_phy, this->period_, per_type);
  Coord pt_mapped;
                            
  for (int k = 0; k != num_periods; ++k) 
  {
    //std::cout << "per type = " << per_type[k] << std::endl;
    if (per_type[k] == mesh::NO_PERIODIC_BDY) 
    {
      // point does not lie on periodic boundary
      continue;
    }
    if (per_type[k] == mesh::MASTER_PERIODIC_BDY) 
    {
      // dof lies on master periodic boundary -> compute corresponding
      // coordinates on slave boundary
      mesh::periodify_master_to_slave<DataType, DIM>(co_phy, this->period_, k, pt_mapped);
    }
    if (per_type[k] == mesh::SLAVE_PERIODIC_BDY) 
    {
      // dof lies on slave periodic boundary -> compute corresponding
      // coordinates on master boundary
      mesh::periodify_slave_to_master<DataType, DIM>(co_phy, this->period_, k, pt_mapped);
    }
    //std::cout << "periodic mapped point " << pt_mapped << std::endl;
    found = this->inverse_impl(pt_mapped, co_ref);
    //std::cout << "found ? " << found << std::endl;
    if (found)
    {
      return true;
    }
  }
  return false;
}

template < class DataType, int DIM >
bool CellTransformation< DataType, DIM >::contains_physical_point(
    const Coord &coord_phys, Coord& cr) const {

  bool found_ref_point = false;
  // Compute reference coordinates
  if constexpr(DIM == 1) {
    found_ref_point = this->inverse(coord_phys, cr);
    LOG_DEBUG(2, "Physical Point " << coord_phys[0] << " has ref coords "
                                   << cr[0] << "? -> " << found_ref_point);
  }
  else if constexpr(DIM == 2) {
    found_ref_point = this->inverse(coord_phys, cr);
    LOG_DEBUG(2, "Physical Point (" << coord_phys[0] << ", " << coord_phys[1]
                                    << ") has ref coords " << cr[0] << ", "
                                    << cr[1] << ") ? -> " << found_ref_point);
  }
  else if constexpr(DIM == 3) {
    found_ref_point = this->inverse(coord_phys, cr);
    LOG_DEBUG(2, "Physical Point ("
                     << coord_phys[0] << ", " << coord_phys[1] << ", "
                     << coord_phys[2] << ") has ref coords " << cr[0] << ", "
                     << cr[1] << ", " << cr[2] << ") ? -> " << found_ref_point);
  }
  else 
  {
    throw "Invalid dimension!";
  }

  if (!found_ref_point) {
    return false;
  }

  const bool contains_pt = this->contains_reference_point(cr);

  return contains_pt;
}

template<class DataType, int DIM>
void CellTransformation<DataType, DIM>::transform ( const Coord& coord_ref, Coord& coord_mapped ) const 
{
  if constexpr(DIM == 1)
  {
    coord_mapped.set(0, this->x ( coord_ref ));
  }
  else if constexpr(DIM == 2)
  {
    coord_mapped.set(0, this->x ( coord_ref ));
    coord_mapped.set(1, this->y ( coord_ref ));
  }
  else if constexpr(DIM == 3)
  {
    coord_mapped.set(0, this->x ( coord_ref ));
    coord_mapped.set(1, this->y ( coord_ref ));
    coord_mapped.set(2, this->z ( coord_ref ));
  }
  else 
  {
    assert(0);
  }
}

template<class DataType, int DIM>
void CellTransformation<DataType,DIM>::J ( const Coord& coord, mat& mat) const 
{
  if constexpr(DIM == 1)
  {
    mat.set(0, 0, this->x_x(coord));
  }
  else if constexpr(DIM == 2)
  {
    mat.set(0, 0, this->x_x(coord));
    mat.set(0, 1, this->x_y(coord));
    mat.set(1, 0, this->y_x(coord));
    mat.set(1, 1, this->y_y(coord));
  }
  else if constexpr(DIM == 3)
  {
      mat.set(0, 0, this->x_x(coord));
      mat.set(0, 1, this->x_y(coord));
      mat.set(0, 2, this->x_z(coord));
      
      mat.set(1, 0, this->y_x(coord));
      mat.set(1, 1, this->y_y(coord));
      mat.set(1, 2, this->y_z(coord));

      mat.set(2, 0, this->z_x(coord));
      mat.set(2, 1, this->z_y(coord));
      mat.set(2, 2, this->z_z(coord));
  }
  else 
  {
    assert(0);
  }
}

template<class DataType, int DIM>
void CellTransformation<DataType,DIM>::J_and_detJ ( const Coord& coord, mat& mat, DataType& detJ) const 
{
  if constexpr(DIM == 1)
  {
      mat.set(0, 0, this->x_x(coord));
      detJ = mat(0,0);
  }
  else if constexpr(DIM == 2)
  {
      mat.set(0, 0, this->x_x(coord));
      mat.set(0, 1, this->x_y(coord));
      mat.set(1, 0, this->y_x(coord));
      mat.set(1, 1, this->y_y(coord));
      detJ = mat(0,0) * mat(1,1) - mat(0,1) * mat(1,0);
  }
  else if constexpr(DIM == 3)
  {
      mat.set(0, 0, this->x_x(coord));
      mat.set(0, 1, this->x_y(coord));
      mat.set(0, 2, this->x_z(coord));
      
      mat.set(1, 0, this->y_x(coord));
      mat.set(1, 1, this->y_y(coord));
      mat.set(1, 2, this->y_z(coord));

      mat.set(2, 0, this->z_x(coord));
      mat.set(2, 1, this->z_y(coord));
      mat.set(2, 2, this->z_z(coord));
      
      // TODO_VECTORIZE
      detJ = mat(0,0) * (mat(1,1) * mat(2,2) - mat(1,2) *  mat(2,1))
           - mat(0,1) * (mat(1,0) * mat(2,2) - mat(1,2) *  mat(2,0))
           + mat(0,2) * (mat(1,0) * mat(2,1) - mat(1,1) *  mat(2,0));
  }
  else 
  {
    assert(0);
  }
}

template<class DataType, int DIM>
void CellTransformation<DataType, DIM>::H ( const Coord& coord, size_t d, mat& mat) const 
{
  if constexpr(DIM == 1)
  {
      switch (d)
      {
        case 0:
          mat.set(0, 0, this->x_xx(coord));
          break;
        default:
          assert(0);
          break;
      }
  }
  else if constexpr(DIM == 2)
  {
      switch (d)
      {
        case 0:
          mat.set(0, 0, this->x_xx(coord));
          mat.set(0, 1, this->x_xy(coord));
          mat.set(1, 0, mat(0,1));
          mat.set(1, 1, this->x_yy(coord));
          break;
        case 1:
          mat.set(0, 0, this->y_xx(coord));
          mat.set(0, 1, this->y_xy(coord));
          mat.set(1, 0, mat(0,1));
          mat.set(1, 1, this->y_yy(coord));
          break;
        default:
          assert(0);
          break;
      }
  }
  else if constexpr(DIM == 3)
  {
      switch (d)
      {
        case 0:
          mat.set(0, 0, this->x_xx(coord));
          mat.set(0, 1, this->x_xy(coord));
          mat.set(0, 2, this->x_xz(coord));
          mat.set(1, 0, mat(0,1));
          mat.set(1, 1, this->x_yy(coord));
          mat.set(1, 2, this->x_yz(coord));
          mat.set(2, 0, mat(0,2));
          mat.set(2, 1, mat(1,2));
          mat.set(2, 2, this->x_zz(coord));
          break;
        case 1:
          mat.set(0, 0, this->y_xx(coord));
          mat.set(0, 1, this->y_xy(coord));
          mat.set(0, 2, this->y_xz(coord));
          mat.set(1, 0, mat(0,1));
          mat.set(1, 1, this->y_yy(coord));
          mat.set(1, 2, this->y_yz(coord));
          mat.set(2, 0, mat(0,2));
          mat.set(2, 1, mat(1,2));
          mat.set(2, 2, this->y_zz(coord));
          break;
        case 2:
          mat.set(0, 0, this->z_xx(coord));
          mat.set(0, 1, this->z_xy(coord));
          mat.set(0, 2, this->z_xz(coord));
          mat.set(1, 0, mat(0,1));
          mat.set(1, 1, this->z_yy(coord));
          mat.set(1, 2, this->z_yz(coord));
          mat.set(2, 0, mat(0,2));
          mat.set(2, 1, mat(1,2));
          mat.set(2, 2, this->z_zz(coord));
          break;
        default:
          assert(0);
          break;
      }
  }
  else 
  {
    assert(0);
  }
}

template<class DataType, int DIM>
DataType CellTransformation<DataType, DIM>::detJ ( const Coord& pt) const 
{
  if constexpr(DIM == 1)
  {
    return this->x_x(pt);
  }
  else if constexpr(DIM == 2)
  {
    return this->x_x(pt) * this->y_y(pt) - this->x_y(pt) * this->y_x(pt);
  }
  else if constexpr(DIM == 3)
  {
    return this->x_x(pt) * (this->y_y(pt) * this->z_z(pt) - this->y_z(pt) * this->z_y(pt)) 
           - this->x_y(pt) * (this->y_x(pt) * this->z_z(pt) - this->y_z(pt) * this->z_x(pt))
           + this->x_z(pt) * (this->y_x(pt) * this->z_y(pt) - this->y_y(pt) * this->z_x(pt));
  }
  else 
  {
    assert(0);
  }
}

template<class DataType, int DIM>
void CellTransformation<DataType,DIM>::grad_detJ ( const Coord& coord, Coord& grad) const 
{
  if constexpr(DIM == 1)
  {
    grad.set(0, this->detJ_x(coord));
  }
  else if constexpr(DIM == 2)
  {
    grad.set(0, this->detJ_x(coord));
    grad.set(1, this->detJ_y(coord));
  }
  else if constexpr(DIM == 3)
  {
    grad.set(0, this->detJ_x(coord));
    grad.set(1, this->detJ_y(coord));
    grad.set(2, this->detJ_z(coord));
  }
  else 
  {
    assert(0);
  }
}

template<class DataType, int DIM>
void CellTransformation<DataType,DIM>::grad_inv_detJ ( const Coord& coord, Coord& grad) const 
{
  const DataType detJ = this->detJ(coord);
  const DataType inv_detJ_2 = -1. / (detJ * detJ);

  if constexpr(DIM == 1)
  {
    grad.set(0, inv_detJ_2 * this->detJ_x(coord));
  }
  else if constexpr(DIM == 2)
  {
    grad.set(0, inv_detJ_2 * this->detJ_x(coord));
    grad.set(1, inv_detJ_2 * this->detJ_y(coord));
  }
  else if constexpr(DIM == 3)
  {
    grad.set(0, inv_detJ_2 * this->detJ_x(coord));
    grad.set(1, inv_detJ_2 * this->detJ_y(coord));
    grad.set(2, inv_detJ_2 * this->detJ_z(coord));
  }
  else 
  {
    assert(0);
  }
}

template<class DataType, int DIM>
void CellTransformation<DataType,DIM>::hessian_detJ ( const Coord& coord, mat& mat) const 
{
  switch (DIM)
  {
    case 1:
      mat.set(0, 0, this->detJ_xx(coord));
      break;
    case 2:
      mat.set(0, 0, this->detJ_xx(coord));
      mat.set(0, 1, this->detJ_xy(coord));
      mat.set(1, 0, mat(0,1));
      mat.set(1, 1, this->detJ_yy(coord));
      break;
    case 3:
      mat.set(0, 0, this->detJ_xx(coord));
      mat.set(0, 1, this->detJ_xy(coord));
      mat.set(0, 2, this->detJ_xz(coord));
      
      mat.set(1, 0, mat(0,1));
      mat.set(1, 1, this->detJ_yy(coord));
      mat.set(1, 2, this->detJ_yz(coord));

      mat.set(2, 0, mat(0,2));
      mat.set(2, 1, mat(1,2));
      mat.set(2, 2, this->detJ_zz(coord));
      break;
    default:
      assert (0);
      break;
  }
}

template<class DataType, int DIM>
DataType CellTransformation<DataType, DIM>::detJ_x ( const Coord& pt) const 
{
  if (this->order_ <= 1)
  {
    return 0.;
  }
  switch (DIM)
  {
    case 1:
      return this->x_xx(pt);
    case 2:
      return this->x_xx(pt) * this->y_y(pt) + this->x_x(pt) * this->y_xy(pt)  
           - this->x_xy(pt) * this->y_x(pt) - this->x_y(pt) * this->y_xx(pt); 
    case 3:
      return  this->x_xx(pt) * (this->y_y(pt)  * this->z_z(pt) - this->y_z(pt) * this->z_y(pt))
            + this->x_x(pt)  * (this->y_xy(pt) * this->z_z(pt) + this->y_y(pt) * this->z_xz(pt) 
                              - this->y_xz(pt) * this->z_y(pt) - this->y_z(pt) * this->z_xy(pt))
            - this->x_xy(pt) * (this->y_x(pt)  * this->z_z(pt) - this->y_z(pt) * this->z_x(pt))
            - this->x_y(pt)  * (this->y_xx(pt) * this->z_z(pt) + this->y_x(pt) * this->z_xz(pt) 
                              - this->y_xz(pt) * this->z_x(pt) - this->y_z(pt) * this->z_xx(pt))
            + this->x_xz(pt) * (this->y_x(pt)  * this->z_y(pt) - this->y_y(pt) * this->z_x(pt))
            + this->x_z(pt)  * (this->y_xx(pt) * this->z_y(pt) + this->y_x(pt) * this->z_xy(pt)
                              - this->y_xy(pt) * this->z_x(pt) - this->y_y(pt) * this->z_xx(pt));
    default:
      assert(0);
      break;
  }
}

template<class DataType, int DIM>
DataType CellTransformation<DataType, DIM>::detJ_y ( const Coord& pt) const 
{
  if (this->order_ <= 1)
  {
    return 0.;
  }
  switch (DIM)
  {
    case 1:
      return 0.;
    case 2:
      return this->x_xy(pt) * this->y_y(pt) + this->x_x(pt) * this->y_yy(pt)  
           - this->x_yy(pt) * this->y_x(pt) - this->x_y(pt) * this->y_xy(pt); 
    case 3:
      return  this->x_xy(pt) * (this->y_y(pt)  * this->z_z(pt) - this->y_z(pt) * this->z_y(pt))
            + this->x_x(pt)  * (this->y_yy(pt) * this->z_z(pt) + this->y_y(pt) * this->z_yz(pt) 
                              - this->y_yz(pt) * this->z_y(pt) - this->y_z(pt) * this->z_yy(pt))
            - this->x_yy(pt) * (this->y_x(pt)  * this->z_z(pt) - this->y_z(pt) * this->z_x(pt))
            - this->x_y(pt)  * (this->y_xy(pt) * this->z_z(pt) + this->y_x(pt) * this->z_yz(pt) 
                              - this->y_yz(pt) * this->z_x(pt) - this->y_z(pt) * this->z_xy(pt))
            + this->x_yz(pt) * (this->y_x(pt)  * this->z_y(pt) - this->y_y(pt) * this->z_x(pt))
            + this->x_z(pt)  * (this->y_xy(pt) * this->z_y(pt) + this->y_x(pt) * this->z_yy(pt)
                              - this->y_yy(pt) * this->z_x(pt) - this->y_y(pt) * this->z_xy(pt));  
    default:
      assert(0);
      break;
  }
}

template<class DataType, int DIM>
DataType CellTransformation<DataType, DIM>::detJ_z ( const Coord& pt) const 
{
  if (this->order_ <= 1)
  {
    return 0.;
  }
  switch (DIM)
  {
    case 1:
      return 0.;
    case 2:
      return 0.;
    case 3:
      return  this->x_xz(pt) * (this->y_y(pt)  * this->z_z(pt) - this->y_z(pt) * this->z_y(pt))
            + this->x_x(pt)  * (this->y_yz(pt) * this->z_z(pt) + this->y_y(pt) * this->z_zz(pt) 
                              - this->y_zz(pt) * this->z_y(pt) - this->y_z(pt) * this->z_yz(pt))
            - this->x_yz(pt) * (this->y_x(pt)  * this->z_z(pt) - this->y_z(pt) * this->z_x(pt))
            - this->x_y(pt)  * (this->y_xz(pt) * this->z_z(pt) + this->y_x(pt) * this->z_zz(pt) 
                              - this->y_zz(pt) * this->z_x(pt) - this->y_z(pt) * this->z_xz(pt))
            + this->x_zz(pt) * (this->y_x(pt)  * this->z_y(pt) - this->y_y(pt) * this->z_x(pt))
            + this->x_z(pt)  * (this->y_xz(pt) * this->z_y(pt) + this->y_x(pt) * this->z_yz(pt)
                              - this->y_yz(pt) * this->z_x(pt) - this->y_y(pt) * this->z_xz(pt));                                            
    default:
      assert(0);
      break;
  }
}

template<class DataType, int DIM>
DataType CellTransformation<DataType, DIM>::detJ_xx ( const Coord& pt) const 
{
  if (this->order_ <= 2)
  {
    return 0.;
  }
  switch (DIM)
  {
    case 1:
      return  this->x_xxx(pt);
    case 2:
      return  this->x_xxx(pt) * this->y_y(pt)  + this->x_xx(pt) * this->y_xy(pt) 
            + this->x_xx(pt)  * this->y_xy(pt) + this->x_x(pt)  * this->y_xxy(pt)  
            - this->x_xxy(pt) * this->y_x(pt)  - this->x_xy(pt) * this->y_xx(pt) 
            - this->x_xy(pt)  * this->y_xx(pt) - this->x_y(pt)  * this->y_xxx(pt) ;  
    case 3:
      return       this->x_xxx(pt) * (this->y_y(pt)   * this->z_z(pt)  - this->y_z(pt)  * this->z_y(pt))
            + 2. * this->x_xx(pt)  * (this->y_xy(pt)  * this->z_z(pt)  + this->y_y(pt)  * this->z_xz(pt) 
                                    - this->y_xz(pt)  * this->z_y(pt)  - this->y_z(pt)  * this->z_xy(pt))
            +      this->x_x(pt)   * (this->y_xxy(pt) * this->z_z(pt)  + this->y_xy(pt) * this->z_xz(pt) 
                                    + this->y_xy(pt)  * this->z_xz(pt) + this->y_y(pt)  * this->z_xxz(pt)
                                    - this->y_xxz(pt) * this->z_y(pt)  - this->y_xz(pt) * this->z_xy(pt) 
                                    - this->y_xz(pt)  * this->z_xy(pt) - this->y_z(pt)  * this->z_xxy(pt))
            -      this->x_xxy(pt) * (this->y_x(pt)   * this->z_z(pt)  - this->y_z(pt)  * this->z_x(pt))
            - 2. * this->x_xy(pt)  * (this->y_xx(pt)  * this->z_z(pt)  + this->y_x(pt)  * this->z_xz(pt) 
                                    - this->y_xz(pt)  * this->z_x(pt)  - this->y_z(pt)  * this->z_xx(pt))
            -      this->x_y(pt)   * (this->y_xxx(pt) * this->z_z(pt)  + this->y_xx(pt) * this->z_xz(pt) 
                                    + this->y_xx(pt)  * this->z_xz(pt) + this->y_x(pt)  * this->z_xxz(pt)  
                                    - this->y_xxz(pt) * this->z_x(pt)  - this->y_xz(pt) * this->z_xx(pt) 
                                    - this->y_xz(pt)  * this->z_xx(pt) - this->y_z(pt)  * this->z_xxx(pt))
            +      this->x_xxz(pt) * (this->y_x(pt)   * this->z_y(pt)  - this->y_y(pt)  * this->z_x(pt))
            + 2. * this->x_xz(pt)  * (this->y_xx(pt)  * this->z_y(pt)  + this->y_x(pt)  * this->z_xy(pt)
                                    - this->y_xy(pt)  * this->z_x(pt)  - this->y_y(pt)  * this->z_xx(pt))
            +      this->x_z(pt)   * (this->y_xxx(pt) * this->z_y(pt)  + this->y_xx(pt) * this->z_xy(pt)
                                    + this->y_xx(pt)  * this->z_xy(pt) + this->y_x(pt)  * this->z_xxy(pt)
                                    - this->y_xxy(pt) * this->z_x(pt)  - this->y_xy(pt) * this->z_xx(pt)
                                    - this->y_xy(pt)  * this->z_xx(pt) - this->y_y(pt)  * this->z_xxx(pt));
    default:
      assert(0);
      break;
  }
}

template<class DataType, int DIM>
DataType CellTransformation<DataType, DIM>::detJ_xy ( const Coord& pt) const 
{
  if (this->order_ <= 2)
  {
    return 0.;
  }
  switch (DIM)
  {
    case 1:
      return 0.;
    case 2:
      return  this->x_xxy(pt) * this->y_y(pt)  + this->x_xx(pt) * this->y_yy(pt) 
          + this->x_xy(pt)  * this->y_xy(pt) + this->x_x(pt)  * this->y_xyy(pt)  
          - this->x_xyy(pt) * this->y_x(pt)  - this->x_xy(pt) * this->y_xy(pt) 
          - this->x_yy(pt)  * this->y_xx(pt) - this->x_y(pt)  * this->y_xxy(pt);   
    case 3:
      return  this->x_xxy(pt) * (this->y_y(pt)   * this->z_z(pt)  - this->y_z(pt)  * this->z_y(pt))
            + this->x_xy(pt)  * (this->y_yy(pt)  * this->z_z(pt)  + this->y_y(pt)  * this->z_yz(pt)
                               - this->y_yz(pt)  * this->z_y(pt)  - this->y_z(pt)  * this->z_yy(pt))
            + this->x_xy(pt)  * (this->y_xy(pt)  * this->z_z(pt)  + this->y_y(pt)  * this->z_xz(pt) 
                               - this->y_xz(pt)  * this->z_y(pt)  - this->y_z(pt)  * this->z_xy(pt))
            + this->x_x(pt)   * (this->y_xyy(pt) * this->z_z(pt)  + this->y_xy(pt) * this->z_yz(pt) 
                               + this->y_yy(pt)  * this->z_xz(pt) + this->y_y(pt)  * this->z_xyz(pt) 
                               - this->y_xyz(pt) * this->z_y(pt)  - this->y_xz(pt) * this->z_yy(pt) 
                               - this->y_yz(pt)  * this->z_xy(pt) - this->y_z(pt)  * this->z_xyy(pt))
            - this->x_xyy(pt) * (this->y_x(pt)   * this->z_z(pt)  - this->y_z(pt)  * this->z_x(pt))
            - this->x_xy(pt)  * (this->y_xy(pt)  * this->z_z(pt)  + this->y_x(pt)  * this->z_yz(pt) 
                               - this->y_yz(pt)  * this->z_x(pt)  - this->y_z(pt)  * this->z_xy(pt))
            - this->x_yy(pt)  * (this->y_xx(pt)  * this->z_z(pt)  + this->y_x(pt)  * this->z_xz(pt) 
                               - this->y_xz(pt)  * this->z_x(pt)  - this->y_z(pt)  * this->z_xx(pt))        
            - this->x_y(pt)   * (this->y_xxy(pt) * this->z_z(pt)  + this->y_xx(pt) * this->z_yz(pt) 
                               + this->y_xy(pt)  * this->z_xz(pt) + this->y_x(pt)  * this->z_xyz(pt) 
                               - this->y_xyz(pt) * this->z_x(pt)  - this->y_xz(pt) * this->z_xy(pt) 
                               - this->y_yz(pt)  * this->z_xx(pt) - this->y_z(pt)  * this->z_xxy(pt))
            + this->x_xyz(pt) * (this->y_x(pt)   * this->z_y(pt)  - this->y_y(pt)  * this->z_x(pt))
            + this->x_xz(pt)  * (this->y_xy(pt)  * this->z_y(pt)  + this->y_x(pt)  * this->z_yy(pt) 
                               - this->y_yy(pt)  * this->z_x(pt)  - this->y_y(pt)  * this->z_xy(pt))
            + this->x_yz(pt)  * (this->y_xx(pt)  * this->z_y(pt)  + this->y_x(pt)  * this->z_xy(pt)
                               - this->y_xy(pt)  * this->z_x(pt)  - this->y_y(pt)  * this->z_xx(pt))
            + this->x_z(pt)   * (this->y_xxy(pt) * this->z_y(pt)  + this->y_xx(pt) * this->z_yy(pt) 
                               + this->y_xy(pt)  * this->z_xy(pt) + this->y_x(pt)  * this->z_xyy(pt)
                               - this->y_xyy(pt) * this->z_x(pt)  - this->y_xy(pt) * this->z_xy(pt) 
                               - this->y_yy(pt)  * this->z_xx(pt) - this->y_y(pt)  * this->z_xxy(pt)); 
    default:
      assert(0);
      break;
  }
}

template<class DataType, int DIM>
DataType CellTransformation<DataType, DIM>::detJ_xz ( const Coord& pt) const 
{
  if (this->order_ <= 2)
  {
    return 0.;
  }
  switch (DIM)
  {
    case 1:
      return 0.;
    case 2:
      return 0.;
    case 3:
      return  this->x_xxz(pt) * (this->y_y(pt)   * this->z_z(pt)  - this->y_z(pt)  * this->z_y(pt))
            + this->x_xx(pt)  * (this->y_yz(pt)  * this->z_z(pt)  + this->y_y(pt)  * this->z_zz(pt) 
                               - this->y_zz(pt)  * this->z_y(pt)  - this->y_z(pt)  * this->z_yz(pt))
            + this->x_xz(pt)  * (this->y_xy(pt)  * this->z_z(pt)  + this->y_y(pt)  * this->z_xz(pt) 
                               - this->y_xz(pt)  * this->z_y(pt)  - this->y_z(pt)  * this->z_xy(pt))
            + this->x_x(pt)   * (this->y_xyz(pt) * this->z_z(pt)  + this->y_xy(pt) * this->z_zz(pt) 
                               + this->y_yz(pt)  * this->z_xz(pt) + this->y_y(pt)  * this->z_xzz(pt) 
                               - this->y_xzz(pt) * this->z_y(pt)  - this->y_xz(pt) * this->z_yz(pt)
                               - this->y_zz(pt)  * this->z_xy(pt) - this->y_z(pt)  * this->z_xyz(pt))
            - this->x_xyz(pt) * (this->y_x(pt)   * this->z_z(pt)  - this->y_z(pt)  * this->z_x(pt))
            - this->x_xy(pt)  * (this->y_xz(pt)  * this->z_z(pt)  + this->y_x(pt)  * this->z_zz(pt) 
                               - this->y_zz(pt)  * this->z_x(pt)  - this->y_z(pt)  * this->z_xz(pt))
            - this->x_yz(pt)  * (this->y_xx(pt)  * this->z_z(pt)  + this->y_x(pt)  * this->z_xz(pt) 
                               - this->y_xz(pt)  * this->z_x(pt)  - this->y_z(pt)  * this->z_xx(pt))
            - this->x_y(pt)   * (this->y_xxz(pt) * this->z_z(pt)  + this->y_xx(pt) * this->z_zz(pt) 
                               + this->y_xz(pt)  * this->z_xz(pt) + this->y_x(pt)  * this->z_xzz(pt) 
                               - this->y_xzz(pt) * this->z_x(pt)  - this->y_xz(pt) * this->z_xz(pt) 
                               - this->y_zz(pt)  * this->z_xx(pt) - this->y_z(pt)  * this->z_xxz(pt))
            + this->x_xzz(pt) * (this->y_x(pt)   * this->z_y(pt)  - this->y_y(pt)  * this->z_x(pt))
            + this->x_xz(pt)  * (this->y_xz(pt)  * this->z_y(pt)  + this->y_x(pt)  * this->z_yz(pt) 
                               - this->y_yz(pt)  * this->z_x(pt)  - this->y_y(pt)  * this->z_xz(pt))
            + this->x_zz(pt)  * (this->y_xx(pt)  * this->z_y(pt)  + this->y_x(pt)  * this->z_xy(pt)
                               - this->y_xy(pt)  * this->z_x(pt)  - this->y_y(pt)  * this->z_xx(pt))                                            
            + this->x_z(pt)   * (this->y_xxz(pt) * this->z_y(pt)  + this->y_xx(pt) * this->z_yz(pt) 
                               + this->y_xz(pt)  * this->z_xy(pt) + this->y_x(pt)  * this->z_xyz(pt)
                               - this->y_xyz(pt) * this->z_x(pt)  - this->y_xy(pt) * this->z_xz(pt) 
                               - this->y_yz(pt)  * this->z_xx(pt) - this->y_y(pt)  * this->z_xxz(pt));  
    default:
      assert(0);
      break;
  }
}

template<class DataType, int DIM>
DataType CellTransformation<DataType, DIM>::detJ_yy ( const Coord& pt) const 
{
  if (this->order_ <= 2)
  {
    return 0.;
  }
  switch (DIM)
  {
    case 1:
      return 0.;
    case 2:
      return this->x_xyy(pt) * this->y_y(pt)  + this->x_xy(pt) * this->y_yy(pt) 
           + this->x_xy(pt)  * this->y_yy(pt) + this->x_x(pt)  * this->y_yyy(pt)  
           - this->x_yyy(pt) * this->y_x(pt)  - this->x_yy(pt) * this->y_xy(pt) 
           - this->x_yy(pt)  * this->y_xy(pt) - this->x_y(pt)  * this->y_xyy(pt); 
    case 3:
      return  this->x_xyy(pt) * (this->y_y(pt)   * this->z_z(pt)  - this->y_z(pt)  * this->z_y(pt))
            + this->x_xy(pt)  * (this->y_yy(pt)  * this->z_z(pt)  + this->y_y(pt)  * this->z_yz(pt) 
                               - this->y_yz(pt)  * this->z_y(pt)  - this->y_z(pt)  * this->z_yy(pt))
            + this->x_xy(pt)  * (this->y_yy(pt)  * this->z_z(pt)  + this->y_y(pt)  * this->z_yz(pt) 
                               - this->y_yz(pt)  * this->z_y(pt)  - this->y_z(pt)  * this->z_yy(pt))
            + this->x_x(pt)   * (this->y_yyy(pt) * this->z_z(pt)  + this->y_yy(pt) * this->z_yz(pt) 
                               + this->y_yy(pt)  * this->z_yz(pt) + this->y_y(pt)  * this->z_yyz(pt) 
                               - this->y_yyz(pt) * this->z_y(pt)  - this->y_yz(pt) * this->z_yy(pt) 
                               - this->y_yz(pt)  * this->z_yy(pt) - this->y_z(pt)  * this->z_yyy(pt))
            - this->x_yyy(pt) * (this->y_x(pt)   * this->z_z(pt)  - this->y_z(pt)  * this->z_x(pt))
            - this->x_yy(pt)  * (this->y_xy(pt)  * this->z_z(pt)  + this->y_x(pt)  * this->z_yz(pt) 
                               - this->y_yz(pt)  * this->z_x(pt)  - this->y_z(pt)  * this->z_xy(pt))
            - this->x_yy(pt)  * (this->y_xy(pt)  * this->z_z(pt)  + this->y_x(pt)  * this->z_yz(pt) 
                               - this->y_yz(pt)  * this->z_x(pt)  - this->y_z(pt)  * this->z_xy(pt))
            - this->x_y(pt)   * (this->y_xyy(pt) * this->z_z(pt)  + this->y_xy(pt) * this->z_yz(pt) 
                               + this->y_xy(pt)  * this->z_yz(pt) + this->y_x(pt)  * this->z_yyz(pt) 
                               - this->y_yyz(pt) * this->z_x(pt)  - this->y_yz(pt) * this->z_xy(pt) 
                               - this->y_yz(pt)  * this->z_xy(pt) - this->y_z(pt)  * this->z_xyy(pt))
            + this->x_yyz(pt) * (this->y_x(pt)   * this->z_y(pt)  - this->y_y(pt)  * this->z_x(pt))
            + this->x_yz(pt)  * (this->y_xy(pt)  * this->z_y(pt)  + this->y_x(pt)  * this->z_yy(pt) 
                               - this->y_yy(pt)  * this->z_x(pt)  - this->y_y(pt)  * this->z_xy(pt))
            + this->x_yz(pt)  * (this->y_xy(pt)  * this->z_y(pt)  + this->y_x(pt)  * this->z_yy(pt)
                               - this->y_yy(pt)  * this->z_x(pt)  - this->y_y(pt)  * this->z_xy(pt)) 
            + this->x_z(pt)   * (this->y_xyy(pt) * this->z_y(pt)  + this->y_xy(pt) * this->z_yy(pt) 
                               + this->y_xy(pt)  * this->z_yy(pt) + this->y_x(pt)  * this->z_yyy(pt)
                               - this->y_yyy(pt) * this->z_x(pt)  - this->y_yy(pt) * this->z_xy(pt) 
                               - this->y_yy(pt)  * this->z_xy(pt) - this->y_y(pt)  * this->z_xyy(pt));
    default:
      assert(0);
      break;
  }
}

template<class DataType, int DIM>
DataType CellTransformation<DataType, DIM>::detJ_yz ( const Coord& pt) const 
{
  if (this->order_ <= 2)
  {
    return 0.;
  }
  switch (DIM)
  {
    case 1:
      return 0.;
    case 2:
      return 0.;
    case 3:
      return  this->x_xyz(pt) * (this->y_y(pt)   * this->z_z(pt)  - this->y_z(pt)  * this->z_y(pt))
            + this->x_xy(pt)  * (this->y_yz(pt)  * this->z_z(pt)  + this->y_y(pt)  * this->z_zz(pt) 
                               - this->y_zz(pt)  * this->z_y(pt)  - this->y_z(pt)  * this->z_yz(pt))
            + this->x_xz(pt)  * (this->y_yy(pt)  * this->z_z(pt)  + this->y_y(pt)  * this->z_yz(pt) 
                               - this->y_yz(pt)  * this->z_y(pt)  - this->y_z(pt)  * this->z_yy(pt))
            + this->x_x(pt)   * (this->y_yyz(pt) * this->z_z(pt)  + this->y_yy(pt) * this->z_zz(pt) 
                               + this->y_yz(pt)  * this->z_yz(pt) + this->y_y(pt)  * this->z_yzz(pt) 
                               - this->y_yzz(pt) * this->z_y(pt)  - this->y_yz(pt) * this->z_yz(pt) 
                               - this->y_zz(pt)  * this->z_yy(pt) - this->y_z(pt)  * this->z_yyz(pt))
            - this->x_yyz(pt) * (this->y_x(pt)   * this->z_z(pt)  - this->y_z(pt)  * this->z_x(pt))
            - this->x_yy(pt)  * (this->y_xz(pt)  * this->z_z(pt)  + this->y_x(pt)  * this->z_zz(pt) 
                               - this->y_zz(pt)  * this->z_x(pt)  - this->y_z(pt)  * this->z_xz(pt))
            - this->x_yz(pt)  * (this->y_xy(pt)  * this->z_z(pt)  + this->y_x(pt)  * this->z_yz(pt) 
                               - this->y_yz(pt)  * this->z_x(pt)  - this->y_z(pt)  * this->z_xy(pt))
            - this->x_y(pt)   * (this->y_xyz(pt) * this->z_z(pt)  + this->y_xy(pt) * this->z_zz(pt) 
                               + this->y_xz(pt)  * this->z_yz(pt) + this->y_x(pt)  * this->z_yzz(pt) 
                               - this->y_yzz(pt) * this->z_x(pt)  - this->y_yz(pt) * this->z_xz(pt) 
                               - this->y_zz(pt)  * this->z_xy(pt) - this->y_z(pt)  * this->z_xyz(pt))
            + this->x_yzz(pt) * (this->y_x(pt)   * this->z_y(pt)  - this->y_y(pt)  * this->z_x(pt))
            + this->x_yz(pt)  * (this->y_xz(pt)  * this->z_y(pt)  + this->y_x(pt)  * this->z_yz(pt) 
                               - this->y_yz(pt)  * this->z_x(pt)  - this->y_y(pt)  * this->z_xz(pt))
            + this->x_zz(pt)  * (this->y_xy(pt)  * this->z_y(pt)  + this->y_x(pt)  * this->z_yy(pt)
                               - this->y_yy(pt)  * this->z_x(pt)  - this->y_y(pt)  * this->z_xy(pt))                                            
            + this->x_z(pt)   * (this->y_xyz(pt) * this->z_y(pt)  + this->y_xy(pt) * this->z_yz(pt) 
                               + this->y_xz(pt)  * this->z_yy(pt) + this->y_x(pt)  * this->z_yyz(pt)
                               - this->y_yyz(pt) * this->z_x(pt)  - this->y_yy(pt) * this->z_xz(pt) 
                               - this->y_yz(pt)  * this->z_xy(pt) - this->y_y(pt)  * this->z_xyz(pt));
    default:
      assert(0);
      break;
  }
}

template<class DataType, int DIM>
DataType CellTransformation<DataType, DIM>::detJ_zz ( const Coord& pt) const 
{
  if (this->order_ <= 2)
  {
    return 0.;
  }
  switch (DIM)
  {
    case 1:
      return 0.;
    case 2:
      return 0.;
    case 3:
      return  this->x_xzz(pt) * (this->y_y(pt)   * this->z_z(pt)  - this->y_z(pt)  * this->z_y(pt))
            +  this->x_xz(pt) * (this->y_yz(pt)  * this->z_z(pt)  + this->y_y(pt)  * this->z_zz(pt) 
                               - this->y_zz(pt)  * this->z_y(pt)  - this->y_z(pt)  * this->z_yz(pt))
            + this->x_xz(pt)  * (this->y_yz(pt)  * this->z_z(pt)  + this->y_y(pt)  * this->z_zz(pt) 
                               - this->y_zz(pt)  * this->z_y(pt)  - this->y_z(pt)  * this->z_yz(pt))
            + this->x_x(pt)   * (this->y_yzz(pt) * this->z_z(pt)  + this->y_yz(pt) * this->z_zz(pt) 
                               + this->y_yz(pt)  * this->z_zz(pt) + this->y_y(pt)  * this->z_zzz(pt) 
                               - this->y_zzz(pt) * this->z_y(pt)  - this->y_zz(pt) * this->z_yz(pt) 
                               - this->y_zz(pt)  * this->z_yz(pt) - this->y_z(pt)  * this->z_yzz(pt))
            - this->x_yzz(pt) * (this->y_x(pt)   * this->z_z(pt)  - this->y_z(pt)  * this->z_x(pt))
            - this->x_yz(pt)  * (this->y_xz(pt)  * this->z_z(pt)  + this->y_x(pt)  * this->z_zz(pt) 
                               - this->y_zz(pt)  * this->z_x(pt)  - this->y_z(pt)  * this->z_xz(pt))
            - this->x_yz(pt)  * (this->y_xz(pt)  * this->z_z(pt)  + this->y_x(pt)  * this->z_zz(pt) 
                               - this->y_zz(pt)  * this->z_x(pt)  - this->y_z(pt)  * this->z_xz(pt))
            - this->x_y(pt)   * (this->y_xzz(pt) * this->z_z(pt)  + this->y_xz(pt) * this->z_zz(pt)
                               + this->y_xz(pt)  * this->z_zz(pt) + this->y_x(pt)  * this->z_zzz(pt) 
                               - this->y_zzz(pt) * this->z_x(pt)  - this->y_zz(pt) * this->z_xz(pt) 
                               - this->y_zz(pt)  * this->z_xz(pt) - this->y_z(pt)  * this->z_xzz(pt))
            + this->x_zzz(pt) * (this->y_x(pt)   * this->z_y(pt)  - this->y_y(pt)  * this->z_x(pt))
            + this->x_zz(pt)  * (this->y_xz(pt)  * this->z_y(pt)  + this->y_x(pt)  * this->z_yz(pt) 
                               - this->y_yz(pt)  * this->z_x(pt)  - this->y_y(pt)  * this->z_xz(pt))
            + this->x_zz(pt)  * (this->y_xz(pt)  * this->z_y(pt)  + this->y_x(pt)  * this->z_yz(pt)
                               - this->y_yz(pt)  * this->z_x(pt)  - this->y_y(pt)  * this->z_xz(pt))                                          
            + this->x_z(pt)   * (this->y_xzz(pt) * this->z_y(pt)  + this->y_xz(pt) * this->z_yz(pt) 
                               + this->y_xz(pt)  * this->z_yz(pt) + this->y_x(pt)  * this->z_yzz(pt)
                               - this->y_yzz(pt) * this->z_x(pt)  - this->y_yz(pt) * this->z_xz(pt) 
                               - this->y_yz(pt)  * this->z_xz(pt) - this->y_y(pt)  * this->z_xzz(pt));
    default:
      assert(0);
      break;
  }
}

} // namespace doffem
} // namespace hiflow

#endif
