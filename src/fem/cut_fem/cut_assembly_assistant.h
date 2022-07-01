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

/// \author Philipp Gerstner

#ifndef HIFLOW_CUTFEM_ASSEMBLY_H_
#define HIFLOW_CUTFEM_ASSEMBLY_H_

#include "assembly/function_values.h"
#include "common/log.h"
#include "common/vector_algebra_descriptor.h"
#include "common/array_tools.h"
#include "fem/cell_trafo/cell_transformation.h"
#include "fem/cell_trafo/linear_triangle_transformation.h"
#include "fem/cell_trafo/bilinear_quad_transformation.h"
#include "fem/cell_trafo/linear_line_surface_transformation.h"
#include "fem/cut_fem/cell_cut.h"
#include "mesh/cell_type.h"
#include "mesh/entity.h"
#include "mesh/geometric_tools.h"
#include "mesh/types.h"
#include "mesh/iterator.h"
#include "quadrature/quadrature.h"
#include "quadrature/custom_quadrature_type.h"
#include "quadrature/mapped_quadrature_type.h"
#include "space/element.h"
#include "space/vector_space.h"

#include <map>
#include <cmath>
#include <vector>
#include <numeric>
#include <set>

namespace hiflow {

namespace doffem {

static int CUTTED_CELLS = 0;

enum class CutAsmMode 
{
  Cell = 0,
  Facet = 1
};

enum class CellDomain 
{
  LOWER = 1,
  HIGHER = 2,
  ZERO = 0,
  NOT_SET = -1
};

// convention for domain side:
// CellDomain::LOWER              -> DomainSide::LOW_XI 
// CellDomain::HIGHER             -> DomainSide::HIGH_XI 
// CellDomain::ZERO : xi < thresh -> DomainSide::LOW_XI
//                    xi > thresh -> DomainSide::HIGH_XI



template <class DataType, int DIM, class DomainFunction>
class CutAssemblyAssistant
{
public:
  typedef EntityNumber VertexIndex;
  using VertexCoord = typename StaticLA<DIM, DIM, DataType>::ColVectorType;
  using vec = typename StaticLA<DIM, DIM, DataType>::ColVectorType;
  using PRmat = typename StaticLA<DIM, DIM-1, DataType>::MatrixType;
  
  CutAssemblyAssistant ()
  {
    Xi_ = nullptr;
    ref_if_line_ = RefCellSPtr<DataType, DIM-1>(new RefCellLineStd<DataType, DIM-1>);
    ref_if_tri_  = RefCellSPtr<DataType, DIM-1>(new RefCellTriStd<DataType, DIM-1>);
    ref_if_quad_  = RefCellSPtr<DataType, DIM-1>(new RefCellQuadStd<DataType, DIM-1>);
    
    ref_cell_tri_ = RefCellSPtr<DataType, DIM>(new RefCellTriStd<DataType, DIM>); 
    ref_cell_quad_ = RefCellSPtr<DataType, DIM>(new RefCellQuadStd<DataType, DIM>); 
    ref_cell_tet_ = RefCellSPtr<DataType, DIM>(new RefCellTetStd<DataType, DIM>);
  }
  
  ~CutAssemblyAssistant()
  {
    ref_if_line_.reset();
    ref_if_tri_.reset();
    ref_if_quad_.reset();
    ref_cell_tri_.reset();
    ref_cell_quad_.reset();
    ref_cell_tet_.reset();
  }

  // outward unit normal w.r.t. Domain A
  inline vec nA(int q) const
  {
    return this->nf_[q];
  }

  // outward unit normal w.r.t. Domain B
  inline vec nB(int q) const
  {
    return (-1.) * this->nf_[q];
  }
  
  inline vec n(DomainSide side, int q) const
  {
    switch (side)
    {
      case DomainSide::LOW_XI:
        return this->nf_[q];
        break;
      case DomainSide::HIGH_XI:
        return (-1.) * this->nf_[q];
        break; 
      default:
        assert(false);
        break;
    }
    return vec();
  }
  
  void init (const DomainFunction& xi, 
             const DataType distinction_threshold,
             const DataType vertex_cut_tol)
  {
    this->Xi_ = &(xi);
    this->dist_thresh_ = distinction_threshold;
    this->vertex_cut_tol_ = vertex_cut_tol;
  }

  void set_interface_base_quad (int order)
  {
    this->base_interface_quadrature_line_.set_quadrature_by_order("GaussLine", order);
    this->base_interface_quadrature_quad_.set_quadrature_by_order("GaussQuadrilateral", order);
    this->base_interface_quadrature_tri_.set_quadrature_by_order("GaussTriangle", order);
  }
  
  void set_subcell_base_quad (int order)
  {
    this->base_subcell_quadrature_quad_.set_quadrature_by_order("GaussQuadrilateral", order);
    this->base_subcell_quadrature_tri_.set_quadrature_by_order("GaussTriangle", order);
    this->base_subcell_quadrature_tet_.set_quadrature_by_order("GaussTetrahedron", order);
  }
  
  void set_interface_base_quad (int size_line, int size_tri, int size_quad)
  {
    this->base_interface_quadrature_line_.set_quadrature_by_size("GaussLine", size_line);
    this->base_interface_quadrature_quad_.set_quadrature_by_size("GaussQuadrilateral", size_quad);
    this->base_interface_quadrature_tri_.set_quadrature_by_size("GaussTriangle", size_tri);
  }
  
  void set_subcell_base_quad (int size_tri, int size_quad, int size_tet)
  {
    this->base_subcell_quadrature_quad_.set_quadrature_by_size("GaussQuadrilateral", size_quad);
    this->base_subcell_quadrature_tri_.set_quadrature_by_size("GaussTriangle", size_tri);
    this->base_subcell_quadrature_tet_.set_quadrature_by_size("GaussTetrahedron", size_tet);
  }
   
  void init_dg_cell_flags(int num_cells)
  {
    this->is_dg_cell_cut_.clear();
    this->is_dg_cell_cut_.resize(num_cells, false);
  }
  
  int num_dg_cell_cuts() const;
  
  void set_dg_cell_cuts (const std::vector<bool>& dg_flags)
  {
    this->is_dg_cell_cut_ = dg_flags;
  }
    
  const std::vector<bool>& get_dg_cell_cuts() const
  {
    return this->is_dg_cell_cut_;
  }

  std::vector<bool>& get_dg_cell_cuts()
  {
    return this->is_dg_cell_cut_;
  }
  
protected:
  // to be called from application local assembler, before AssemblyAssistang::initialize_for_element is called
  // (see example PoissonInterface)
  // mode:             cell (0) or internal interface (1) assembly
  // assemble_contrib: if true, current element contributes to linear system.
  //                   if false, upcoming local assembly can be skipped
  // cut_type:         0 :  proper cut          -> get 2 subcells, each having positive volume
  //                   1 :  cut plane coincides 
  //                        with cell interface -> need to call DG interface assembler
  //                   2 :  single cut point, 
  //                        or cut edge (in 3D) -> do nothing
  //                   -1:  pathological case   -> perhaps multiple cuts per cell, cannot continue
  
  void initialize_for_element(const Element< DataType, DIM > &element,
                              const Quadrature< DataType > &cell_quadrature,
                              const CutAsmMode mode,
                              bool& assemble_contrib,
                              CellDomain cell_domain,
                              CutType& cut_type);
                              
  void get_quad_point(const DataType xq, const DataType yq, const DataType zq, vec& pt) const;
   
  void map_quad_point(const DataType xq, const DataType yq, const DataType zq, 
                      CCellTrafoSPtr<DataType, DIM> c_trafo,
                      vec& pt) const;
                                                                                                  
  CutType cut_cell(const Element< DataType, DIM > &element);
  
  void compute_subcell_quad(const Element< DataType, DIM > &element);
  
  void init_interface_transformations(const Element< DataType, DIM > &element);
  void compute_interface_quad(const Element< DataType, DIM > &element);
  void determine_interface_normals(const Element< DataType, DIM > &element);

  void create_composition_quad ( const QuadString& quad_name,
                                 const mesh::CellType::Tag cell_tag,
                                 const std::vector<SingleQuadrature<DataType>>& sub_quads,
                                 Quadrature<DataType>& comp_quad);

  void adapt_quad_weights (CCellTrafoSPtr<DataType, DIM> c_trafo,
                           const std::vector<DataType>& xq,
                           const std::vector<DataType>& yq,
                           const std::vector<DataType>& zq,
                           std::vector<DataType>& wq) const;
                           
  std::vector<DataType> compute_cell_area(const Element< DataType, DIM > &element,
                                          CCellTrafoSPtr<DataType, DIM> c_trafo,
                                          const Quadrature< DataType > &cell_quadrature);
                                        
  bool check_cell_area(const Element< DataType, DIM > &element,
                       CCellTrafoSPtr<DataType, DIM> c_trafo,
                       const Quadrature< DataType > &cell_quadrature);
                                                                          
  CellCutter<DataType, DIM> cell_cutter_;
    
  const DomainFunction* Xi_;
  DataType dist_thresh_;
  
  DataType KA_, KB_, K_;
  DataType diam_K_;

  // Jacobian of surface transformation
  FunctionValues< PRmat > Df_;
  FunctionValues< vec > nf_;
  
  //vec nF_;
  //Mat< DIM, DIM - 1, DataType > R_;
  
  std::vector< std::vector< VertexCoord > > cutpoint_coords_;
  std::vector< DomainSide > subcell_domain_;
  std::vector< std::vector<mesh::Id> > subcell_v_index_;
  std::vector< std::vector<VertexCoord > > subcell_v_coord_;
  std::vector< std::vector< mesh::Id > > cutpoint_ids_;
  std::vector< mesh::CellType::Tag > subcell_tag_;

                             
  std::vector<bool> is_dg_cell_cut_;
  
  std::vector< SurfaceTrafoSPtr<DataType, DIM-1, DIM> > iface_trafos_;

  std::unordered_map<mesh::Id, SubCellPointType> vertex_types_;

  Quadrature<DataType> interface_quad_;
  Quadrature<DataType> cell_quad_;
  
  std::vector<SingleQuadrature<DataType> > subcell_quads_;
  std::vector<SingleQuadrature<DataType> > subiface_quads_;
   
  std::vector<DataType> xq_;
  std::vector<DataType> yq_;
  std::vector<DataType> zq_;
  std::vector<DataType> wq_;
  std::vector<DomainSide> dq_;
  
  std::vector<DataType> tmp_quad_x_;
  std::vector<DataType> tmp_quad_y_;
  std::vector<DataType> tmp_quad_z_;
  std::vector<DataType> tmp_quad_w_;
  
  // DIM = 2  
  Quadrature< DataType > base_interface_quadrature_line_;
  Quadrature< DataType > base_subcell_quadrature_tri_;
  Quadrature< DataType > base_subcell_quadrature_quad_;
  
  // DIM = 3
  Quadrature< DataType > base_interface_quadrature_tri_;
  Quadrature< DataType > base_interface_quadrature_quad_;
  Quadrature< DataType > base_subcell_quadrature_tet_;

  RefCellSPtr<DataType, DIM-1> ref_if_line_;
  RefCellSPtr<DataType, DIM-1> ref_if_tri_;
  RefCellSPtr<DataType, DIM-1> ref_if_quad_;

  RefCellSPtr<DataType, DIM> ref_cell_tri_;
  RefCellSPtr<DataType, DIM> ref_cell_quad_;
  RefCellSPtr<DataType, DIM> ref_cell_tet_;

  DataType vertex_cut_tol_;
  DataType Khat_;
};

template <class DataType, int DIM, class DomainFunction>
void CutAssemblyAssistant<DataType, DIM, DomainFunction>::get_quad_point(const DataType xq, const DataType yq, const DataType zq, 
                                                                         vec& pt) const 
{
  if constexpr (DIM == 1)
  {
    pt.set(0, xq);
  }
  else if constexpr (DIM == 2)
  {
    pt.set(0, xq);
    pt.set(1, yq);
  }
  else if constexpr (DIM == 3)
  {
    pt.set(0, xq);
    pt.set(1, yq);
    pt.set(2, zq);
  }
  else 
  {
    assert (false);
  }  
}

template <class DataType, int DIM, class DomainFunction>
void CutAssemblyAssistant<DataType, DIM, DomainFunction>::map_quad_point(const DataType xq, const DataType yq, const DataType zq, 
                                                                         CCellTrafoSPtr<DataType, DIM> c_trafo,
                                                                         vec& pt) const 
{
  vec ref_pt;
  this->get_quad_point(xq, yq, zq, ref_pt);
  c_trafo->transform(ref_pt, pt);
}

template <class DataType, int DIM, class DomainFunction>
void CutAssemblyAssistant<DataType, DIM, DomainFunction>::adapt_quad_weights (CCellTrafoSPtr<DataType, DIM> c_trafo,
                                                                              const std::vector<DataType>& xq,
                                                                              const std::vector<DataType>& yq,
                                                                              const std::vector<DataType>& zq,
                                                                              std::vector<DataType>& wq) const 
{
  const int size = wq.size();
  assert (size > 0);
  for (int q=0; q != size; ++q)
  {
    // get base quad point in reference cell
    vec ref_pt;
    this->get_quad_point(xq[q], yq[q], zq[q], ref_pt);
    const DataType dJ_q = std::abs(c_trafo->detJ(ref_pt));
    
    wq[q] *= dJ_q;
  }
}

template <class DataType, int DIM, class DomainFunction>
void CutAssemblyAssistant<DataType, DIM, DomainFunction>::compute_subcell_quad(const Element< DataType, DIM > &element)
{
  const int num_subcells = subcell_tag_.size();
    
  std::vector<Quadrature<DataType>* > base_subcell_quad(num_subcells, nullptr);
  std::vector<CellTrafoSPtr<DataType, DIM> > subcell_trafo(num_subcells, 0);
  CCellTrafoSPtr<DataType, DIM> c_trafo = element.get_cell_transformation();
      
  if constexpr (DIM == 2)
  {
    for (int i = 0; i!= num_subcells; ++i)
    {
      if (subcell_tag_[i] == mesh::CellType::TRIANGLE)
      {
        subcell_trafo[i] = CellTrafoSPtr<DataType, DIM>(new LinearTriangleTransformation<DataType, DIM>(ref_cell_tri_));
        base_subcell_quad[i] = &(this->base_subcell_quadrature_tri_);
      }
      else if (subcell_tag_[i] == mesh::CellType::QUADRILATERAL)
      {
        subcell_trafo[i] = CellTrafoSPtr<DataType, DIM>(new BiLinearQuadTransformation<DataType, DIM>(ref_cell_quad_));
        base_subcell_quad[i] = &(this->base_subcell_quadrature_quad_);
      }
      else
      {
        assert (false);
      }
    }
  }
  else if constexpr (DIM == 3)
  {
    for (int i = 0; i!= num_subcells; ++i)
    {
      if (subcell_tag_[i] == mesh::CellType::TETRAHEDRON)
      {
        subcell_trafo[i] = CellTrafoSPtr<DataType, DIM>(new LinearTetrahedronTransformation<DataType, DIM>(ref_cell_tet_));
        base_subcell_quad[i] = &(this->base_subcell_quadrature_tet_);
      }
      else
      {
        assert (false);
      }
    }
  }
       
  subcell_quads_.clear();
  subcell_quads_.resize(num_subcells);
    
  for (int i = 0; i!= num_subcells; ++i)
  {
    // TODO: maybe filter out subcells of very small size 
    // (leading to extremely small determinant of jacobian of subcell transformation)
    subcell_trafo[i]->reinit(subcell_v_coord_[i]);
    
    // map quadrature from base_quad (fitting to subcell) to outer cell
    tmp_quad_x_.clear();
    tmp_quad_y_.clear();
    tmp_quad_z_.clear();
    tmp_quad_w_.clear();
    SubCellQuadratureMapping<DataType, DIM> quad_mapping (subcell_trafo[i], c_trafo);
    quad_mapping.map_quadrature_data(*base_subcell_quad[i], 
                                     tmp_quad_x_, tmp_quad_y_, tmp_quad_z_, tmp_quad_w_);
    
    // adapt quad weight by determinant of subcell transformation
    this->adapt_quad_weights(subcell_trafo[i], tmp_quad_x_, tmp_quad_y_, tmp_quad_z_, tmp_quad_w_);
      
    // create custom quadrature
    subcell_quads_[i].set_custom_quadrature(base_subcell_quad[i]->order(),
                                            base_subcell_quad[i]->get_cell_tag(),
                                            tmp_quad_x_, tmp_quad_y_, tmp_quad_z_, tmp_quad_w_);
  }
}

template <class DataType, int DIM, class DomainFunction>
void CutAssemblyAssistant<DataType, DIM, DomainFunction>::init_interface_transformations(const Element< DataType, DIM > &element)
{
  // create mapping [0,1]       -> interface cut line 
  //             or [0,1]x[0,1] -> interface cut plane

  const int nb_sub_faces = this->cutpoint_coords_.size();
  this->iface_trafos_.clear();
  this->iface_trafos_.resize(nb_sub_faces);

  for (int i=0; i!=nb_sub_faces; ++i)
  {
    SurfaceTrafoSPtr<DataType, DIM-1, DIM> s_trafo;
    if constexpr (DIM == 2)
    {
      assert (cutpoint_coords_[i].size() == 2);
      s_trafo = SurfaceTrafoSPtr<DataType, DIM-1, DIM>(new LinearLineSurfaceTransformation<DataType, DIM-1, DIM>(ref_if_line_));
    }
    else if constexpr (DIM == 3)
    {
      if (cutpoint_coords_[i].size() == 3)
      {
        s_trafo = SurfaceTrafoSPtr<DataType, DIM-1, DIM>(new LinearTriangleSurfaceTransformation<DataType, DIM-1, DIM>(ref_if_tri_));
      }
      else if (cutpoint_coords_[i].size() == 4)
      {
        s_trafo = SurfaceTrafoSPtr<DataType, DIM-1, DIM>(new BiLinearQuadSurfaceTransformation<DataType, DIM-1, DIM>(ref_if_quad_));
      }
      else
      {
        assert (false);
      }
    }
    s_trafo->reinit(cutpoint_coords_[i]);
    this->iface_trafos_[i] = s_trafo;
  }
}

template <class DataType, int DIM, class DomainFunction>
void CutAssemblyAssistant<DataType, DIM, DomainFunction>::compute_interface_quad(const Element< DataType, DIM > &element)
{
  // create mapping [0,1]       -> interface cut line 
  //             or [0,1]x[0,1] -> interface cut plane

  const int nb_sub_faces = this->cutpoint_coords_.size();
  CCellTrafoSPtr<DataType, DIM> c_trafo = element.get_cell_transformation();
  subiface_quads_.clear();
  subiface_quads_.resize(nb_sub_faces);

  for (int i=0; i!=nb_sub_faces; ++i)
  {
    Quadrature<DataType>* base_if_quad = nullptr;
    SurfaceTrafoSPtr<DataType, DIM-1, DIM> s_trafo = this->iface_trafos_[i];
    if constexpr (DIM == 2)
    {
      assert (cutpoint_coords_[i].size() == 2);
      base_if_quad = &(this->base_interface_quadrature_line_);
    }
    else if constexpr (DIM == 3)
    {
      if (cutpoint_coords_[i].size() == 3)
      {
        base_if_quad = &(this->base_interface_quadrature_tri_);
      }
      else if (cutpoint_coords_[i].size() == 4)
      {
        base_if_quad = &(this->base_interface_quadrature_quad_);
      }
      else
      {
        assert (false);
      }
    }
  
    // map quadrature
    SubEntityQuadratureMapping<DataType, DIM-1, DIM> quad_mapping (s_trafo, c_trafo);

    tmp_quad_x_.clear();
    tmp_quad_y_.clear();
    tmp_quad_z_.clear();
    tmp_quad_w_.clear();
    quad_mapping.map_quadrature_data(*base_if_quad, tmp_quad_x_, tmp_quad_y_, tmp_quad_z_, tmp_quad_w_);
  
    // create custom quadrature
    this->subiface_quads_[i].set_custom_quadrature(base_if_quad->order(),
                                                   base_if_quad->get_cell_tag(),
                                                   tmp_quad_x_, tmp_quad_y_, tmp_quad_z_, tmp_quad_w_);
  }
}

template <class DataType, int DIM, class DomainFunction>
void CutAssemblyAssistant<DataType, DIM, DomainFunction>::determine_interface_normals(const Element< DataType, DIM > &element)
{
  CCellTrafoSPtr<DataType, DIM> c_trafo = element.get_cell_transformation();

  // get coordinate of any cell vertex that lies not on current inteface
  int vertex_nr = 0;
  
  for (auto v_it = element.get_cell().begin_incident(0), 
       e_it = element.get_cell().end_incident(0);
       v_it != e_it; ++v_it)
  {
    assert (this->vertex_types_.find(v_it->id()) != this->vertex_types_.end());
    if (this->vertex_types_[v_it->id()] == SubCellPointType::NoCutPoint )
    {
      break;
    }

    vertex_nr++;
  }
  
  std::vector< DataType > cell_pts;
  element.get_cell().get_coordinates (cell_pts, vertex_nr);

  vec v_pt (cell_pts);

  const DataType xi = this->Xi_->evaluate (element.get_cell(), v_pt);      
  DomainSide v_dom = DomainSide::NOT_SET;

  LOG_DEBUG(CELLCUT_DBG, "sub cell " << 0 << " vertex " << vertex_nr << " and Xi = " << xi << " -> domain A");

  if ( xi < dist_thresh_)
    v_dom = DomainSide::LOW_XI;
  else
    v_dom = DomainSide::HIGH_XI;


  const int nb_sub_faces = this->cutpoint_coords_.size();
  std::vector<int> sub_num_q(nb_sub_faces, 0);
  int num_q = 0;

  for (int i=0; i!= nb_sub_faces; ++i)
  {
    num_q += this->subiface_quads_[i].size();
    sub_num_q[i] = this->subiface_quads_[i].size();
  }

  this->Df_.clear();
  this->nf_.clear();
  this->Df_.zeros(num_q);
  this->nf_.zeros(num_q);

  // loop through sub-interfaces
  int q = 0;
  for (int i=0; i!= nb_sub_faces; ++i)
  {
    auto s_trafo = this->iface_trafos_[i];
    const auto& s_quad = this->subiface_quads_[i];

    // loop through subintef quad points 
    for (int qs = 0; qs != sub_num_q[i]; ++qs)
    {
      // quad point on reference surface
      Vec<DIM-1, DataType> xr_q;
      s_quad.get_point(qs, xr_q);
      LOG_DEBUG(CELLCUT_DBG, "interface index = " << i << " quad index = " << q << " :: " << 
                "xr_q = " << xr_q);

      // quad point on phyiscal surface
      vec x_q;
      s_trafo->transform(xr_q, x_q);
      LOG_DEBUG(CELLCUT_DBG, "interface index = " << i << " quad index = " << q << " :: " << 
                "x_q = " << x_q);

      // Jacobian of surface transformation
      s_trafo->J(xr_q, this->Df_[q]);
    
      // normal of surface 
      vec n_q;
      s_trafo->normal(xr_q, n_q);   
      LOG_DEBUG(CELLCUT_DBG, "interface index = " << i << " quad index = " << q << " :: " << 
                "n_q = " << n_q);

      // determine orientation of normal 
      // q_2_v directs from the interface to a subcell vertex
      const DataType vx_dot_n = dot(v_pt - x_q, n_q);

      DataType sign_n = 1.;

      if (vx_dot_n > 1e-12)
      {
        // n_q points towards v_pt
        if (v_dom == DomainSide::LOW_XI)  // n_q is outward normal w.r.t. Domain B   
          sign_n = -1.;
        else if (v_dom == DomainSide::HIGH_XI) // n_q is outward normal w.r.t. Domain A
          sign_n = 1.;
        else 
          assert (false);
      }
      else if (vx_dot_n < -1e-12)
      {
        // n_q points away from v_pt
        if (v_dom == DomainSide::LOW_XI) // n_q is outward normal w.r.t. Domain A
          sign_n = 1.;
        else if (v_dom == DomainSide::HIGH_XI) // n_q is outward normal w.r.t. Domain B
          sign_n = -1.;
        else 
          assert (false);
      }
      else  
      {
        assert (false);
      }

      this->nf_[q] = sign_n * n_q;
      q++;
    }
  }
}

template <class DataType, int DIM, class DomainFunction>
CutType CutAssemblyAssistant<DataType, DIM, DomainFunction>::cut_cell(const Element< DataType, DIM > &element)
{       
  // cut_types:
  // -1:  pathological case   -> perhaps multiple cuts per cell, cannot continue
  // 0 :  proper cut          -> get 2 subcells, each having positive volume
  // 1 :  cut plane coincides 
  //      with cell interface -> need to call DG interface assembler
  // 2 :  single cut point, 
  //      or cut edge (in 3D) -> do nothing
  // 3 :  no cut at all       -> do nothing
  
  LOG_DEBUG(CELLCUT_DBG, "cell entity " << element.get_cell());
      
  // get cut points on edges
  bool decompose_non_simplex = true;
  CutType cut_type = cell_cutter_.cut_cell(element.get_cell(), 
                                           Xi_, 
                                           dist_thresh_, 
                                           this->vertex_cut_tol_, 
                                           decompose_non_simplex,
                                           cutpoint_ids_, 
                                           cutpoint_coords_, 
                                           subcell_domain_, 
                                           subcell_tag_, 
                                           subcell_v_index_, 
                                           subcell_v_coord_,
                                           vertex_types_);

#ifdef CUT_DBG_OUT
  std::cout << std::endl;
  std::cout << "cell index " << element.cell_index() << " cut cell " << as_integer(cut_type) << std::endl;
  std::cout << "vertices " << std::endl;
  std::vector<DataType> v_coords;
  element.get_cell().get_coordinates(v_coords);
  for (int i=0; i!= v_coords.size() / DIM; ++i)
  {
    for (int d=0; d!= DIM; ++d)
    {
      std::cout << v_coords[i*DIM+d] << " ";
    }
    std::cout << " || ";
  }
  std::cout << std::endl;
  std::cout << " cut points " << std::endl;
  for (int i=0; i!=cutpoint_coords_.size(); ++i)
  {
    LOG_DEBUG(CELLCUT_DBG, "cut facet " << i << " : ");
    for (int v=0; v!= cutpoint_coords_[i].size(); ++v)
    {
      LOG_DEBUG(CELLCUT_DBG, "       cut point " << v << " : " << cutpoint_coords_[i][v] );
    }
  }
  //if (cutpoint_coords_.size() == 2)
  //{
  //  LOG_DEBUG(CELLCUT_DBG, "interface length = " << distance(cutpoint_coords_[1], cutpoint_coords_[0]));
  //}
  
  for (int i=0; i!=subcell_v_coord_.size(); ++i)
  {
    LOG_DEBUG(CELLCUT_DBG, "subcell " << i << " : tag " << subcell_tag_[i] << ", " 
                  << string_from_range(subcell_v_coord_[i].begin(),subcell_v_coord_[i].end()) );
  }
#endif

  if (cut_type == CutType::Pathologic) 
  {
    assert(false);
    return CutType::Pathologic;
  }
 
  if (this->is_dg_cell_cut_.size() != 0)
  {
    assert (element.cell_index() < this->is_dg_cell_cut_.size());
    if (cut_type == CutType::Interface)
    {
      this->is_dg_cell_cut_[element.cell_index()] = true;
    }
    else
    {
      this->is_dg_cell_cut_[element.cell_index()] = false;
    }
  }

  LOG_DEBUG(CELLCUT_DBG, "cut type " << as_integer(cut_type) );
    
  if ((cut_type != CutType::Proper) && (cut_type != CutType::Interface))
  {
    return cut_type;
  }

  return cut_type;
}

template <class DataType, int DIM, class DomainFunction>
std::vector<DataType> CutAssemblyAssistant<DataType, DIM, DomainFunction>::compute_cell_area(const Element< DataType, DIM > &element,
                                                                                             CCellTrafoSPtr<DataType, DIM> c_trafo,
                                                                                             const Quadrature< DataType > &cell_quadrature)
{
  std::vector<DataType> res(3,0.);

  // loop over quad points in cell quad
  const int size = cell_quadrature.size();
  for (int q=0; q != size; ++q)
  {
    // get cell quad point
    vec ref_pt;
    this->get_quad_point(cell_quadrature.x(q), cell_quadrature.y(q), cell_quadrature.z(q), ref_pt);
    
    const DataType dJ_q = std::abs(c_trafo->detJ(ref_pt)); 
    const DataType w_q = cell_quadrature.w(q);
      
    res[0] += w_q * dJ_q;
  }
    
  DataType KA = 0.;
  DataType KB = 0.;
  const int nb_subcell = this->subcell_quads_.size();
    
  for (int i=0; i != nb_subcell; ++i)
  {
    const int size = this->subcell_quads_[i].size();
    assert (size > 0);
      
    DataType integral = 0.;
    for (int q=0; q != size; ++q)
    {
      const DataType w_q = this->subcell_quads_[i].w(q);
      integral += w_q;
    }
    if (this->subcell_domain_[i] == DomainSide::LOW_XI)
    {
      LOG_DEBUG(CELLCUT_DBG, "add area " << integral << " of side A to subcell " << i);
      KA += integral;
    }
    else if (this->subcell_domain_[i] == DomainSide::HIGH_XI)
    {
      LOG_DEBUG(CELLCUT_DBG, "add area " << integral << " of side B to subcell " << i);
      KB += integral;
    }
    else
    {
      assert (false);
    }
  }
     
  res[1] = KA;
  res[2] = KB;
  
  assert (std::abs(res[0] - (res[1] + res[2])) / res[0] < 1e-6);
  
  LOG_DEBUG(CELLCUT_DBG, "AREA: " << res[0] << " ?= " << res[1] + res[2] << " = " << res[1] << " + " << res[2]);

  return res;
}

template <class DataType, int DIM, class DomainFunction>
bool CutAssemblyAssistant<DataType, DIM, DomainFunction>::check_cell_area(const Element< DataType, DIM > &element,
                                                                          CCellTrafoSPtr<DataType, DIM> c_trafo,
                                                                          const Quadrature< DataType > &cell_quadrature)
{
  DataType area_1 = 0.;
    
  // loop over quad points in cell quad
  const int size = cell_quadrature.size();
  for (int q=0; q != size; ++q)
  {
    // get cell quad point
    vec ref_pt;
    this->get_quad_point(cell_quadrature.x(q), cell_quadrature.y(q), cell_quadrature.z(q), ref_pt);
    
    const DataType dJ_q = std::abs(c_trafo->detJ(ref_pt)); 
    const DataType w_q = cell_quadrature.w(q);
      
    area_1 += w_q * dJ_q;
  }
    
  const int size2 = this->cell_quad_.size();
  assert (size2 > 0);
  assert (size == size2);
  
  DataType area_2 = 0.;
  for (int q=0; q != size2; ++q)
  {
    const DataType w_q = this->cell_quad_.w(q);
    area_2 += w_q;
  }
    
  bool res = (std::abs(area_1 - area_2) / area_1 < 1e-6);
  
  LOG_DEBUG(CELLCUT_DBG, "AREA: " << area_1 << " ?= " << area_2);

  return res;
}

template <class DataType, int DIM, class DomainFunction>
void CutAssemblyAssistant<DataType, DIM, DomainFunction>::initialize_for_element(const Element< DataType, DIM > &element,
                                                                                 const Quadrature< DataType > &cell_quadrature,
                                                                                 const CutAsmMode mode,
                                                                                 bool& assemble_contrib,
                                                                                 CellDomain cell_domain,
                                                                                 CutType& cut_type) 
{
  // mode: 0: cell assembly, 1: inner facet assembly
  CCellTrafoSPtr<DataType, DIM> c_trafo = element.get_cell_transformation();  
  const int cell_index = element.cell_index();
  cut_type = CutType::None;
    
  // determine in which way the cell is cut into pieces
  if (cell_domain == CellDomain::ZERO)
  {
    CUTTED_CELLS++;
    cut_type = this->cut_cell(element);
  }
    
  if (cut_type == CutType::Interface)
  {
    // DG case
    if (mode == CutAsmMode::Facet)
    {
      // no inner facet contribution
      assemble_contrib = false;
      return;
    }
  }
  if (cut_type == CutType::Point || cut_type == CutType::None)
  {
    // pathological case: single corner cut or no cut at all
    if (mode == CutAsmMode::Facet)
    {
      // no inner facet contribution
      assemble_contrib = false;
      return;
    }
  }
    
  this->Khat_ = 0.;
  const int size = cell_quadrature.size();
  for (int q=0; q != size; ++q)
  {
    Khat_ += cell_quadrature.w(q);
  }

  this->diam_K_ = c_trafo->cell_diameter();
  //this->diam_K_ = compute_entity_diameter<DataType, DIM> (element.get_cell());

  if (( (cut_type != CutType::Pathologic) && (cut_type != CutType::Proper)) && (mode == CutAsmMode::Cell))
  {
    // non-proper cell cut, i.e. one side has measure 0
    // -> integrate over whole cell
    auto name = cell_quadrature.name();
    const int num_q = cell_quadrature.size();
    
    // get weights of corresponding to size determined by outer quadselectfunctions
    this->wq_ = cell_quadrature.weights(num_q);
      
    this->adapt_quad_weights(element.get_cell_transformation(), 
                             cell_quadrature.qpts_x(num_q),
                             cell_quadrature.qpts_y(num_q),
                             cell_quadrature.qpts_z(num_q),
                             this->wq_);
                               
    this->cell_quad_.set_custom_quadrature(name,
                                           cell_quadrature.order(), 
                                           cell_quadrature.get_cell_tag(),
                                           cell_quadrature.qpts_x(num_q),
                                           cell_quadrature.qpts_y(num_q),
                                           cell_quadrature.qpts_z(num_q),
                                           this->wq_);
                                             
    this->dq_.clear();
    if (cell_domain == CellDomain::LOWER)
    {
      this->dq_.resize(this->cell_quad_.size(), DomainSide::LOW_XI);
    }
    else if (cell_domain == CellDomain::HIGHER)
    {
      this->dq_.resize(this->cell_quad_.size(), DomainSide::HIGH_XI);
    }
    else if (cell_domain == CellDomain::ZERO)
    {
      vec pt;
      this->map_quad_point(cell_quadrature.x(0), cell_quadrature.y(0), cell_quadrature.z(0), c_trafo, pt);
      
      assert (this->Xi_ != nullptr);
      const DataType xi_q = this->Xi_->evaluate(element.get_cell(), pt);
      if (xi_q < dist_thresh_)
      {
        this->dq_.resize(this->cell_quad_.size(), DomainSide::LOW_XI);
      }
      else
      {
        this->dq_.resize(this->cell_quad_.size(), DomainSide::HIGH_XI);
      }
    }
#ifndef NDEBUG
    assert (this->check_cell_area(element, c_trafo, cell_quadrature));
#endif
    assemble_contrib = true;
    return;
  }
  
  // cut_type = 0 -> proper cut with two subcells of positive measure each  
  assert (cut_type == CutType::Proper);
      
  // create quadrature on subcells
  this->compute_subcell_quad(element); 
    
  // create quadrature on interface intersection facet
  if (mode == CutAsmMode::Facet)
  {
    this->init_interface_transformations(element);
    this->compute_interface_quad(element);
    this->determine_interface_normals(element);  
    QuadString quad_name = "cell_cutface";
    quad_name.append_value(cell_index);
    this->create_composition_quad(quad_name, mesh::CellType::NOT_SET, this->subiface_quads_, this->interface_quad_);
  }
         
  // create joint quad for both subcells
  if (mode == CutAsmMode::Cell)
  {
    QuadString quad_name = "cell_interface";
    quad_name.append_value(cell_index);

    this->create_composition_quad(quad_name, mesh::CellType::NOT_SET, this->subcell_quads_, this->cell_quad_);
    
    const int num_q = this->cell_quad_.size();  
    const int nb_subcell = this->subcell_quads_.size();
    this->dq_.clear();
    this->dq_.reserve(num_q);
    
    for (int i=0; i!=nb_subcell; ++i)
    {
      DomainSide subcell_dom = this->subcell_domain_[i];
      assert (subcell_dom != DomainSide::NOT_SET);

      const int quad_size = this->subcell_quads_[i].size();
      for (int q=0; q!=quad_size; ++q)
      {
        this->dq_.push_back(subcell_dom);
      }
    }
  }
    
  // compute weights for averaging of coefficients  
  std::vector<DataType> area = this->compute_cell_area(element, c_trafo, cell_quadrature);
  
  this->K_ = area[0];
  this->KA_ = area[1];
  this->KB_ = area[2];
  
  assert (this->KA_ > 0.);
  assert (this->KB_ > 0.);
  assert (std::abs(KA_ + KB_ - K_) / K_ < 1e-6);
  
  assemble_contrib = true;
}
  
template <class DataType, int DIM, class DomainFunction>
void CutAssemblyAssistant<DataType, DIM, DomainFunction>::create_composition_quad
  ( const QuadString& quad_name,
    const mesh::CellType::Tag cell_tag,
    const std::vector<SingleQuadrature<DataType>>& sub_quads,
    Quadrature<DataType>& comp_quad)
{
  int num_q = 0;
  int order = 1e3;
  const int nb_sub = sub_quads.size();
  for (int i=0; i!=nb_sub; ++i)
  {
    num_q += sub_quads[i].size();
    order = std::min(order, sub_quads[i].order());
  }
      
  this->xq_.clear();
  this->yq_.clear();
  this->zq_.clear();
  this->wq_.clear();
    
  this->xq_.reserve(num_q);
  this->yq_.reserve(num_q);
  this->zq_.reserve(num_q);
  this->wq_.reserve(num_q);
    
  for (int i=0; i!=nb_sub; ++i)
  {
    const int quad_size = sub_quads[i].size();
    for (int q=0; q!=quad_size; ++q)
    {
      this->xq_.push_back(sub_quads[i].x(q));
      this->yq_.push_back(sub_quads[i].y(q));
      this->zq_.push_back(sub_quads[i].z(q));
      this->wq_.push_back(sub_quads[i].w(q));
    }
  }
  
  comp_quad.set_custom_quadrature(quad_name,
                                  order, 
                                  cell_tag,
                                  this->xq_,
                                  this->yq_,
                                  this->zq_,
                                  this->wq_);
}



template <class DataType, int DIM, class DomainFunction>
int CutAssemblyAssistant<DataType, DIM, DomainFunction>::num_dg_cell_cuts() const
{
  assert (this->is_dg_cell_cut_.size() > 0);
  int sum = 0;
  for (int c = 0; c!= this->is_dg_cell_cut_.size(); ++c)
  {
    if (this->is_dg_cell_cut_[c])
    {
      sum++;
    }
  }
  return sum;
}


} // namespace mesh
} // namespace hiflow

#endif
