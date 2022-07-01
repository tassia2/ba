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

#include <memory>

#include "fem/fe_instance.h"
#include "fem/fe_reference.h"
#include "fem/fe_transformation.h"
#include "fem/reference_cell.h"
#include "dof/dof_fem_types.h"
#include "dof/dof_impl/dof_container.h"
#include "dof/dof_impl/dof_container_rt_bdm.h"
#include "dof/dof_impl/dof_container_lagrange.h"
#include "fem/ansatz/ansatz_space.h"
#include "fem/ansatz/ansatz_sum.h"
#include "fem/ansatz/ansatz_p_line_lagrange.h"
#include "fem/ansatz/ansatz_p_tri_lagrange.h"
#include "fem/ansatz/ansatz_aug_p_tri_mono.h"
#include "fem/ansatz/ansatz_aug_p_tet_mono.h"
#include "fem/ansatz/ansatz_skew_aug_p_quad_mono.h"
#include "fem/ansatz/ansatz_skew_aug_p_hex_mono.h"
#include "fem/ansatz/ansatz_p_tet_lagrange.h"
#include "fem/ansatz/ansatz_pyr_lagrange.h"
#include "fem/ansatz/ansatz_q_quad_lagrange.h"
#include "fem/ansatz/ansatz_q_hex_lagrange.h"
#include "fem/cell_trafo/cell_transformation.h"
#include "fem/cell_trafo/linear_line_transformation.h"
#include "fem/cell_trafo/linear_triangle_transformation.h"
#include "fem/cell_trafo/linear_tetrahedron_transformation.h"
#include "fem/cell_trafo/linear_pyramid_transformation.h"
#include "fem/cell_trafo/bilinear_quad_transformation.h"
#include "fem/cell_trafo/trilinear_hexahedron_transformation.h"
#include "fem/cell_trafo/linear_quad_transformation.h"
#include "fem/cell_trafo/linear_hexahedron_transformation.h"
#include "mesh/geometric_tools.h"

#define nFORCE_NONLINEAR_TRAFO

namespace hiflow {
namespace doffem {

template <class DataType, int DIM>
void create_RTBDM_container (DofContainerType type, 
                             size_t deg, 
                             CRefCellSPtr<DataType, DIM> ref_cell, 
                             DofContainerUPtr<DataType, DIM>& dofs);

template <>
void create_RTBDM_container (DofContainerType type, 
                             size_t deg, 
                             CRefCellSPtr<float, 1> ref_cell, 
                             DofContainerUPtr<float, 1>& dofs) 
{
  std::cout << "RT BDM elements are not available for 1D " << std::endl;
  quit_program();
}

template <>
void create_RTBDM_container (DofContainerType type, 
                             size_t deg, 
                             CRefCellSPtr<double, 1>  ref_cell, 
                             DofContainerUPtr<double, 1>& dofs) 
{
  std::cout << "RT BDM elements are not available for 1D " << std::endl;
  quit_program();
}

template <>
void create_RTBDM_container (DofContainerType type, 
                             size_t deg, 
                             CRefCellSPtr<float, 2>  ref_cell, 
                             DofContainerUPtr<float, 2>& dofs) 
{
  // create DofContainer corresponding to BDM elements
  auto tmp_dofs = new DofContainerRTBDM<float, 2>(ref_cell);
  tmp_dofs->init(deg, type);
  dofs.reset(tmp_dofs);
}

template <>
void create_RTBDM_container (DofContainerType type, 
                             size_t deg, 
                             CRefCellSPtr<double, 2>  ref_cell, 
                             DofContainerUPtr<double, 2>& dofs) 
{
  // create DofContainer corresponding to BDM elements
  auto tmp_dofs = new DofContainerRTBDM<double, 2>(ref_cell);
  tmp_dofs->init(deg, type);
  dofs.reset(tmp_dofs);
}

template <>
void create_RTBDM_container (DofContainerType type, 
                             size_t deg, 
                             CRefCellSPtr<float, 3>  ref_cell, 
                             DofContainerUPtr<float, 3>& dofs) 
{
  // create DofContainer corresponding to BDM elements
  auto tmp_dofs = new DofContainerRTBDM<float, 3>(ref_cell);
  tmp_dofs->init(deg, type);
  dofs.reset(tmp_dofs);
}

template <>
void create_RTBDM_container (DofContainerType type, 
                             size_t deg, 
                             CRefCellSPtr<double, 3>  ref_cell, 
                             DofContainerUPtr<double, 3>& dofs) 
{
  // create DofContainer corresponding to BDM elements
  auto tmp_dofs = new DofContainerRTBDM<double, 3>(ref_cell);
  tmp_dofs->init(deg, type);
  dofs.reset(tmp_dofs);
}

// degrees[c][d] : polynomial degree of component c in spatial direction d
template <class DataType, int DIM>
void create_lagrange_element (mesh::CellType::Tag topo_cell_type,
                              size_t degree, 
                              size_t nb_comp,
                              AnsatzSpaceUPtr<DataType, DIM>& ansatz,
                              DofContainerUPtr<DataType, DIM>& dofs,
                              CRefCellSPtr<DataType, DIM>& ref_cell,
                              FETrafoUPtr<DataType, DIM>& fe_trafo,
                              bool& is_modal_basis,
                              FEConformity& conform,
                              bool force_p_element) 
{
 
  // create ansatz space object and select referenc cell
  switch (topo_cell_type)
  {
    case mesh::CellType::LINE:
      ref_cell = std::make_shared<RefCellLineStd<DataType, DIM> >();
      ansatz = std::make_unique< PLineLag<DataType, DIM> >(ref_cell);
      break;
    case mesh::CellType::TRIANGLE:
      ref_cell = std::make_shared<RefCellTriStd<DataType, DIM> >();
      ansatz = std::make_unique< PTriLag<DataType, DIM> > (ref_cell);
      break;
    case mesh::CellType::QUADRILATERAL:
      ref_cell = std::make_shared<RefCellQuadStd<DataType, DIM> >();
      if(force_p_element)
      {
        ansatz = std::make_unique< PTriLag<DataType, DIM> >(ref_cell);
      }
      else
      {
        ansatz = std::make_unique< QQuadLag<DataType, DIM> >(ref_cell);
      }
      break;
    case mesh::CellType::TETRAHEDRON:
      ref_cell = std::make_shared<RefCellTetStd<DataType, DIM> >();
      ansatz = std::make_unique< PTetLag<DataType, DIM> >(ref_cell);
      break;
    case mesh::CellType::HEXAHEDRON:
      ref_cell = std::make_shared<RefCellHexStd<DataType, DIM> >();
      if(force_p_element)
      {
        ansatz = std::make_unique< PTetLag<DataType, DIM> >(ref_cell);
      }
      else
      {
        ansatz = std::make_unique< QHexLag<DataType, DIM> >(ref_cell);
      }
      break;
    case mesh::CellType::PYRAMID:
      ref_cell = std::make_shared<RefCellPyrStd<DataType, DIM> >();
      ansatz = std::make_unique< PyrLag<DataType, DIM> >(ref_cell);
      break;
    default:
      std::cout << "Unexpected reference cell type " << std::endl;
      quit_program();
  }
  ansatz->init(degree, nb_comp);
  
  // create DofContainer correpsonding to Lagrange elements
  auto tmp_dofs = new DofContainerLagrange<DataType, DIM>(ref_cell);
  tmp_dofs->init(degree, nb_comp, force_p_element);
  dofs.reset(tmp_dofs);

  // create FE transformation
  fe_trafo = std::make_unique < FETransformationStandard<DataType, DIM> >();
  
  is_modal_basis = true;
  conform = FEConformity::H1;
}

template <class DataType, int DIM>
void create_BDM_element (mesh::CellType::Tag topo_cell_type, 
                         size_t deg, 
                         AnsatzSpaceUPtr<DataType, DIM>& ansatz,
                         DofContainerUPtr<DataType, DIM>& dofs,
                         CRefCellSPtr<DataType, DIM>& ref_cell,
                         FETrafoUPtr<DataType, DIM>& fe_trafo,
                         bool& is_modal_basis,
                         FEConformity& conform) 
{
  // create ansatz space object     
  switch (topo_cell_type)
  {
    case mesh::CellType::TRIANGLE:
      ref_cell = std::make_shared<RefCellTriStd<DataType, DIM> >();
      ansatz = std::make_unique< PTriLag<DataType, DIM> >(ref_cell);
      break;
    case mesh::CellType::QUADRILATERAL:
    {  
      ref_cell = std::make_shared<RefCellQuadStd<DataType, DIM> >();
      // P_(deg) space on quadrilateral with two components  with Lagrange basis functions
      auto tmp_ansatz_space_1 = new PTriLag<DataType, DIM>(ref_cell);
      tmp_ansatz_space_1->init(deg, 2);

      // skew augmented P space on quadrilateral with monomial basis functions
      auto tmp_ansatz_space_2 = new SkewAugPQuadMono<DataType, DIM>(ref_cell);
      tmp_ansatz_space_2->init(deg);

      // ansatz space is sum of two polynomial spaces
      auto ansatz_space_sum = new AnsatzSpaceSum<DataType, DIM> (ref_cell);
      
      CAnsatzSpaceSPtr<DataType, DIM> ansatz_1(tmp_ansatz_space_1);
      CAnsatzSpaceSPtr<DataType, DIM> ansatz_2(tmp_ansatz_space_2);
      ansatz_space_sum->init(ansatz_1, ansatz_2, AnsatzSpaceType::BDM);
      ansatz.reset(ansatz_space_sum);
      break;
    }
    case mesh::CellType::TETRAHEDRON:
      //NOT_YET_IMPLEMENTED;
      ref_cell = std::make_shared<RefCellTetStd<DataType, DIM> >();
      ansatz = std::make_unique< PTetLag<DataType, DIM>> (ref_cell);
      break;
    case mesh::CellType::HEXAHEDRON:
    { 
      //NOT_YET_IMPLEMENTED; 
      ref_cell = std::make_shared<RefCellHexStd<DataType, DIM> >();
      // P_(deg) space on hexahedron with three components  with Lagrange basis functions
      auto ansatz_space_1 = new PTetLag<DataType, DIM>(ref_cell);
      ansatz_space_1->init(deg, 3);

      // skew augmented P space on hexahedron with monomial basis functions
      auto ansatz_space_2 = new SkewAugPHexMono<DataType, DIM>(ref_cell);
      ansatz_space_2->init(deg);

      // ansatz space is sum of two polynomial spaces
     auto ansatz_space_sum = new AnsatzSpaceSum<DataType, DIM> (ref_cell);

      CAnsatzSpaceSPtr<DataType, DIM> ansatz_1(ansatz_space_1);
      CAnsatzSpaceSPtr<DataType, DIM> ansatz_2(ansatz_space_2);
      ansatz_space_sum->init(ansatz_1, ansatz_2, AnsatzSpaceType::BDM);

      ansatz.reset(ansatz_space_sum);
      break;
    }
    default:
      std::cout << "Unexpected reference cell type " << std::endl;
      quit_program();
  }
     
  if (topo_cell_type != mesh::CellType::QUADRILATERAL && topo_cell_type != mesh::CellType::HEXAHEDRON)
  { 
    ansatz->init(deg, DIM);
  }

  create_RTBDM_container (DofContainerType::BDM, deg, ref_cell, dofs);
   
  // create e transformation
  fe_trafo = std::make_unique< FETransformationContraPiola<DataType, DIM> >();
    
  is_modal_basis = false;
  conform = FEConformity::HDIV;
}

template <class DataType, int DIM>
void create_RT_element (mesh::CellType::Tag topo_cell_type, 
                         size_t deg, 
                         AnsatzSpaceUPtr<DataType, DIM>& ansatz,
                         DofContainerUPtr<DataType, DIM>& dofs,
                         CRefCellSPtr<DataType, DIM>& ref_cell,
                         FETrafoUPtr<DataType, DIM>& fe_trafo,
                         bool& is_modal_basis,
                         FEConformity& conform) 
{
  // create ansatz space object     
  if (topo_cell_type == mesh::CellType::TRIANGLE)
  {
    ref_cell = CRefCellSPtr<DataType, DIM> (new RefCellTriStd<DataType, DIM>());
      
    auto tmp_ansatz_space_1 = new PTriLag<DataType, DIM>(ref_cell);
    auto tmp_ansatz_space_2 = new AugPTriMono<DataType, DIM>(ref_cell);
    tmp_ansatz_space_1->init(deg, DIM);
    tmp_ansatz_space_2->init(deg);
    auto ansatz_sum = new AnsatzSpaceSum<DataType, DIM>(ref_cell);

    CAnsatzSpaceSPtr<DataType, DIM> ansatz_1(tmp_ansatz_space_1);
    CAnsatzSpaceSPtr<DataType, DIM> ansatz_2(tmp_ansatz_space_2);
    ansatz_sum->init(ansatz_1, ansatz_2, AnsatzSpaceType::RT);
    
    ansatz.reset(ansatz_sum);
  }
  else if (topo_cell_type == mesh::CellType::QUADRILATERAL)
  {
    ref_cell = CRefCellSPtr<DataType, DIM> (new RefCellQuadStd<DataType, DIM>());

    QQuadLag<DataType, DIM>* ansatz_ = new QQuadLag<DataType, DIM>(ref_cell);

    std::vector< std::vector <size_t> > degrees;
    degrees.resize(2);
    degrees[0].resize(2);
    degrees[1].resize(2);
    degrees[0][0] = deg + 1;
    degrees[0][1] = deg;
    degrees[1][0] = deg;
    degrees[1][1] = deg + 1;
    ansatz_->init(degrees);

    ansatz.reset(ansatz_); 
  }
  else if (topo_cell_type == mesh::CellType::TETRAHEDRON)
  {
    //NOT_YET_IMPLEMENTED;
    ref_cell = CRefCellSPtr<DataType, DIM> (new RefCellTetStd<DataType, DIM>());
         
    auto tmp_ansatz_space_1 = new PTetLag<DataType, DIM>(ref_cell);
    auto tmp_ansatz_space_2 = new AugPTetMono<DataType, DIM>(ref_cell);
    tmp_ansatz_space_1->init(deg, DIM);
    tmp_ansatz_space_2->init(deg);
    CAnsatzSpaceSPtr<DataType, DIM> ansatz_1(tmp_ansatz_space_1);
    CAnsatzSpaceSPtr<DataType, DIM> ansatz_2(tmp_ansatz_space_2);

    auto ansatz_sum = new AnsatzSpaceSum<DataType, DIM>(ref_cell);
    ansatz_sum->init(ansatz_1, ansatz_2, AnsatzSpaceType::RT);

    ansatz.reset(ansatz_sum);
  }
  else if (topo_cell_type == mesh::CellType::HEXAHEDRON)
  {
    ref_cell = CRefCellSPtr<DataType, DIM> (new RefCellHexStd<DataType, DIM>());

    QHexLag<DataType, DIM>* ansatz_ = new QHexLag<DataType, DIM>(ref_cell);

    std::vector< std::vector <size_t> > degrees;
    degrees.resize(3);
    degrees[0].resize(3);
    degrees[1].resize(3);
    degrees[2].resize(3);
    degrees[0][0] = deg + 1;
    degrees[0][1] = deg;
    degrees[0][2] = deg;
    degrees[1][0] = deg;
    degrees[1][1] = deg + 1;
    degrees[1][2] = deg;
    degrees[2][0] = deg;
    degrees[2][1] = deg;
    degrees[2][2] = deg + 1;
    ansatz_->init(degrees);
    
    ansatz.reset(ansatz_); 
  }
  else 
  {
    std::cout << "Unexpected reference cell type " << std::endl;
    quit_program();
  }
  
  create_RTBDM_container (DofContainerType::RT, deg, ref_cell, dofs);
   
  // create e transformation
  fe_trafo  =std::make_unique< FETransformationContraPiola<DataType, DIM> > ();
    
  is_modal_basis = false;
  conform = FEConformity::HDIV;
}


template <class DataType, int DIM>
void FEInstance<DataType, DIM>::clear()
{
  this->lagrange_only_ = true;
  this->ref_elements_.clear();
  this->ansatz_spaces_.clear();
  this->dof_containers_.clear();
  this->fe_trafos_.clear();
  this->max_fe_conform_.clear();
  this->nb_nonlin_trafos_ = 0;
}

template <class DataType, int DIM>
CRefElementSPtr<DataType, DIM> FEInstance<DataType, DIM>::get_fe (size_t fe_id) const
{
  assert (fe_id < this->ref_elements_.size());
  return this->ref_elements_[fe_id];
}

template <class DataType, int DIM>
size_t FEInstance<DataType, DIM>::add_fe ( FEType fe_type, 
                                           mesh::CellType::Tag topo_cell_type, 
                                           const std::vector<int> &param )
{
  // create instance of a Finite Element, defined by general type, topology of reference cell and additional parameters
  
  // first check whether instance was already created. If yes, return corresponding index
  for (size_t l=0; l<added_fe_types_.size(); ++l)
  {
    if (fe_type == added_fe_types_[l])
    {
      if (topo_cell_type == added_cell_types_[l])
      {
        if (param == added_params_[l])
        {
          return l;
        }
      }
    }
  }
  
  AnsatzSpaceUPtr<DataType, DIM> ansatz(nullptr);
  DofContainerUPtr<DataType, DIM> dofs(nullptr);
  FETrafoUPtr<DataType, DIM> fe_trafo (nullptr);
  CRefCellSPtr<DataType, DIM> ref_cell (nullptr);
  RefCellType cell_type;
    
  // Note: modal_basis = true <=> the dofs sigma_i and the basis functions phi_j in ansatz satisfy dof_i(phi_j) = delta_{ij} by construction
  bool modal_basis = false;
  FEConformity conform = FEConformity::L2;
  
  // This is the only location that has to be modified after having implemented a new type of element
  if (fe_type == FEType::LAGRANGE)
  {
    assert (param.size() == 1);
    assert (param[0] >= 0);
    create_lagrange_element(topo_cell_type, param[0], 1, ansatz, dofs, ref_cell, fe_trafo, modal_basis, conform, false);
  }
  else if (fe_type == FEType::LAGRANGE_VECTOR)
  {
    assert (param.size() == 1);
    assert (param[0] >= 0);
    create_lagrange_element(topo_cell_type, param[0], DIM, ansatz, dofs, ref_cell, fe_trafo, modal_basis, conform, false);
  }
  else if (fe_type == FEType::LAGRANGE_P)
  {
    assert (param.size() == 1);
    assert (param[0] >= 0);
    create_lagrange_element(topo_cell_type, param[0], 1, ansatz, dofs, ref_cell, fe_trafo, modal_basis, conform, true);
  }
  else if (fe_type == FEType::BDM)
  {  
    this->lagrange_only_ = false;
    assert (param.size() == 1);
    assert (param[0] > 0);
    create_BDM_element(topo_cell_type, param[0], ansatz, dofs, ref_cell, fe_trafo, modal_basis, conform);
  }
  else if (fe_type == FEType::RT)
  { 
    this->lagrange_only_ = false;
    assert (param.size() == 1);
    assert (param[0] >= 0);
    create_RT_element(topo_cell_type, param[0], ansatz, dofs, ref_cell, fe_trafo, modal_basis, conform);
  }
  else
  {
    std::cout << "Unexpected fe type " << std::endl;
    quit_program();
  }
  // initialize reference element
  assert (ansatz.get() != nullptr);
  assert (dofs.get() != nullptr);
  assert (fe_trafo.get() != nullptr);
  assert (ref_cell);
  
  assert (ansatz->type() != AnsatzSpaceType::NOT_SET);
  assert (dofs->type() != DofContainerType::NOT_SET);

  int ind =  this->fe_trafos_.size();
  this->fe_trafos_.push_back(std::move(fe_trafo));
  this->dof_containers_.push_back(std::move(dofs));
  this->ansatz_spaces_.push_back(std::move(ansatz));

  RefElementUPtr<DataType, DIM> ref_elem = std::make_unique<RefElement<DataType, DIM> >();
  ref_elem->init(this->ansatz_spaces_[ind], this->dof_containers_[ind], this->fe_trafos_[ind], modal_basis, fe_type);
  ref_elem->set_instance_id(this->ref_elements_.size());
  this->ref_elements_.push_back(std::move(ref_elem));
  this->ref_cells_.push_back(ref_cell);
  
  assert (ansatz == 0);

  this->max_fe_conform_.push_back(conform);
  
  this->added_fe_types_.push_back(fe_type);
  this->added_cell_types_.push_back(topo_cell_type);
  this->added_params_.push_back(param);
  
  return this->ref_elements_.size()-1;
}

template <class DataType, int DIM>
CellTrafoSPtr<DataType, DIM> FEInstance<DataType, DIM>::create_cell_trafo (size_t fe_id, 
                                                                          std::vector<DataType> coord_vtx,
                                                                          const mesh::Entity &cell,
                                                                          const std::vector< mesh::MasterSlave >& period) const
//CellTrafoSPtr<DataType, DIM> FEInstance<DataType, DIM>::create_cell_trafo (size_t fe_id, int align_number) const
{
  assert (fe_id < this->ref_elements_.size());
  CRefCellSPtr<DataType, DIM> ref_cell = this->ref_cells_[fe_id];
  
  CellTrafoSPtr<DataType, DIM> cell_trafo(nullptr);

  switch(ref_cell->type())
  {
    case RefCellType::LINE_STD:
      cell_trafo = std::make_shared< LinearLineTransformation<DataType, DIM> > (ref_cell, period);
      break;
    case RefCellType::TRI_STD:
      cell_trafo = std::make_shared< LinearTriangleTransformation<DataType, DIM> > (ref_cell, period);
      break;
    case RefCellType::TET_STD:
      cell_trafo = std::make_shared< LinearTetrahedronTransformation<DataType, DIM> > (ref_cell, period);
      break;
    case RefCellType::PYR_STD:
      cell_trafo = std::make_shared< LinearPyramidTransformation<DataType, DIM> > (ref_cell, period);
      break;
    case RefCellType::QUAD_STD:
#ifdef FORCE_NONLINEAR_TRAFO
      if (mesh::is_parallelogram(coord_vtx) && false)
#else
      if (mesh::is_parallelogram(coord_vtx))
#endif
      {
        cell_trafo = std::make_shared< LinearQuadTransformation<DataType, DIM> > (ref_cell, period);
      }
      else
      {
        cell_trafo = std::make_shared< BiLinearQuadTransformation<DataType, DIM> > (ref_cell, period);
        this->nb_nonlin_trafos_++;
      }
      break;
    case RefCellType::HEX_STD:
#ifdef FORCE_NONLINEAR_TRAFO
      if (mesh::is_parallelepiped(coord_vtx) && false)
#else
      if (mesh::is_parallelepiped(coord_vtx))
#endif
      {
        cell_trafo = std::make_shared< LinearHexahedronTransformation<DataType, DIM> > (ref_cell, period);
      }
      else
      {
        cell_trafo = std::make_shared< TriLinearHexahedronTransformation <DataType, DIM> > (ref_cell, period);
        this->nb_nonlin_trafos_++;
      }
      break;
    default:
      std::cout << "Unexpected ref cell type " << std::endl;
      quit_program();
  }
  cell_trafo->reinit(coord_vtx, cell);
  return cell_trafo;
}

template class FEInstance< float, 3 >;
template class FEInstance< float, 2 >;
template class FEInstance< float, 1 >;

template class FEInstance< double, 3 >;
template class FEInstance< double, 2 >;
template class FEInstance< double, 1 >;

} // namespace doffem
} // namespace hiflow
