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
#include <algorithm>

#include "common/array_tools.h"
#include "dof/dof_impl/dof_container_rt_bdm.h"
#include "dof/dof_impl/dof_container.h"
#include "dof/dof_impl/dof_functional_cell_moment.h"
#include "dof/dof_impl/dof_functional_facet_normal_moment.h"
#include "fem/ansatz/ansatz_sum.h"
#include "fem/ansatz/ansatz_p_line_lagrange.h"
#include "fem/ansatz/ansatz_p_tri_lagrange.h"
#include "fem/ansatz/ansatz_p_tet_lagrange.h"
#include "fem/ansatz/ansatz_skew_aug_p_tri_mono.h"
#include "fem/ansatz/ansatz_skew_aug_p_tet_mono.h"
#include "fem/ansatz/ansatz_q_quad_lagrange.h"
#include "fem/ansatz/ansatz_q_hex_lagrange.h"
#include "fem/ansatz/ansatz_space.h"
#include "fem/reference_cell.h"
#include "mesh/entity.h"
#include "quadrature/quadrature.h"

namespace hiflow {
namespace doffem {

int DBG_LVL_RTBDM = 0;

template<class DataType, int DIM>
DofContainerRTBDM<DataType, DIM>::~DofContainerRTBDM()
{
  this->clear();
}

template<class DataType, int DIM>
void DofContainerRTBDM<DataType, DIM>::clear ()
{
  DofContainer<DataType, DIM>::clear();
  
  this->qc_x_.clear();
  this->qf_x_.clear();
  this->qc_w_.clear();
  this->qf_w_.clear();
  this->ds_.clear();
  this->test_vals_c_.clear();
  this->trial_vals_c_.clear();
  this->test_vals_f_.clear();
  this->trial_vals_f_.clear();
  
  /*
  if (this->ref_facet_ != nullptr)
  {
    delete this->ref_facet_;
  }
  */
  this->facet_test_space_.reset();
  this->cell_test_space_.reset();
  this->cell_test_space_1_.reset();
  this->cell_test_space_2_.reset();
}

template<class DataType, int DIM>
void DofContainerRTBDM<DataType, DIM>::init (size_t degree, DofContainerType type)
{
  assert ( DIM == 2 || DIM == 3);
  assert (type == DofContainerType::BDM || type == DofContainerType::RT);
  
  assert (degree >= 1 || type == DofContainerType::RT);
  
  // this routine should be called only once
  if (this->initialized_ && this->degree_ == degree && type == this->type_)
  {
    return;
  }
  
  this->clear();
  this->type_ = type;
  int min_deg_for_cell_dofs = -1;
  
  if (type == DofContainerType::BDM)
  {
    this->name_ = "DOF_BDM";
    min_deg_for_cell_dofs = 2;
  }
  else
  {
    this->name_ = "DOF_RT";
    min_deg_for_cell_dofs = 1;
  }
      
  this->degree_ = degree;

  // initialize reference cell, quadratures and test spaces
  QuadString cell_quad_name;
  QuadString facet_quad_name;

  RefCellType ref_cell_type = this->ref_cell_->type();

  if (ref_cell_type == RefCellType::TRI_STD)
  {
    assert (DIM == 2);
    // P space on 1D line with Lagrange basis functions
    this->ref_facet_ = RefCellSPtr<DataType, DIM-1>(new RefCellLineStd<DataType, DIM-1>());
    auto tmp_f_space = new PLineLag<DataType, DIM-1> (this->ref_facet_);
    tmp_f_space->init(degree);
    facet_test_space_.reset(tmp_f_space);
    if (degree >= min_deg_for_cell_dofs)
    {
      if (this->type_ == DofContainerType::BDM)
      {
        // P_(deg-2) space on triangle with two components  with Lagrange basis functions
        auto tmp_space_1 = new PTriLag<DataType, DIM>(this->ref_cell_);
        tmp_space_1->init(degree-2, 2);
        // skew augmented P space on triangle  with monomial basis functions
        auto tmp_space_2 = new SkewAugPTriMono<DataType, DIM>(this->ref_cell_);
        tmp_space_2->init(degree-2);
        this->cell_test_space_1_.reset(tmp_space_1);
        this->cell_test_space_2_.reset(tmp_space_2);
        // cell test space is sum of two polynomial spaces
        auto cell_test_space_sum = new AnsatzSpaceSum<DataType, DIM> (this->ref_cell_);
        cell_test_space_sum->init(this->cell_test_space_1_, this->cell_test_space_2_, AnsatzSpaceType::P_AUG);
        
        this->cell_test_space_.reset(cell_test_space_sum);
      }
      else
      {
        // cell test space: P_(deg-1) space on triangle with two components  with Lagrange basis functions 
        auto tmp_cell_space = new PTriLag<DataType, DIM>(this->ref_cell_);
        tmp_cell_space->init(degree-1, 2);
        this->cell_test_space_.reset(tmp_cell_space);
      }
    }
    cell_quad_name = "GaussTriangle";
    facet_quad_name = "GaussLine";
    this->nb_facets_ = 3;
  }
  else if (ref_cell_type == RefCellType::QUAD_STD)
  {  
    assert (DIM == 2);
    // P space on 1D line with Lagrange basis functions
    this->ref_facet_ = RefCellSPtr<DataType, DIM-1>(new RefCellLineStd<DataType, DIM-1>());
    auto tmp_space = new PLineLag<DataType, DIM-1>(this->ref_facet_); 
    tmp_space->init(degree);
    this->facet_test_space_.reset(tmp_space);
    if (degree >= min_deg_for_cell_dofs)
    {
      if (this->type_ == DofContainerType::BDM)
      {
        // P_{deg-2} space on quadrilateral with two components with Lagrange basis functions
        auto tmp_cell_space = new PTriLag<DataType, DIM>(this->ref_cell_); 
        tmp_cell_space->init(degree-2, 2);
      
        this->cell_test_space_.reset(tmp_cell_space);
      }
      else
      {
        // cell test space: Q_{deg-1,deg} x Q_{deg,deg-1} space on quadrilateral with Lagrange basis functions
        auto tmp_cell_space = new QQuadLag<DataType, DIM>(this->ref_cell_); 
        std::vector< std::vector <size_t> > degrees;
        degrees.resize(2);
        degrees[0].resize(2);
        degrees[1].resize(2);
        degrees[0][0] = degree - 1;
        degrees[0][1] = degree;
        degrees[1][0] = degree;
        degrees[1][1] = degree - 1;
        tmp_cell_space->init(degrees);
        this->cell_test_space_.reset(tmp_cell_space);
      }
    }
    cell_quad_name = "GaussQuadrilateral";
    facet_quad_name = "GaussLine";
    this->nb_facets_ = 4;
  }
  else if (ref_cell_type == RefCellType::TET_STD)
  {
    assert (DIM == 3);
    // P space on triangle with Lagrange basis functions
    this->ref_facet_ = RefCellSPtr<DataType, DIM-1>(new RefCellTriStd<DataType, DIM-1>());
    auto tmp_space = new PTriLag<DataType, DIM-1> (this->ref_facet_);
    tmp_space->init(degree);
    this->facet_test_space_.reset(tmp_space);
    if (degree >= min_deg_for_cell_dofs)
    {
      if (this->type_ == DofContainerType::BDM)
      {
        // P_(deg-2) space on tetrahedron with three components  with Lagrange basis functions
        auto space_1 = new PTetLag<DataType, DIM>(this->ref_cell_);
        space_1->init(degree-2, 3);
        this->cell_test_space_1_.reset(space_1);
        // skew augmented P space on tetrahedron  with monomial basis functions
        auto space_2 = new SkewAugPTetMono<DataType, DIM>(this->ref_cell_);
        space_2->init(degree-1);
        this->cell_test_space_2_.reset(space_2);
        // cell test space is sum of two polynomial spaces
        auto cell_test_space_sum = new AnsatzSpaceSum<DataType, DIM> (this->ref_cell_);
        cell_test_space_sum->init(this->cell_test_space_1_, this->cell_test_space_2_, AnsatzSpaceType::P_AUG);
        this->cell_test_space_.reset(cell_test_space_sum);
      }
      else
      {
        // cell test space: P_(deg-1) space on tetrahedron with three components  with Lagrange basis functions 
        auto tmp_cell_space = new PTetLag<DataType, DIM>(this->ref_cell_);
        tmp_cell_space->init(degree-1, 3);
        this->cell_test_space_.reset(tmp_cell_space);
      }
    }
    assert (DIM == 3);
    cell_quad_name = "GaussTetrahedron";
    facet_quad_name = "GaussTriangle";
    this->nb_facets_ = 4;
  }
  else if (ref_cell_type == RefCellType::HEX_STD)
  {  
    assert (DIM == 3);
    this->ref_facet_ = RefCellSPtr<DataType, DIM-1>(new RefCellQuadStd<DataType, DIM-1>());
    if (this->type_ == DofContainerType::BDM)
    {
	    // P space on 2D quadrilateral with Lagrange basis functions
      auto tmp_space = new PTriLag<DataType, DIM-1> (this->ref_facet_);
      tmp_space->init(degree);
      this->facet_test_space_.reset(tmp_space);
    }
    else
    {
      // Q space on 2D quadrilateral with Lagrange basis functions
      auto tmp_space = new QQuadLag<DataType, DIM-1> (this->ref_facet_);
      tmp_space->init(degree);
      this->facet_test_space_.reset(tmp_space);
    }
    if (degree >= min_deg_for_cell_dofs)
    {
      if (this->type_ == DofContainerType::BDM)
      {
        // P_{deg-2} space on hexahedron with three components with Lagrange basis functions
        auto tmp_cell_space = new PTetLag<DataType, DIM>(this->ref_cell_); 
        tmp_cell_space->init(degree-2, 3);
        this->cell_test_space_.reset(tmp_cell_space);
      }
      else
      {
        // cell test space: Q_{deg-1,deg,deg} x Q_{deg,deg-1,deg} x Q_{deg,deg,deg-1} 
        // space on hexahedron with Lagrange basis functions
        auto tmp_cell_space = new QHexLag<DataType, DIM>(this->ref_cell_);
        std::vector< std::vector <size_t> > degrees;
        degrees.resize(3);
        degrees[0].resize(3);
        degrees[1].resize(3);
        degrees[2].resize(3);
        degrees[0][0] = degree - 1;
        degrees[0][1] = degree;
        degrees[0][2] = degree;
        degrees[1][0] = degree;
        degrees[1][1] = degree - 1;
        degrees[1][2] = degree;
        degrees[2][0] = degree;
        degrees[2][1] = degree;
        degrees[2][2] = degree - 1;
        tmp_cell_space->init(degrees);
        this->cell_test_space_.reset(tmp_cell_space);
      }
    }
    cell_quad_name = "GaussHexahedron";
    facet_quad_name = "GaussQuadrilateral";
    this->nb_facets_ = 6;
  }
  else
  {
    std::cerr << "Unexpected reference cell type " << std::endl;
    quit_program();
  }

  /// -----------------------------------
  /// initialize facet moment functionals
   
  // initialize quadrature TODO: correct maximal poly deg?
  int facet_order = std::max(this->facet_test_space_->max_deg(), static_cast<size_t>(3)) + degree;  
  this->facet_quad_.set_quadrature_by_order( facet_quad_name, facet_order);
  int num_qf = this->facet_quad_.size();

  //std::cout << this->facet_test_space_->max_deg() << " " << degree << " -> " << facet_order << " " << num_qf << std::endl;
  
  this->qf_x_.resize(num_qf);
  this->qf_w_.resize(num_qf, 0.);
  
  //std::cout << " facet quad " << std::endl;  
  for (size_t q=0; q<num_qf; ++q)
  {
    this->qf_x_[q].set(0, this->facet_quad_.x(q));
    if constexpr (DIM == 3)
    {
      this->qf_x_[q].set(1, this->facet_quad_.y(q));
    }
    this->qf_w_[q] = this->facet_quad_.w(q);
    //std::cout << this->qf_w_[q] << " ";
  }
  //std::cout << std::endl;
  
  // setup surface integration element  
  this->ds_.resize(nb_facets_);
  for (size_t facet_nr = 0; facet_nr < nb_facets_; ++facet_nr)
  {
    this->ds_[facet_nr].resize(num_qf, 0.);
    for (size_t q=0; q<num_qf; ++q)
    {
      // HIER ÜBERPRÜFEN, FÜR 1 STIMMT LÖSUNG mit EXAKTER Lösung überein, Warum?????
      this->ds_[facet_nr][q] = this->ref_cell_->ds(facet_nr, this->qf_x_[q]);
      //std::cout << facet_nr << " : " << this->ref_cell_->ds(facet_nr, this->qf_x_[q]) << std::endl;
    }
  }

  // compute values of test functions at quadrature points
  this->trial_vals_f_.resize(nb_facets_);
  this->test_vals_f_.clear();
  this->test_vals_f_.resize(num_qf);
  size_t size_f = this->facet_test_space_->weight_size();
  for (size_t q=0; q<num_qf; ++q)
  {
    this->test_vals_f_[q].resize(size_f, 0.);
    this->facet_test_space_->N(this->qf_x_[q], this->test_vals_f_[q]);
  }

  // get facet normal vectors
  const size_t nb_facet_dof = this->facet_test_space_->dim();
  
  // initialize dof functionals
  for (size_t facet_nr = 0; facet_nr < nb_facets_; ++facet_nr)
  {
    Coord facet_vec;
    this->ref_cell_->compute_facet_normal (facet_nr, facet_vec);
    
    LOG_DEBUG(DBG_LVL_RTBDM, "normal of facet " << facet_nr << " : " << facet_vec);

    for (size_t i=0; i<nb_facet_dof; ++i)
    {
      DofFacetNormalMoment<DataType, DIM> * dof = new DofFacetNormalMoment<DataType, DIM>();
      dof->init (this->facet_test_space_.get(),
                &this->test_vals_f_,
                &this->qf_w_,
                &this->ds_[facet_nr],
                 i,
                 this->ref_cell_,
                 facet_nr,
                 facet_vec);

      this->push_back( dof );
    } 
  }
  
  LOG_DEBUG(DBG_LVL_RTBDM, "Nb facet dofs for RT-BDM = " << nb_facets_ * nb_facet_dof);
  
  /// ----------------------------------
  /// initialize cell moment functionals
  if (degree >= min_deg_for_cell_dofs)
  {  
    // init cell quadrature
    int cell_order = 0;
    int num_qc = 0;

    cell_order = std::max(this->cell_test_space_->max_deg(), static_cast<size_t>(3)) + degree;
    this->cell_quad_.set_quadrature_by_order( cell_quad_name, cell_order);
    num_qc = this->cell_quad_.size();
    LOG_DEBUG(DBG_LVL_RTBDM, "max poly deg of cell test space = " << this->cell_test_space_->max_deg())
    LOG_DEBUG(DBG_LVL_RTBDM, "poly deg of ansatz space = " << degree);
    LOG_DEBUG(DBG_LVL_RTBDM, "cell quad order = " << cell_order << ", # quad points  = " << num_qc);
  
    // setup quadrature points and weights
    this->qc_x_.resize(num_qc);
    this->qc_w_.resize(num_qc, 0.);
  
    //std::cout << " cell quad " << std::endl;
    for (size_t q=0; q<num_qc; ++q)
    {
      this->qc_x_[q].set(0, this->cell_quad_.x(q));
      this->qc_x_[q].set(1, this->cell_quad_.y(q));
      if constexpr (DIM == 3)
      {
        this->qc_x_[q].set(2, this->cell_quad_.z(q));
      }
      this->qc_w_[q] = this->cell_quad_.w(q);
      //std::cout << this->qc_w_[q] << " ";
    }
    //std::cout << std::endl;

    // init cell test space
    size_t size_c = 0;
    size_c = this->cell_test_space_->weight_size();
    this->test_vals_c_.clear();
    this->test_vals_c_.resize(num_qc);
  
    for (size_t q=0; q<num_qc; ++q)
    {
      this->test_vals_c_[q].resize(size_c, 0.);
      this->cell_test_space_->N(this->qc_x_[q], this->test_vals_c_[q]);
    }  
    
    // init cell dof functionals
    const size_t nb_cell_dof = this->cell_test_space_->dim();
    for (size_t i=0; i<nb_cell_dof; ++i)
    {
      DofCellMoment<DataType, DIM> * dof = new DofCellMoment<DataType, DIM>();
      dof->init (this->cell_test_space_.get(),
                 &this->test_vals_c_,
                 &this->qc_w_,
                 i,
                 this->ref_cell_);

      this->push_back( dof );
    }
    LOG_DEBUG(DBG_LVL_RTBDM, "Nb cell dofs for RT-BDM = " << nb_cell_dof);
  }
  
  DofContainer<DataType, DIM>::init();
}

template<class DataType, int DIM>
void DofContainerRTBDM<DataType, DIM>::evaluate (FunctionSpace<DataType, DIM> const * space, 
                                                 const std::vector< cDofId > & dof_ids, 
                                                 std::vector< std::vector<DataType> >& dof_values ) const
{
  assert (this->initialized_);
  assert (space != nullptr);
  assert (space->ref_cell_type() == this->ref_cell_->type());

  const size_t facet_dim = this->facet_test_space_->dim();
  const size_t nb_facet_dof = facet_dim * this->nb_facets_;
  const size_t num_qf = this->qf_x_.size();
  const size_t num_qc = this->qc_x_.size();
  const size_t size_trial = space->weight_size();

  dof_values.resize(dof_ids.size());
  
  this->trial_vals_f_.clear();
  this->trial_vals_f_.resize(nb_facets_);
  
  this->trial_vals_c_.clear();

  int min_deg_for_cell_dofs = -1;
  if (this->type_ == DofContainerType::BDM)
  {
    min_deg_for_cell_dofs = 2;
  }
  else
  {
    min_deg_for_cell_dofs = 1;
  }
  
  // loop over dofs
#ifndef NDEBUG
  cDofId max_elem = 0;
  for (size_t l = 0; l < dof_ids.size(); ++l)
  {
    if (dof_ids[l] > max_elem)
      max_elem = dof_ids[l];
  }
  //std::cout << "nb facet dofs " << nb_facet_dof << " , number dofs " << max_elem+1 << std::endl;
#endif

  for (size_t l = 0; l < dof_ids.size(); ++l)
  {
    cDofId dof_id = dof_ids[l];
    assert (dof_id < this->dofs_.size());
    assert (this->dofs_[dof_id] != nullptr);
    
    set_to_zero(space->dim(), dof_values[l]);

    if (dof_id < nb_facet_dof)
    { // evaluate facet moment
   
      const size_t facet_nr = dof_id / facet_dim;

      // check if trial values have already been computed for current facet
      if (this->trial_vals_f_[facet_nr].size() == 0)
      {
        // evaluate all trial functions on all quad points on current facet
        this->trial_vals_f_[facet_nr].resize(num_qf);
      
        // loop over all facet quad points
        for (size_t q=0; q<num_qf; ++q)
        {
          this->trial_vals_f_[facet_nr][q].resize(size_trial, 0.);

          // evaluate basis functions at projected quad point
          space->N( this->ref_cell_->Tf(facet_nr, this->qf_x_[q]), this->trial_vals_f_[facet_nr][q] );
/*
          std::cout << " trial vals on facet " << facet_nr << " for dof " << dof_id << std::endl;
          std::cout << this->ref_cell_->Tf(facet_nr, this->qf_x_[q])[0] << ", " 
                    << this->ref_cell_->Tf(facet_nr, this->qf_x_[q])[1] << ": " << std::endl << "        ";
          for (size_t l=0; l<this->trial_vals_f_[facet_nr][q].size(); ++l)
          {
            std::cout << this->trial_vals_f_[facet_nr][q][l] << " ";
          }
          std::cout << std::endl << std::endl;;
*/
        }
      }
    
      DofFacetNormalMoment<DataType, DIM> * dof = dynamic_cast< DofFacetNormalMoment<DataType, DIM> * > (this->dofs_[dof_id]);
      assert (dof != nullptr);
      
      // passs trial values to facet moment
      dof->set_trial_values ( &this->trial_vals_f_[facet_nr] );
    
      // evaluate dof
      dof->evaluate (space, dof_values[l]);
/*      
      std::cout << " got dof values " << std::endl;
      for (size_t i=0; i<dof_values[l].size(); ++i)
      {
        std::cout << dof_values[l][i] << " ";
      }
      std::cout << std::endl << std::endl;
*/
    }
    else
    { // evaluate cell moment
      assert (this->degree_ >= min_deg_for_cell_dofs);

      const size_t nb_cell_dof = this->cell_test_space_->dim();
      // check if trial values have already been computed 
      // TODO: combine this with a check of the ansatz space ID
      if (this->trial_vals_c_.size() == 0)
      {
        // evaluate all trial functions on all quad points on current cell
        this->trial_vals_c_.resize(num_qc);
      
        // loop over all facet quad points
        for (size_t q=0; q<num_qc; ++q)
        {
          this->trial_vals_c_[q].resize(size_trial, 0.);
          
          // evaluate basis functions at projected quad point
          space->N( this->qc_x_[q], this->trial_vals_c_[q] );
        }
      }
    
      DofCellMoment<DataType, DIM> * dof = dynamic_cast< DofCellMoment<DataType, DIM> * > (this->dofs_[dof_id]);
      assert (dof != nullptr);
    
      // passs trial values to cell moment 
      dof->set_trial_values ( &this->trial_vals_c_ );
    
      // evaluate dof
      dof->evaluate (space, dof_values[l]);
    }
  }

}

template < class DataType, int DIM > 
void DofContainerRTBDM<DataType, DIM>::evaluate (RefCellFunction<DataType, DIM> const * func, 
                                                 const std::vector< cDofId > & dof_ids, 
                                                 std::vector< std::vector<DataType> >& dof_values ) const
{
  //TODO: check for nan

  assert (this->initialized_);
  const size_t nb_func = func->nb_func();
  const size_t nb_comp = func->nb_comp();
  
  dof_values.resize(dof_ids.size());

  // reset trial values
  this->trial_vals_c_.clear();
  this->trial_vals_f_.clear();
  this->trial_vals_f_.resize(nb_facets_);
  
  int min_deg_for_cell_dofs = -1;
  if (this->type_ == DofContainerType::BDM)
  {
    min_deg_for_cell_dofs = 2;
  }
  else
  {
    min_deg_for_cell_dofs = 1;
  }
  
  // loop over all specified dof functionals
  for (size_t l = 0; l < dof_ids.size(); ++l)
  {
    const cDofId dof_id = dof_ids[l];
    assert (dof_id >= 0);
    assert (dof_id < this->dim_);
    assert (this->dofs_[dof_id] != nullptr);
    
    set_to_zero(nb_func, dof_values[l]);
    
    const size_t facet_dim = this->facet_test_space_->dim();
    const size_t nb_facet_dof = facet_dim * this->nb_facets_;
    const size_t num_qf = this->qf_x_.size();
    const size_t num_qc = this->qc_x_.size();
    const size_t size_trial = nb_comp * nb_func ;

    if (dof_id < nb_facet_dof)
    { // evaluate facet moment
   
      const size_t facet_nr = dof_id / facet_dim;

      // check if trial values have already been computed for current facet
      if (this->trial_vals_f_[facet_nr].size() == 0)
      {
        // evaluate all trial functions on all quad points on current facet
        this->trial_vals_f_[facet_nr].resize(num_qf);
      
        // loop over all facet quad points
        for (size_t q=0; q<num_qf; ++q)
        {
          this->trial_vals_f_[facet_nr][q].resize(size_trial, 0.);

          // evaluate function at projected quad point
          func->evaluate( this->ref_cell_->Tf(facet_nr, this->qf_x_[q]), this->trial_vals_f_[facet_nr][q] );
/*
          std::cout << this->ref_cell_->Tf(facet_nr, this->qf_x_[q])[0] << ", " 
                    << this->ref_cell_->Tf(facet_nr, this->qf_x_[q])[1] << ": ";
          for (size_t l=0; l<this->trial_vals_f_[facet_nr][q].size(); ++l)
          {
            std::cout << this->trial_vals_f_[facet_nr][q][l] << " ";
          }
          std::cout << std::endl;
*/
        }
      }
      DofFacetNormalMoment<DataType, DIM> * dof = dynamic_cast< DofFacetNormalMoment<DataType, DIM> * > (this->dofs_[dof_id]);
      
      // passs trial values to facet moment 
      dof->set_trial_values ( &this->trial_vals_f_[facet_nr] );
    
      // evaluate dof functional and insert values for all functions into solution array
      dof->evaluate (func, 0, dof_values[l]);
    }
    else
    { // evaluate cell moment
      assert (this->degree_ >= min_deg_for_cell_dofs);

      const size_t nb_cell_dof = this->cell_test_space_->dim();
    
      // check if trial values have already been computed 
      if (this->trial_vals_c_.size() == 0)
      {
        // evaluate all trial functions on all quad points on current cell
        this->trial_vals_c_.resize(num_qc);
      
        // loop over all facet quad points
        for (size_t q=0; q<num_qc; ++q)
        {
          this->trial_vals_c_[q].resize(size_trial, 0.);

          // evaluate function at projected quad point
          func->evaluate( this->qc_x_[q], this->trial_vals_c_[q] );
        }
      }
    
      DofCellMoment<DataType, DIM> * dof = dynamic_cast< DofCellMoment<DataType, DIM> * > (this->dofs_[dof_id]);
      
      // passs trial values to facet moment 
      dof->set_trial_values ( &this->trial_vals_c_ );
      // evaluate dof functional and insert values for all functions into solution array
      dof->evaluate (func, 0, dof_values[l]);
    }
  }
  // reset trial values
  this->trial_vals_c_.clear();
  this->trial_vals_f_.clear();
}

template class DofContainerRTBDM< float, 3 >;
template class DofContainerRTBDM< float, 2 >;

template class DofContainerRTBDM< double, 3 >;
template class DofContainerRTBDM< double, 2 >;

} // namespace doffem
} // namespace hiflow
