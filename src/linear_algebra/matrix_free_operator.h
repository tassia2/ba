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

/// @author Philipp Gerstner

#ifndef HIFLOW_LINEAR_ALGEBRA_MATRIXFREE_OPERATOR_H
#define HIFLOW_LINEAR_ALGEBRA_MATRIXFREE_OPERATOR_H

#include "assembly/assembly_types.h"
#include "assembly/assembly_utils.h"
#include "assembly/assembly_routines.h"
#include "assembly/generic_assembly_algorithm.h"
#include "common/log.h"
#include "common/vector_algebra.h"
#include "linear_algebra/linear_operator.h"
#include "space/vector_space.h"
#include "dof/dof_fem_types.h"
#include <assert.h>
#include <cstddef>

namespace hiflow {
namespace la {

/// \brief Class for matrix-free linear operator
/// LAD: Type of vector 
/// N: dimension of stencil
/// actual mat-vec product is defined in Application object

template < class LAD, int DIM, class Application> 
class MatrixFree : public virtual LinearOperator<typename LAD::DataType> {
public:
  typedef typename LAD::VectorType VectorType;
  typedef typename LAD::DataType DataType;
  
  MatrixFree () :ap_(nullptr), space_(nullptr) {};
  
  ~MatrixFree  () {};
  
  void set_application(Application * ap) {
    assert(ap!=nullptr);
    this->ap_ = ap;
  }
  
  void set_space(VectorSpace<DataType, DIM>* space) {
    assert(space!= nullptr);
    this->space_ = space;    
  }
  
   void VectorMult(Vector< DataType >& in, Vector< DataType > *out) const{
    out->Update(); 
    assert(this->ap_!=nullptr);
    assert(this->space_!=nullptr);
    in.Update();
    this->ap_->VectorMult(space_, &in, out);
    out->Update();
  }
  Application * ap_;
  VectorSpace<DataType, DIM>* space_;
};

/// \brief Class for matrix-free linear operator
/// LAD: Type of vector 
/// N: dimension of stencil
/// mat-vec product is defined in local assembly objects (CellLocalAssembler, IfaceLocalAssembler)

template < class LAD, int DIM, class CellLocalAssembler, class IfaceLocalAssembler > 
class MatrixFreeOperator : public virtual LinearOperator<typename LAD::DataType> 
{
public:
  using DataType = typename LAD::DataType ;
  using GlobalVector = typename LAD::VectorType ;
  using VectorSpacePtr = const VectorSpace<DataType, DIM>*;

  using CellAssembler = CellVectorAssembly< InteriorAssemblyAlgorithm, DataType, DIM >;
  using IfaceAssembler = InterfaceVectorAssembly<DataType,DIM>;

  using QuadratureSelectionFunction = std::function< void (const Element< DataType, DIM > &,
                                                           Quadrature< DataType > &) >;

  using IFQuadratureSelectionFunction = std::function < void ( const Element< DataType, DIM > &, 
                                                               const Element< DataType, DIM > &, 
                                                               int, int,
                                                               Quadrature< DataType > &, 
                                                               Quadrature< DataType > &) >;
    
  MatrixFreeOperator()
  {}

  virtual ~MatrixFreeOperator() 
  {
    cell_assembler_.reset();
  }

  virtual void Init (const VectorSpace<DataType, DIM>& space,
                     CellLocalAssembler* local_asm_cell,
                     IfaceLocalAssembler* local_asm_iface,
                     const std::vector<int>& fixed_dofs,
                     const std::vector<mesh::EntityNumber>* cell_traversal = nullptr,
                     const mesh::InterfaceList* if_list = nullptr,
                     const QuadratureSelectionFunction q_select = DefaultQuadratureSelection< DataType, DIM >(),
                     const IFQuadratureSelectionFunction if_q_select = DefaultInterfaceQuadratureSelection< DataType, DIM >());

  virtual void VectorMult(Vector< DataType > &in,
                          Vector< DataType > *out) const override
  {
    GlobalVector *casted_in = dynamic_cast< GlobalVector * >(&in);
    GlobalVector *casted_out = dynamic_cast< GlobalVector * >(out);
    
    assert(casted_in != nullptr);
    assert(casted_out != nullptr);

    this->VectorMult(*casted_in, casted_out);
  }

  virtual void VectorMultAdd(DataType alpha, Vector< DataType > &in,
                             DataType beta, Vector< DataType > *out) const override 
  {
    GlobalVector *casted_in = dynamic_cast< GlobalVector * >(&in);
    GlobalVector *casted_out = dynamic_cast< GlobalVector * >(out);
    
    assert(casted_in != nullptr);
    assert(casted_out != nullptr);

    this->VectorMultAdd(alpha, *casted_in, beta, casted_out);
  }

  inline void VectorMult(GlobalVector &in,
                         GlobalVector *out) const 
  {
    this->VectorMultAdd(1., in, 0., out);
  }
                          
  /// out = beta * out + alpha * this * in
  void VectorMultAdd(DataType alpha, GlobalVector &in,
                     DataType beta,  GlobalVector *out) const;

  virtual bool IsInitialized() const 
  { 
    return this->is_initialized_; 
  }

private:
  bool use_insertion_for_lagrange_without_hanging_nodes_ = true;

  bool is_initialized_ = false;
  bool assemble_cell_ = false;
  bool assemble_iface_ = false;
  IFQuadratureSelectionFunction if_q_select_;  
  QuadratureSelectionFunction q_select_;
  VectorSpacePtr space_;

  std::vector<mesh::EntityNumber> cell_traversal_;
  mesh::InterfaceList if_list_;

  CellLocalAssembler* local_cell_assembler_ = nullptr;
  IfaceLocalAssembler* local_iface_assembler_ = nullptr;

  std::shared_ptr<CellAssembler> cell_assembler_ = 0;
  IfaceAssembler iface_assembler_;

  std::vector<int> fixed_dofs_;
  mutable std::vector<DataType> _fixed_values;
};

template < class LAD, int DIM, class CellLocalAssembler, class IfaceLocalAssembler >
void MatrixFreeOperator <LAD, DIM, CellLocalAssembler, IfaceLocalAssembler>::Init (const VectorSpace<DataType, DIM>& space,
                                                                                   CellLocalAssembler* local_asm_cell,
                                                                                   IfaceLocalAssembler* local_asm_iface,
                                                                                   const std::vector<int>& fixed_dofs,
                                                                                   const std::vector<mesh::EntityNumber>* cell_traversal,
                                                                                   const mesh::InterfaceList* if_list,
                                                                                   const QuadratureSelectionFunction q_select,
                                                                                   const IFQuadratureSelectionFunction if_q_select)
{
  assert (local_asm_cell != nullptr || local_asm_iface != nullptr);
  this->fixed_dofs_ = fixed_dofs;
  this->space_ = &space;

  if (local_asm_cell != nullptr)
  {
    assemble_cell_ = true;
    this->local_cell_assembler_ = local_asm_cell;
  }
  else 
  {
    assemble_cell_ = false;
    this->local_cell_assembler_ = nullptr;
  }

  if (local_asm_iface != nullptr)
  {
    assemble_iface_ = true;
    this->local_iface_assembler_ = local_asm_iface;
  }
  else 
  {
    assemble_iface_ = false;
    this->local_iface_assembler_ = nullptr;
  }

  if (assemble_cell_)
  {
    this->q_select_ = q_select;
    if (cell_traversal != nullptr)
    {
      //this->cell_traversal_ = *cell_traversal;
      this->cell_assembler_.reset(new CellAssembler(space, *cell_traversal));
    }
    else 
    {
      this->cell_assembler_.reset(new CellAssembler(space));
    }
  }

  if (assemble_iface_)
  {
    if (if_list != nullptr)
    {
      this->if_list_ = *if_list;
    }
    else 
    {
      this->if_list_ = mesh::InterfaceList::create(space.meshPtr());
    }
    this->if_q_select_ = if_q_select;
  }

  this->is_initialized_ = true;
}

template < class LAD, int DIM, class CellLocalAssembler, class IfaceLocalAssembler >
void MatrixFreeOperator <LAD, DIM, CellLocalAssembler, IfaceLocalAssembler>::VectorMultAdd(DataType alpha, 
                                                                                           GlobalVector &in,
                                                                                           DataType beta,  
                                                                                           GlobalVector *out) const
{
  out->Scale(beta);

  // cellwise contributions
  if (this->assemble_cell_)
  {
    assert (this->cell_assembler_ != 0);
    assert (this->local_cell_assembler_ != nullptr);
    this->local_cell_assembler_->set_vector(in, alpha);
    
    this->cell_assembler_->set_you_know_what_you_are_doing_flags(use_insertion_for_lagrange_without_hanging_nodes_);
    this->cell_assembler_->set_vector(*out);
    this->cell_assembler_->reset_traversal();
    cell_assembler_->assemble(*this->local_cell_assembler_, this->q_select_);
    //auto in_norm = in.Norm2();
    //auto out_norm = out->Norm2();
    //std::cout << alpha << " " << beta << " " << in_norm << " " << out_norm << std::endl;
  }
  // interface contributions
  if (this->assemble_iface_)
  {
    assert (this->local_iface_assembler_ != nullptr);
    this->local_iface_assembler_->set_vector(in, alpha);
    this->iface_assembler_.assemble (*this->space_, 
                                     this->if_list_, 
                                     this->if_q_select_, 
                                     *this->local_iface_assembler_, 
                                     *out);
  }

  // take care of fixed dofs 
  const auto nb_fixed_dofs = this->fixed_dofs_.size();
  if (nb_fixed_dofs > 0)
  {
    this->_fixed_values.resize(nb_fixed_dofs,0.);
    in.GetValues(vec2ptr(this->fixed_dofs_), nb_fixed_dofs, vec2ptr(this->_fixed_values));
    out->SetValues(vec2ptr(this->fixed_dofs_), nb_fixed_dofs, vec2ptr(this->_fixed_values));
  }

  out->Update();
}

}
}

#endif