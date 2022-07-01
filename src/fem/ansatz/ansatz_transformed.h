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

#ifndef __FEM_ANSATZ_SPACE_TRANSFORMED_H_
#define __FEM_ANSATZ_SPACE_TRANSFORMED_H_

#include <cassert>
#include <iostream>
#include <string>
#include <vector>
#include <functional>
#include <cmath>

#include "common/vector_algebra_descriptor.h"
#include "common/macros.h"
#include "common/log.h"
#include "fem/ansatz/ansatz_space.h"


namespace hiflow {
namespace doffem {

///
/// \class AnsatzSpaceTransformed ansatz_transformed.h
/// \brief Ansatz space that is a coordinate-transformed version of a given Ansatz space
/// \author Jonas Roller


template < class DataType, int DIM > 
class AnsatzSpaceTransformed final : public virtual AnsatzSpace<DataType, DIM> 
{

public:
  using Coord = typename StaticLA<DIM, DIM, DataType>::ColVectorType;
  using mat = typename StaticLA<DIM, DIM, DataType>::MatrixType;

  typedef std::function< void (const Coord &, std::vector<DataType> &) > BasisEvalFunction;

  /// Default Constructor
  AnsatzSpaceTransformed(CRefCellSPtr<DataType, DIM> ref_cell);

  /// Default Destructor
  virtual ~AnsatzSpaceTransformed();

  void init (AnsatzSpace<DataType, DIM> const * space, 
             AnsatzSpaceType type, 
             const mat* mat, 
             const Coord* vec);  

  /// For given point, get values of all shapefunctions on reference cell
  /// The following routines implement basis functions evaluations for tensor product spaces in case nb_comp > 1.
  /// In this case, derived classes only need to provide functin evaluations for each specific component.
  /// Use the routine iv2ind for accessing function evaluations for basis function i and component var 

  void N(const Coord &pt, std::vector< DataType > &weight) const override;
  void N_x(const Coord &pt, std::vector< DataType > &weight) const override;
  void N_y(const Coord &pt, std::vector< DataType > &weight) const override;
  void N_z(const Coord &pt, std::vector< DataType > &weight) const override;

  void N_xx(const Coord &pt, std::vector< DataType > &weight) const override;
  void N_xy(const Coord &pt, std::vector< DataType > &weight) const override;
  void N_xz(const Coord &pt, std::vector< DataType > &weight) const override;
  void N_yy(const Coord &pt, std::vector< DataType > &weight) const override;
  void N_yz(const Coord &pt, std::vector< DataType > &weight) const override;
  void N_zz(const Coord &pt, std::vector< DataType > &weight) const override;

private:
  void compute_degree_hash () const 
  {
    NOT_YET_IMPLEMENTED;
  }

  void evaluate(BasisEvalFunction fun, const Coord &pt, std::vector< DataType > &weight) const;
  
  Coord affine_transformation(const Coord &pt) const;

  AnsatzSpace<DataType, DIM> const * space_;

  const mat* mat_; // Matrix for affine transformation
  const Coord* vec_; // Vector for affine transformation
};

template < class DataType, int DIM >
AnsatzSpaceTransformed< DataType, DIM >::AnsatzSpaceTransformed(CRefCellSPtr<DataType, DIM> ref_cell)
    : AnsatzSpace<DataType, DIM>(ref_cell),
    space_(nullptr),
    mat_(nullptr),
    vec_(nullptr) 
{
  this->name_ = "Ansatz_Transformed";
  this->type_ = AnsatzSpaceType::TRANSFORMED;
}

template < class DataType, int DIM > 
AnsatzSpaceTransformed< DataType, DIM >::~AnsatzSpaceTransformed() 
{
}

template < class DataType, int DIM >
void AnsatzSpaceTransformed< DataType, DIM >::init (AnsatzSpace<DataType, DIM> const * space, AnsatzSpaceType type,
                                        const mat* mat, const Coord* vec)
{
  assert (type != AnsatzSpaceType::NOT_SET);
  assert (space != nullptr);
  assert (space->dim() > 0);

  this->space_ = space;
  this->dim_ = space->dim();

  this->weight_size_ = space->weight_size();
  this->nb_comp_ = space->nb_comp();
  this->tdim_ = space->tdim();

  this->max_deg_ = space->max_deg();
  this->name_ = space->name() + "_Transformed";
  this->ref_cell_ = space->ref_cell();
  this->type_ = type;

  this->mat_ = mat;
  this->vec_ = vec;
  //this->compute_degree_hash();
}

template < class DataType, int DIM >
typename AnsatzSpaceTransformed< DataType, DIM >::Coord 
AnsatzSpaceTransformed< DataType, DIM >::affine_transformation(const Coord &pt) const 
{
  Coord temp; 
  this->mat_->VectorMult(pt, temp);
  return temp + (*this->vec_); // affine transformation: Ax + b  
}

// TODO: avoid copy of weights by keeping N(pt, weight) from AnsatzSpace and overriding N(pt, comp, offset, weight) instead
template < class DataType, int DIM >
void AnsatzSpaceTransformed< DataType, DIM >::evaluate(BasisEvalFunction fun,
                                               const Coord &pt, 
                                               std::vector< DataType > &weight) const 
{ 
  assert (this->space_->weight_size() == weight.size());
  fun(this->affine_transformation(pt), weight);
}

template < class DataType, int DIM >
void AnsatzSpaceTransformed< DataType, DIM >::N(const Coord &pt, std::vector< DataType > &weight) const 
{
  auto fun = [this](const Coord & _pt, std::vector< DataType > & _weight)
                { return this->space_->N(_pt, _weight); };

  this->evaluate(fun, pt, weight);
}

template < class DataType, int DIM >
void AnsatzSpaceTransformed< DataType, DIM >::N_x(const Coord &pt, std::vector< DataType > &weight) const 
{
  auto fun = [this](const Coord & _pt, std::vector< DataType > & _weight)
                { return this->space_->N_x(_pt, _weight); };

  this->evaluate(fun, pt, weight);
}

template < class DataType, int DIM >
void AnsatzSpaceTransformed< DataType, DIM >::N_y(const Coord &pt, std::vector< DataType > &weight) const 
{
  auto fun = [this](const Coord & _pt, std::vector< DataType > & _weight)
                { return this->space_->N_y(_pt, _weight); };

  this->evaluate(fun, pt, weight);
}

template < class DataType, int DIM >
void AnsatzSpaceTransformed< DataType, DIM >::N_z(const Coord &pt, std::vector< DataType > &weight) const 
{
  auto fun = [this](const Coord & _pt, std::vector< DataType > & _weight)
                { return this->space_->N_z(_pt, _weight); };

  this->evaluate(fun, pt, weight);
}

template < class DataType, int DIM >
void AnsatzSpaceTransformed< DataType, DIM >::N_xx(const Coord &pt, std::vector< DataType > &weight) const 
{
  auto fun = [this](const Coord & _pt, std::vector< DataType > & _weight)
                { return this->space_->N_xx(_pt, _weight); };

  this->evaluate(fun, pt, weight);
}

template < class DataType, int DIM >
void AnsatzSpaceTransformed< DataType, DIM >::N_xy(const Coord &pt, std::vector< DataType > &weight) const 
{
   auto fun = [this](const Coord & _pt, std::vector< DataType > & _weight)
                { return this->space_->N_xy(_pt, _weight); };

  this->evaluate(fun, pt, weight);
}

template < class DataType, int DIM >
void AnsatzSpaceTransformed< DataType, DIM >::N_xz(const Coord &pt, std::vector< DataType > &weight) const 
{
   auto fun = [this](const Coord & _pt, std::vector< DataType > & _weight)
                { return this->space_->N_xz(_pt, _weight); };

  this->evaluate(fun, pt, weight);
}

template < class DataType, int DIM >
void AnsatzSpaceTransformed< DataType, DIM >::N_yy(const Coord &pt, std::vector< DataType > &weight) const 
{
  auto fun = [this](const Coord & _pt, std::vector< DataType > & _weight)
                { return this->space_->N_yy(_pt, _weight); };

  this->evaluate(fun, pt, weight);
}

template < class DataType, int DIM >
void AnsatzSpaceTransformed< DataType, DIM >::N_yz(const Coord &pt, std::vector< DataType > &weight) const 
{
  auto fun = [this](const Coord & _pt, std::vector< DataType > & _weight)
                { return this->space_->N_yz(_pt, _weight); };

  this->evaluate(fun, pt, weight);
}

template < class DataType, int DIM >
void AnsatzSpaceTransformed< DataType, DIM >::N_zz(const Coord &pt, std::vector< DataType > &weight) const 
{
  auto fun = [this](const Coord & _pt, std::vector< DataType > & _weight)
                { return this->space_->N_zz(_pt, _weight); };

  this->evaluate(fun, pt, weight);
}

} // namespace doffem
} // namespace hiflow
#endif
