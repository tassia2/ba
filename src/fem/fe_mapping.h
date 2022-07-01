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

#ifndef __FEM_FE_MAPPING_H_
#define __FEM_FE_MAPPING_H_

#include <cassert>
#include <iostream>
#include <string>
#include <vector>
#include <functional>
#include <cmath>
#include <cstring>
//#include <boost/bind/bind.hpp>
#include "common/macros.h"
#include "common/vector_algebra_descriptor.h"
#include "common/log.h"
#include "fem/function_space.h"
#include "fem/fe_transformation.h"
#include "fem/cell_trafo/cell_transformation.h"

/// \author Philipp Gerstner

namespace hiflow {
namespace mesh {
  class Entity;
}

namespace doffem {

template <class DataType, int DIM> class RefElement;
template <class DataType, int DIM> class CellTransformation;
template <class DataType, int DIM> class AnsatzSpace;

//using namespace boost::placeholders;

///
/// \class RefCellFunctionInverseFETrafo
/// \brief Derived class for mapping a function, whose type is given by a template argument, from a physical cell to the reference cell
/// The template argument type "Functor" may contain multiple functions and should provied the following routines: </br>
///
/// size_t nb_func()      : returns number of functions contained in functor
/// size_t nb_comp()      : returns number of components of functions (all functons are assumed to have same number of components)
/// size_t iv2ind(i, c)   : indexing of weights. i <-> index of function, c <-> index of component 
/// size_t weight_size()  : size of weights vector (return argument of evaluate)
/// void evaluate(const Entity& cell, 
///               const Vec<DIM, DataType>& pt, 
///               std::vector<DataType>& weights): returns function values on given entity at given physical coordinates


template < class DataType, int DIM, class Functor > 
class MappingPhys2Ref : public virtual RefCellFunction<DataType, DIM> {

public:
  using Coord = typename StaticLA<DIM, DIM, DataType>::ColVectorType;
  typedef std::function< size_t (size_t, size_t) > IndexFunction;
  
  MappingPhys2Ref ( Functor const * func,
                    mesh::Entity const * entity, 
                    const CFETrafoSPtr<DataType, DIM>& fe_trafo,
                    const CCellTrafoSPtr<DataType, DIM>& cell_trafo)
  : RefCellFunction<DataType,DIM> (func->nb_comp(), func->nb_func()),
  func_(func),
  entity_(entity),
  fe_trafo_(fe_trafo),
  cell_trafo_(cell_trafo)
  {
  }
  
  ~MappingPhys2Ref()
  {
    fe_trafo_.reset();
    cell_trafo_.reset();
  }
  
  inline size_t iv2ind (size_t i, size_t var ) const 
  {
    //return this->fe_trafo_->iv2ind(i, var);
    return this->func_->iv2ind(i, var);
  }
  
  size_t weight_size() const 
  {
    return this->func_->weight_size();
  }
  
  void evaluate (const Coord &ref_pt, std::vector<DataType>& values) const
  {
    assert (this->func_ != nullptr);
    assert (this->entity_ != nullptr);
    assert (this->fe_trafo_ != nullptr);
    assert (this->cell_trafo_ != nullptr);
    
    //IndexFunction ind_fun = boost::bind ( &Functor::iv2ind, func_, _1, _2);
    auto ind_fun = [this] (size_t _i, size_t _var) { return this->func_->iv2ind(_i, _var); };

    // evaluate functor on physical entity
    this->phys_vals_.clear();
    this->phys_vals_.resize (this->func_->weight_size(), 0.);
    
    Coord pt;
    cell_trafo_->transform(ref_pt, pt);
    this->func_->evaluate(*this->entity_, pt, phys_vals_);
    
    // map physical value to value on reference cell
    values.clear();
    
    if (phys_vals_.size() > 0)
    {
      assert (phys_vals_.size() == this->func_->weight_size());
      values.resize(phys_vals_.size(), 0.);
    
      //std::cout << values.size() << " ?= " << phys_vals.size() << " &= " << this->nb_comp_ << " * " << this->nb_func_ << std::endl;
      assert (phys_vals_.size() == this->nb_comp_ * this->nb_func_);
      
      this->fe_trafo_->inverse_map_shape_function_values (*this->cell_trafo_, ref_pt, 0, this->nb_func_, this->nb_comp_, 
                                                          ind_fun, phys_vals_, values);
    }
    else
    {
      // If Functor is a Dirichlet BC evaluator, it may return empty values if pt lies on a non Dirichlet boundary facet
      // In this case, the NaN values are sorted out in the function compute_dirichlet_dofs_and_values/()
      values.resize(this->nb_comp_ * this->nb_func_, static_cast<DataType> (nan("empty")));
    }

/*
std::cout << "[MappingPhys2Ref] " << std::endl;
std::cout << " ref pt " << ref_pt[0] << " " << ref_pt[1] << std::endl;
std::cout << " phys pt " << pt[0] << " " << pt[1] << std::endl;
std::cout << " phys vals " ;
for (size_t l=0; l<phys_vals.size(); ++l)
std::cout << " " << phys_vals[l];

std::cout << std::endl;

std::cout << " mapped vals ";
for (size_t l=0; l<values.size(); ++l)
std::cout << " " << values[l];

std::cout << std::endl;
*/
  
  }
  
private:
  Functor const * func_;
  mesh::Entity const * entity_;
  CFETrafoSPtr<DataType, DIM> fe_trafo_;
  CCellTrafoSPtr<DataType, DIM> cell_trafo_;
  
  mutable std::vector<DataType> phys_vals_;
};

/// class MapperRefFunctionEval evaluates all mapped basis function of a given FunctionSpace (which lives on the reference cell) 
/// at a given set of points.
/// The mapping of the functions is performed by a given FETranformation object and determined 
/// by the given CellTransformation
/// This class satisfies the requirements of the "Functor" template in RefCellFunctionInverseFETrafo

template < class DataType, int DIM, class Functor  > 
class MappingRef2Phys {

public:
  using Coord = typename StaticLA<DIM, DIM, DataType>::ColVectorType;
  typedef std::function< size_t (size_t, size_t) > IndexFunction;
  
  MappingRef2Phys( Functor const * func,
                   const CFETrafoSPtr<DataType, DIM> & fe_trafo,
                   const CCellTrafoSPtr<DataType, DIM>& cell_trafo,
                   int cell_trafo_index = -1)
  : 
  func_(func),
  fe_trafo_(fe_trafo),
  cell_trafo_(cell_trafo),
  cell_trafo_index_(cell_trafo_index)
  {
  }
  
  ~MappingRef2Phys()
  {
    fe_trafo_.reset();
    cell_trafo_.reset();
  }

  inline size_t nb_comp () const
  {
    assert (this->func_ != nullptr);
    return this->func_->nb_comp();
  }

  inline size_t nb_func () const
  {
    assert (this->func_ != nullptr);
    return this->func_->dim();
  }
  
  inline size_t weight_size() const 
  {
    return this->nb_comp() * this->nb_func();
  }
   
  inline size_t iv2ind(size_t j, size_t v) const 
  {
    return this->func_->iv2ind(j,v);
  }
  
  /// evaluate all basis functions of ref_fe at all points pts
  /// here, entity is a dummy argument used for compatibility
  void evaluate (const mesh::Entity& entity, const Coord& pt, std::vector<DataType>& vals) const
  {
    assert (this->func_ != nullptr);
    assert (this->cell_trafo_ != nullptr);
    assert (this->fe_trafo_ != nullptr);
    const size_t nb_comp = this->nb_comp();
    const size_t nb_func = this->nb_func();
    
    //IndexFunction ind_fun = boost::bind ( &Functor::iv2ind, func_, _1, _2);
    auto ind_fun = [this] (size_t _i, size_t _var) { return this->func_->iv2ind(_i, _var); };

    if (vals.size() != this->weight_size())
    {
      vals.clear();
      vals.resize(this->weight_size(), 0.);
    }
    else
    {
      for (size_t i=0; i<vals.size(); ++i)
      {
        vals[i] = 0.;
      }
    }
      
    Coord ref_pt;
    if (this->cell_trafo_->inverse(pt, ref_pt))
    {
      // if physical point lies not on cell associated to cell_trafo ->
      // nothing to do since ansatz functions are extended by 0
      
      // evaluate reference shape function at ref_pt
      this->shape_vals_pt_.clear();
      this->shape_vals_pt_.resize (this->func_->weight_size(), 0.);
      this->func_->evaluate(ref_pt, shape_vals_pt_);

      // map shape function values to element living on physical cell
      this->fe_trafo_->map_shape_function_values (*this->cell_trafo_, ref_pt,
                                                  0, nb_func, nb_comp, ind_fun,
                                                  shape_vals_pt_, vals);
    }
  }

  
private:
  Functor const * func_;
  CFETrafoSPtr<DataType, DIM> fe_trafo_;
  CCellTrafoSPtr<DataType, DIM> cell_trafo_;
  int cell_trafo_index_;
  
  mutable std::vector<DataType> shape_vals_pt_;
};

} // namespace doffem
} // namespace hiflow
#endif
