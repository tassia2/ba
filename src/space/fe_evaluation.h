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

#ifndef HIFLOW_SPACE_FE_EVALUATION
#define HIFLOW_SPACE_FE_EVALUATION

/// \author Staffan Ronnas, Martin Baumann, Teresa Beck, Philipp Gerstner

#include <map>
#include <set>
#include <string>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include "common/vector_algebra_descriptor.h"
#include "common/parcom.h"
#include "dof/dof_fem_types.h"
#include "mesh/entity.h"
#include "space/element.h"
#include <boost/function.hpp>

namespace hiflow {

template <class DataType, int DIM> class VectorSpace;

namespace mesh {
class Entity;
template <class DataType, int DIM> class GeometricSearch;

}

namespace la {
template <class DataType> class Vector;
}

namespace doffem {
template <class DataType, int DIM> class RefElement;
}

template< class DataType, int DIM > 
class FlowTrafo 
{
public:
  using Coord = typename StaticLA<DIM, DIM, DataType>::ColVectorType;
  using vec = typename StaticLA<DIM, DIM, DataType>::ColVectorType;
  using mat = typename StaticLA<DIM, DIM, DataType>::MatrixType;

  FlowTrafo () {}
  
  virtual void operator() (const Coord& coords, 
                           const std::vector<DataType>& flow,
                           std::vector<DataType>& mapped_flow) const = 0;
  
  virtual std::vector<size_t> get_trafo_vars() const = 0;
};

template< class DataType, int DIM > 
class FlowTrafoCyl : public FlowTrafo<DataType, DIM>
{
public:
  using Coord = typename StaticLA<DIM, DIM, DataType>::ColVectorType;
  using vec = typename StaticLA<DIM, DIM, DataType>::ColVectorType;
  using mat = typename StaticLA<DIM, DIM, DataType>::MatrixType;

  FlowTrafoCyl (size_t first_flow_var) 
  {
    trafo_vars_.resize(DIM, 0);
    for (size_t l=0; l<DIM; ++l)
    {
      trafo_vars_[l] = first_flow_var + l;
    }
  }
  
  std::vector<size_t> get_trafo_vars() const
  {
    return this->trafo_vars_;
  }
  
  virtual void operator() (const Coord& coords, 
                           const std::vector<DataType>& flow,
                           std::vector<DataType>& mapped_flow) const
  {
    assert (DIM == 2 || DIM == 3);
    assert (flow.size() == DIM);
    assert (mapped_flow.size() == DIM);
    
    const DataType phi = coords[0];
    mapped_flow[0] = cos(phi) * flow[1] - sin(phi) * flow[0];
    mapped_flow[1] = sin(phi) * flow[1] + cos(phi) * flow[0];
    if constexpr (DIM == 3)
    {
      mapped_flow[DIM-1] = flow[DIM-1];
    }
  }
  std::vector<size_t> trafo_vars_;
};

template < class DataType, int DIM > 
class FeEvalCell
{
  typedef hiflow::doffem::gDofId gDofId;
  typedef hiflow::doffem::lDofId lDofId;
  typedef hiflow::doffem::cDofId cDofId;
  using Coord = typename StaticLA<DIM, DIM, DataType>::ColVectorType;
  using vec = typename StaticLA<DIM, DIM, DataType>::ColVectorType;
  using mat = typename StaticLA<DIM, DIM, DataType>::MatrixType;

  
public:
  /// constructor for evaluating all variables in Fe space for multiple coefficient vectors
  /// return of routine evaluate: vals[iv2ind(0,v)] = v-th comoponent of complete Fe space = v-th variable
  FeEvalCell(const VectorSpace< DataType, DIM > &space, 
             std::vector< hiflow::la::Vector<DataType> const *> coeffs);


  /// constructor for evaluating all variables in Fe space
  /// return of routine evaluate: vals[iv2ind(0,v)] = v-th comoponent of complete Fe space = v-th variable
  FeEvalCell(const VectorSpace< DataType, DIM > &space, 
             const hiflow::la::Vector<DataType> &coeff);
                                 
  /// constructor for evaluating one specific fe 
  /// this type is typically used for FE interpolation
  /// return of routine evaluate: vals[iv2ind(0,v)] = v-th comoponent of specified element
  FeEvalCell(const VectorSpace< DataType, DIM > &space, 
             const hiflow::la::Vector<DataType> &coeff, 
             size_t fe_ind);
                 
  FeEvalCell(const VectorSpace< DataType, DIM > &space, 
             const hiflow::la::Vector<DataType> &coeff, 
             const std::vector<size_t>& fe_ind);
             
  /// constructor for evaluating only specific variables
  /// this type is typically used in cell_visualization
  /// return of routine evaluate: vals[iv2ind(0,v)] = v-th comoponent of complete Fe space,
  /// supposed that v is contained in vars
  FeEvalCell(const VectorSpace< DataType, DIM > &space, 
             const hiflow::la::Vector<DataType> &coeff, 
             const std::vector<size_t>& vars, 
             FlowTrafo<DataType, DIM> const * map);
                 
  virtual ~FeEvalCell() {}

  // TODO: add routines for evaluating at a series of points within the same cell
  void evaluate   (const mesh::Entity& cell, const Coord& pt, std::vector<DataType>& vals) const;
  void r_evaluate (const mesh::Entity& cell, const Coord& ref_pt, std::vector<DataType>& vals) const;
  void r_evaluate (const mesh::Entity& cell, const Coord& ref_pt, std::vector< std::vector<DataType>* > vals) const;
  
  void evaluate_grad   (const mesh::Entity& cell, const Coord& pt, std::vector<vec >& vals) const;
  void r_evaluate_grad (const mesh::Entity& cell, const Coord& ref_pt, std::vector<vec >& vals) const;
  void r_evaluate_grad (const mesh::Entity& cell, const Coord& ref_pt, std::vector<std::vector<vec >* > vals) const;
  
  void r_evaluate_weight_and_grad (const mesh::Entity& cell, 
                                   const Coord& ref_pt, 
                                   std::vector<DataType>& vals,
                                   std::vector<vec >& grads) const;
                                   
  void r_evaluate_weight_and_grad (const mesh::Entity& cell, 
                                   const Coord& ref_pt, 
                                   std::vector< std::vector<DataType>* > vals,
                                   std::vector< std::vector< vec >* > grads) const;
                                                            
  inline size_t nb_comp() const
  {
    return this->nb_comp_;
  }

  inline size_t iv2ind(size_t i, size_t v) const
  {
    assert (i==0);
    assert (v < this->nb_comp_);
    return v;
  }

  inline size_t ivar2ind(size_t i, size_t v) const
  {
    assert (i==0);
    assert (v < this->var_order_.size());
    assert (this->var_order_[v] >= 0);
    return this->var_order_[v];
  }
  
  inline size_t weight_size() const
  {
    return this->nb_comp_;
  }
  
  inline size_t nb_func() const
  {
    return 1;
  }

  void set_print (bool flag)
  {
    this->print_ = flag;
  }
  
protected:
  inline size_t fe_comp_2_var(size_t fe_ind, size_t comp) const
  {
    return this->space_.fe_comp_2_var(fe_ind, comp);
  }
 
  void clear_return_values(std::vector<DataType>& vals) const;
  void clear_return_values(std::vector<vec >& vals) const;

  void update_dof_values(const mesh::Entity& cell) const; 
  
  void setup();
  
  const VectorSpace< DataType, DIM > &space_;
  
  std::vector< la::Vector<DataType> const * > coeffs_;
  
  int gdim_;

  size_t nb_comp_;
  
  std::vector<size_t> vars_;
  std::vector<size_t> fe_inds_;
  std::vector< std::vector< size_t> > fe_ind_2_comp_;
  std::vector< int > var_order_;
  mutable bool print_;
  
  FlowTrafo<DataType, DIM> const * flow_trafo_;
  const bool use_flow_trafo_ = false;

  // sorted data sets, i.e. 'val' is sorted corresponding to 'id'
  //mutable std::vector< gDofId > id_;
  mutable std::vector<std::vector< std::vector<DataType> > > _dof_values; 
  mutable int _last_cell_index = -1;
  mutable std::vector<DataType> _weights;
  mutable std::vector<vec > _weights_grad;
  mutable Element<DataType, DIM> _element;
};


template < class DataType, int DIM > 
class FeEvalLocal
{
  typedef hiflow::doffem::gDofId gDofId;
  typedef hiflow::doffem::lDofId lDofId;
  typedef hiflow::doffem::cDofId cDofId;
  using Coord = typename StaticLA<DIM, DIM, DataType>::ColVectorType;
  using vec = typename StaticLA<DIM, DIM, DataType>::ColVectorType;
  using mat = typename StaticLA<DIM, DIM, DataType>::MatrixType;

  
public:
  /// constructor for evaluating all variables in Fe space
  /// return of routine evaluate: vals[iv2ind(0,v)] = v-th comoponent of complete Fe space = v-th variable
  FeEvalLocal(const VectorSpace< DataType, DIM > &space, 
              const hiflow::la::Vector<DataType> &coeff);
                                 
  /// constructor for evaluating one specific fe 
  /// this type is typically used for FE interpolation
  /// return of routine evaluate: vals[iv2ind(0,v)] = v-th comoponent of specified element
  FeEvalLocal(const VectorSpace< DataType, DIM > &space, 
              const hiflow::la::Vector<DataType> &coeff, 
              size_t fe_ind);
                 
  FeEvalLocal(const VectorSpace< DataType, DIM > &space, 
              const hiflow::la::Vector<DataType> &coeff, 
              const std::vector<size_t>& fe_ind);
             
  /// constructor for evaluating only specific variables
  /// this type is typically used in cell_visualization
  /// return of routine evaluate: vals[iv2ind(0,v)] = v-th comoponent of complete Fe space,
  /// supposed that v is contained in vars
  FeEvalLocal(const VectorSpace< DataType, DIM > &space, 
              const hiflow::la::Vector<DataType> &coeff, 
              const std::vector<size_t>& vars, 
              FlowTrafo<DataType, DIM> const * map);
                 
  virtual ~FeEvalLocal();

  void set_trial_cells(const std::vector< int > &trial_cells) const;
  void set_trial_cells(const std::set< int > &trial_cells) const;
    
  bool evaluate (const Coord& pt, DataType& value) const;
  bool evaluate (const Coord& pt, std::vector<DataType>& vals) const;
  
  std::vector<bool> evaluate (const std::vector<Coord>& pt, 
                              std::vector< std::vector<DataType> >& vals) const;

  // here, entity is a dummy argument, for making FeEvalLocal compatible with MappingPhys2Ref
  bool evaluate (const mesh::Entity& entity, const Coord& pt, std::vector<DataType>& vals) const
  {
    return this->evaluate(pt, vals);
  }
  
  // TODO: add routines for evaluating gradient
  bool evaluate_grad              (const Coord& pt, std::vector<vec > & vals) const;
  std::vector<bool> evaluate_grad (const std::vector<Coord>& pt, 
                                   std::vector< std::vector<vec> >& vals) const;
       
  inline size_t nb_comp() const
  {
    assert (this->fe_eval_cell_ != nullptr);
    return this->fe_eval_cell_->nb_comp();
  }
  
  inline size_t iv2ind(size_t i, size_t v) const
  {
    assert (this->fe_eval_cell_ != nullptr);
    return this->fe_eval_cell_->iv2ind(i,v);
  }
  
  inline size_t ivar2ind(size_t i, size_t v) const
  {
    assert (this->fe_eval_cell_ != nullptr);
    return this->fe_eval_cell_->ivar2ind(i,v);
  }
  
  inline size_t weight_size() const
  {
    assert (this->fe_eval_cell_ != nullptr);
    return this->fe_eval_cell_->weight_size();
  }
  
  inline size_t nb_func() const
  {
    assert (this->fe_eval_cell_ != nullptr);
    return this->fe_eval_cell_->nb_func();
  }
      
protected:
  virtual std::vector<bool> evaluate_impl (const std::vector<Coord>& pt, 
                                           std::vector< std::vector<DataType> >& vals) const;
                                      
  void setup();
                        
  void search_points (const std::vector<Coord>& pts, 
                      std::vector< std::vector<int> >& cell_indices, 
                      std::vector< std::vector<Coord> >& ref_pts) const;
                      
  bool check_ref_coords (const Coord& pt, 
                         const std::vector<int> & cell_indices, 
                         const std::vector<Coord> & ref_pts) const;
                                               
  mesh::GeometricSearch<DataType, DIM>* search_;
  
  FeEvalCell<DataType, DIM>* fe_eval_cell_;
  
  const VectorSpace< DataType, DIM > &space_;

  mutable bool print_;
  
  mutable std::vector< int > const * vec_trial_cells_;
  
  mutable std::set< int > const * set_trial_cells_;
    
};

template < class DataType, int DIM > 
class FeEvalGlobal : public FeEvalLocal<DataType, DIM>
{
  typedef hiflow::doffem::gDofId gDofId;
  typedef hiflow::doffem::lDofId lDofId;
  typedef hiflow::doffem::cDofId cDofId;
  using Coord = typename StaticLA<DIM, DIM, DataType>::ColVectorType;
  using vec = typename StaticLA<DIM, DIM, DataType>::ColVectorType;
  using mat = typename StaticLA<DIM, DIM, DataType>::MatrixType;

  
public:
  /// constructor for evaluating all variables in Fe space
  /// return of routine evaluate: vals[iv2ind(0,v)] = v-th comoponent of complete Fe space = v-th variable
  FeEvalGlobal(const VectorSpace< DataType, DIM > &space, 
               const hiflow::la::Vector<DataType> &coeff);
                                 
  /// constructor for evaluating one specific fe 
  /// this type is typically used for FE interpolation
  /// return of routine evaluate: vals[iv2ind(0,v)] = v-th comoponent of specified element
  FeEvalGlobal(const VectorSpace< DataType, DIM > &space, 
               const hiflow::la::Vector<DataType> &coeff, 
               size_t fe_ind);
                 
  FeEvalGlobal(const VectorSpace< DataType, DIM > &space, 
               const hiflow::la::Vector<DataType> &coeff, 
               const std::vector<size_t>& fe_ind);
             
  /// constructor for evaluating only specific variables
  /// this type is typically used in cell_visualization
  /// return of routine evaluate: vals[iv2ind(0,v)] = v-th comoponent of complete Fe space,
  /// supposed that v is contained in vars
  FeEvalGlobal(const VectorSpace< DataType, DIM > &space, 
               const hiflow::la::Vector<DataType> &coeff, 
               const std::vector<size_t>& vars, 
               FlowTrafo<DataType, DIM> const * map);
                 
  virtual ~FeEvalGlobal();
      
protected:
  std::vector<bool> evaluate_impl (const std::vector<Coord>& pt, 
                                   std::vector< std::vector<DataType> >& vals) const;
                                           
  ParCom* parcom_;
};


} // namespace hiflow
#endif
