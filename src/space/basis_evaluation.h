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

#ifndef HIFLOW_SPACE_BASIS_EVALUATION
#define HIFLOW_SPACE_BASIS_EVALUATION

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
#include <boost/function.hpp>

#define nUSE_STL_MAP
#define nALLOW_BASIS_PICKING

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

template < class DataType, int DIM > 
class BasisEvalLocal
{
  using Coord = typename StaticLA<DIM, DIM, DataType>::ColVectorType;
  using vec = typename StaticLA<DIM, DIM, DataType>::ColVectorType;

  typedef hiflow::doffem::gDofId BasisId;
  typedef int BasisIt;
  typedef size_t CellDofIt;
  typedef int CellIndex;
  typedef size_t CellIt;
  typedef size_t PtIt;
  
public:
  /// constructor for evaluating all basis functions of specified element 
  /// return of routine evaluate: vals[iv2ind(i,v)] = v-th comoponent of the specified fe                                
  BasisEvalLocal(const VectorSpace< DataType, DIM > &space, 
                 size_t fe_ind);

  BasisEvalLocal(const VectorSpace< DataType, DIM > &space, 
                 size_t fe_ind,
                 CellIndex cell_index);

  BasisEvalLocal(const VectorSpace< DataType, DIM > &space, 
                 size_t fe_ind,
                 const std::vector<CellIndex>& cell_indices);

  BasisEvalLocal(const VectorSpace< DataType, DIM > &space, 
                 size_t fe_ind,
                 const std::set<CellIndex>& cell_indices);

#ifdef ALLOW_BASIS_PICKING                                                             
  BasisEvalLocal(const VectorSpace< DataType, DIM > &space,
                 size_t fe_ind, bool dummy,
                 const std::vector<BasisId> & global_ids);
#endif
  // default constructor-> need to call init()                                 
  BasisEvalLocal()
  {}

  virtual ~BasisEvalLocal() {}

  void clear_init();
  void clear_setup();

  void init (const VectorSpace< DataType, DIM > &space, 
             size_t fe_ind,
             const std::set<CellIndex>& cell_indices,
             bool full_init = true);

  void evaluate (const Coord& pt, std::vector<DataType>& vals) const;
  void evaluate (const std::vector<Coord>& pts, 
                 std::vector< std::vector<DataType> >& vals) const;

  // here, entity is a dummy argument, for making BasisEvalLocal compatible with MappingPhys2Ref
  void evaluate (const mesh::Entity& entity, const Coord& pt, std::vector<DataType>& vals) const
  {
    return this->evaluate(pt, vals);
  }
  
  void get_basis_ids (std::vector<BasisId>& basis_ids) const
  {
    basis_ids = this->basis_ids_; 
  }
  
  inline size_t nb_comp() const
  {
    return this->nb_comp_;
  }

  inline size_t iv2ind(size_t i, size_t v) const
  {
    assert (i < this->nb_func_);
    assert (v < this->nb_comp_);
    //return i * this->nb_comp_ + v;
    return v * this->nb_func_ + i;
  }

  inline size_t weight_size() const
  {
    return this->weight_size_;
  }
  
  inline size_t nb_func() const
  {
    return this->nb_func_;
  }
    
  void set_print (bool flag)
  {
    this->print_ = flag;
  }
  
  inline bool valid_basis(BasisId i) const 
  {
    //return this->inv_basis_ids_.find(i) != this->inv_basis_ids_.end();
#ifdef USE_STL_MAP 
    return (this->basis_ids_sorted_.find(i) != this->basis_ids_sorted_.end());
#else
    int pos = -1;
    return this->basis_ids_sorted_.find(i, &pos);
#endif
  }

private:

  inline BasisIt basis_id_2_it(const BasisId& i) const 
  {
    //return this->inv_basis_ids_[i];
#ifdef USE_STL_MAP 
    const auto it = this->basis_ids_sorted_.find(i);
    if (it != this->basis_ids_sorted_.end())
    {
      return it->second;
    }
    return -1;
#else 
    BasisIt j = -1;
    this->basis_ids_sorted_.find(i, &j);
    return j;
#endif
  }

  void sort_basis_ids()
  {
    sort_and_erase_duplicates<int>(this->basis_ids_);

    this->basis_ids_sorted_.clear();
    this->basis_ids_sorted_.reserve(this->basis_ids_.size());
    for (BasisIt j=0, e_j = this->basis_ids_.size(); j!=e_j; ++j )
    {
  #ifdef USE_STL_MAP
      this->basis_ids_sorted_.insert({this->basis_ids_[j], j});
  #else
      this->basis_ids_sorted_.find_insert(this->basis_ids_[j]);
  #endif
    }
  }

  void clear_return_values(std::vector<DataType>& vals) const;
  void clear_return_values(std::vector<vec >& vals) const;

  void setup_for_evaluation();
  
  void search_pts(const std::vector<Coord>& pts) const; 
  
  const VectorSpace< DataType, DIM > * space_;
   
  size_t nb_func_;
  size_t nb_comp_;
  size_t fe_ind_;
  size_t weight_size_;
  mutable bool print_;
  
  mutable doffem::CRefElementSPtr< DataType, DIM > ref_fe_;
  
  // p := point iterator
  // j := basis iterator
  // i := basis id <-> DofPartion global dof id
  // k := cell iterator
  // c := cell index  <-> mesh cell index

  // ...[j] = i =: i[j]
  std::vector<BasisId> basis_ids_;  

#ifdef USE_STL_MAP
  std::unordered_map<BasisId, BasisIt> basis_ids_sorted_; 
#else
  SortedArray<BasisId> basis_ids_sorted_;
#endif

  // ...[i] = j, where i = i[j]
  //mutable std::unordered_map<BasisId, BasisIt> inv_basis_ids_;
    
  // ... [k] = c =: c[k]
  std::vector<CellIndex> active_cells_;
  
  // ... [c] = \{ i: K_{c} \in \supp(phi_{i}) \}
  std::vector< std::vector<BasisId> > inv_basis_support_;

  // ... [c] = \{ j: K_{c} \in \supp(phi_{i[j]}) \}
  std::vector< std::vector<BasisIt> > inv_basis_support_it_;

  // ...[c][i] = dof_factor, with which basis function i, restricted to K_c, has to be multiplied   
  std::vector< std::vector< DataType > > dof_factors_;
  
  // ...[c][i] = l such that \phi_{K_c,l) = \phi_i restricted to K_c_
  std::vector< std::vector< CellDofIt > > global_2_cell_;
  
  // ...[p][i] = #\{ K \in \supp(phi_{i}) : x \in K \}
  //mutable std::vector< std::map< BasisId, int > > pt_multiplicity_;
  
  // ...[p] = #\{ K : x \in K \}
  mutable std::vector< int > pt_multiplicity_;
  
  // ... [p][k] = (pt \in K_{c[k]} ?)
  mutable std::vector< std::vector< bool > > is_pt_in_cell_;
  
  // ... [p] = \{k : p \in K_{c[k]} \}
  mutable std::vector< std::vector< CellIt > > pt_in_cell_;
  
  // ... [p] = \{ref_pt : p \in K_{c[k]}, p <> ref_pt w.r.t. K_{c[k]}\}
  mutable std::vector< std::vector< Coord > > ref_pts_;

  mutable std::vector<DataType> cur_weights_;
  
  mutable std::vector<DataType> cell_coord_;
  mutable std::vector<double> double_coord_;

  mutable std::vector<size_t> mut_vars_;
  //mutable std::unordered_set<CellIndex> mut_cells_;

  mutable std::vector<CellIndex> mut_cur_cells_;
  mutable std::vector< BasisId > mut_all_gl_dof_ids_on_cell_;
  mutable std::vector< DataType> mut_cur_dof_factors_;
  mutable std::vector< BasisId > mut_tmp_ids_;

};

template < class DataType, int DIM > 
class FeEvalBasisLocal
{
  using Coord = typename StaticLA<DIM, DIM, DataType>::ColVectorType;
  using vec = typename StaticLA<DIM, DIM, DataType>::ColVectorType;

  typedef hiflow::doffem::gDofId gDofId;
  typedef hiflow::doffem::lDofId lDofId;
  typedef hiflow::doffem::cDofId cDofId;
  typedef size_t BasisIt;
  typedef size_t CellDofIt;
  typedef int CellIndex;
  typedef size_t CellIt;
  typedef size_t PtIt;
  
public:
  /// constructor for evaluating one specific fe 
  /// this type is typically used for FE interpolation
  /// return of routine evaluate: vals[iv2ind(0,v)] = v-th comoponent of specified element
  FeEvalBasisLocal(const VectorSpace< DataType, DIM > &space, 
                   const hiflow::la::Vector<DataType> &coeff, 
                   size_t fe_ind);
                                 
  virtual ~FeEvalBasisLocal();

  //void set_trial_cells(const std::vector< int > &trial_cells) const;
  
  bool evaluate              (const Coord& pt, DataType& value) const;
  bool evaluate              (const Coord& pt, std::vector<DataType>& vals) const;
  std::vector<bool> evaluate (const std::vector<Coord>& pt, 
                              std::vector< std::vector<DataType> >& vals) const;
  
  // here, entity is a dummy argument, for making FeEvalBasisLocal compatible with MappingPhys2Ref
  bool evaluate              (const mesh::Entity& entity, const Coord& pt, std::vector<DataType>& vals) const
  {
    return this->evaluate(pt, vals);
  }
  
  inline size_t nb_comp() const
  {
    return this->nb_comp_;
  }
  
  inline size_t iv2ind(size_t i, size_t v) const
  {
    assert (v < this->nb_comp_);
    assert (i == 0);
    return v;
  }
  
  inline size_t weight_size() const
  {
    return this->weight_size_;
  }
  
  inline size_t nb_func() const
  {
    return this->nb_func_;
  }
      
protected:
  virtual std::vector<bool> evaluate_impl (const std::vector<Coord>& pt, 
                                           std::vector< std::vector<DataType> >& vals) const;
                                      
  BasisEvalLocal<DataType, DIM>* basis_eval_;
  
  const VectorSpace< DataType, DIM > &space_;

  const la::Vector<DataType>& coeff_;
  
  size_t nb_comp_;
  size_t nb_func_;
  size_t weight_size_;
  size_t fe_ind_;
  mutable bool print_;
  mutable std::vector< int > trial_cells_;
};


} // namespace hiflow
#endif
