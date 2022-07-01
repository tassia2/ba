// Copyright (C) 2011-2020 Vincent Heuveline
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

#ifndef HIFLOW_SPACE_FE_INTERPOLATION_MAP
#define HIFLOW_SPACE_FE_INTERPOLATION_MAP

/// \author Philipp Gerstner

#include <map>
#include <vector>

#include "common/array_tools.h"
#include "common/parcom.h"
#include "common/vector_algebra_descriptor.h"
#include "common/sorted_array.h"
#include "common/timer.h"
#include "linear_algebra/vector.h"
#include "linear_algebra/lmp/init_vec_mat.h"
#include "linear_algebra/lmp/lmatrix.h"
#include "linear_algebra/lmp/lmatrix_csr_cpu.h"
#include "fem/fe_mapping.h"
#include "mesh/geometric_tools.h"
#include "mesh/entity.h"
#include "mesh/types.h"
#include "mesh/mesh_db_view.h"
#include "space/vector_space.h"
#include "space/fe_evaluation.h"
#include "space/fe_interpolation_cell.h"
#include "space/fe_evaluation.h"
#include "space/basis_evaluation.h"

#define NO_PATTERN_OPTIM
#define NO_RELATED_MESH_OPTIM
#define nNO_EXACT_MATCH_OPTIM
#define nNO_FE_OPTIM
#define nDBG_PATTERN

namespace hiflow {

template < class DataType, int DIM > 
class InterpolationPattern
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
  InterpolationPattern()
  {}

  ~InterpolationPattern()
  {}

  void init (VectorSpace< DataType, DIM> const * in_space,
             VectorSpace< DataType, DIM> const * out_space,
             int out_cell_index,
             const std::set<int>& in_cell_indices, 
             const std::vector<size_t>& in_fe_inds, 
             const std::vector<size_t>& out_fe_inds);

  bool same_cell_pattern (const InterpolationPattern<DataType, DIM>& rhs) const;

  bool same_fe_pattern (const InterpolationPattern<DataType, DIM>& rhs) const;

  bool operator== (const InterpolationPattern<DataType, DIM>& rhs) const
  {
    return (this->same_cell_pattern(rhs) && this->same_fe_pattern(rhs));
  }

  int in_ind_2_unique (int l) const
  {
    assert (l >= 0);
    assert (l < this->in_ind_2_unique_.size());
    return this->in_ind_2_unique_[l];
  }

  int out_ind_2_unique (int l) const
  {
    assert (l >= 0);
    assert (l < this->out_ind_2_unique_.size());
    return this->out_ind_2_unique_[l];
  }

  std::vector<int> in_unique_2_ind (int l) const 
  {
    assert (l >= 0);
    assert (l < this->in_unique_2_ind_.size());
    return this->in_unique_2_ind_[l];
  }

  std::vector<int> out_unique_2_ind (int l) const 
  {
    assert (l >= 0);
    assert (l < this->out_unique_2_ind_.size());
    return this->out_unique_2_ind_[l];
  }

  int in_num_unique () const 
  {
    return this->in_ind_2_unique_.size();
  }

  int out_num_unique () const 
  {
    return this->out_ind_2_unique_.size();
  }

  bool is_cell_match() const;

private:
  void create_fe_sets ( int cell_index,
                        VectorSpace< DataType, DIM> const * space,
                        const std::vector<size_t>& fe_inds, 
                        std::vector< int >& ind_2_unique,
                        std::vector< std::vector< int > >& unique_2_ind,
                        std::vector< doffem::RefElement< DataType, DIM > const * >& unique_fe) const;

  bool same_cell_pattern_impl (VectorSpace< DataType, DIM> const * in_space,
                               VectorSpace< DataType, DIM> const * out_space,
                               int out_index_A,
                               int out_index_B,
                               const std::set<int>& in_indices_A,
                               const std::set<int>& in_indices_B) const;

  VectorSpace< DataType, DIM> const * in_space_;
  VectorSpace< DataType, DIM> const * out_space_;

  int out_cell_index_;
  std::set<int> in_cell_indices_;

  std::vector< int > out_ind_2_unique_, in_ind_2_unique_; 
  std::vector< std::vector< int > >  out_unique_2_ind_, in_unique_2_ind_;
  std::vector< doffem::RefElement< DataType, DIM > const * > out_unique_fe_, in_unique_fe_;
  std::vector<size_t> in_fe_inds_, out_fe_inds_;
};

template < class LAD, int DIM > 
class FeInterMapBase
{
  typedef typename LAD::VectorType Vector;
  typedef typename LAD::DataType DataType;
  using Coord = typename StaticLA<DIM, DIM, DataType>::ColVectorType;
  using vec = typename StaticLA<DIM, DIM, DataType>::ColVectorType;

  typedef hiflow::doffem::gDofId gDofId;
  typedef hiflow::doffem::lDofId lDofId;
  typedef hiflow::doffem::cDofId cDofId;
  typedef size_t BasisIt;
  typedef size_t CellDofIt;
  typedef mesh::EntityNumber CellIndex;
  typedef size_t CellIt;
  typedef size_t PtIt;
  
public:

  FeInterMapBase();

  ~FeInterMapBase()
  {
    this->clear(); 
  }
 
  inline bool is_initialized() const
  {
    return this->initialized_;
  }
  
protected:
  void clear();

  template <class CellInterpolator>
  void init_with_linear_map_without_comm (CellInterpolator * cell_inter,
                                          VectorSpace< DataType, DIM> const * in_space,
                                          VectorSpace< DataType, DIM> const * out_space,
                                          bool related_meshes,
                                          const std::vector<size_t>& in_fe_inds, 
                                          const std::vector<size_t>& out_fe_inds);
                     
  void init_without_linear_map_without_comm (VectorSpace< DataType, DIM> const * in_space,
                                             VectorSpace< DataType, DIM> const * out_space,
                                             const std::vector<size_t>& in_fe_inds, 
                                             const std::vector<size_t>& out_fe_inds);
  
  void build_interpolation_matrix();
    
  void interpolate_with_linear_map_without_comm (const Vector& in_vec, 
                                                 Vector& out_vec) const;
                                                     
  template <class CellInterpolator>
  void interpolate_without_linear_map_without_comm (CellInterpolator * cell_inter, 
                                                    const Vector& in_vec, 
                                                    Vector& out_vec) const;
  
  void interpolate_with_matrix_without_comm (const Vector& in_vec, 
                                             Vector& out_vec) const;

  void interpolate_trans_with_matrix_without_comm (const Vector& in_vec, 
                                                   Vector& out_vec) const;

  VectorSpace< DataType, DIM> const * in_space_;
  VectorSpace< DataType, DIM> const * out_space_;
  
  la::lMatrix< DataType > * diag_;
  la::lMatrix< DataType > * diagT_;
  la::lMatrix< DataType > * odiag_;
  la::lMatrix< DataType > * odiagT_;

  std::vector< InterpolationPattern<DataType, DIM> > inter_patterns_;
  std::vector< int > out_2_pattern_;

  //std::vector<gDofId> in_ids_;
  //std::vector<gDofId> out_ids_;
  
  /// \brief cell_map[K] = {K' \in in_mesh : K \cap K' != \emptyset}
  std::map<CellIndex , std::set<CellIndex> > cell_map_;
  
  /// \brief weights[K] = { A^e_{K,K'} \in R^{m, n} : K' \in cell_map[K]}
  /// with A^e_{K,K'} [i,j] = [cell_inter.eval(phi_{K', j, e})]_i 
  /// m : #dofs on cell K
  /// n : #dofs on cell K'
  
  //std::vector< std::map< int, std::vector< std::vector< std::vector< DataType > > > > > weights_;
  std::vector< std::vector< std::vector< std::vector< DataType > > > > weights_;
  
  /// \brief in_dof_ids[K][e] = global dof ids of basis functions of FE index e of in_space 
  /// whose support has nonempty intersection with out_cell K 
  //std::vector< std::vector< std::vector< doffem::gDofId > > > in_dof_ids_;
  std::vector< std::vector< std::vector< gDofId > > > in_dof_ids_;
  
  //std::map < doffem::gDofId, DataType > in_dof_weights_;
  
  /// \brief out_dof_ids[K][e] = global dof ids on cell K w.r.t. to FE e in out_space
  std::vector< std::vector< std::vector< gDofId > > > out_dof_ids_;
  
  std::vector<size_t> in_fe_inds_;
  std::vector<size_t> out_fe_inds_;

  mutable std::vector< DataType > in_dof_factors_;
  mutable std::vector< DataType > out_dof_factors_;
  mutable std::vector< DataType > in_vals_;
  mutable std::vector< DataType > out_dof_values_;
  
  bool initialized_;
  
  ParCom* parcom_;
};

/// \brief Interpolation map for nodal interpolation
/// For convenience regarding the ugly template argument "CellInterpolator" 
template < class LAD, int DIM> 
class FeInterMapFullNodal : public FeInterMapBase<LAD, DIM> 
{
  typedef typename LAD::DataType DataType;
  typedef typename LAD::VectorType Vector;
  typedef FeInterCellNodal<DataType, DIM, BasisEvalLocal< DataType, DIM > >  NodalCellInterpolator;
  
public:

  FeInterMapFullNodal()
  : FeInterMapBase<LAD, DIM >()
  {
  }

  ~FeInterMapFullNodal()
  {
  }

  void init (VectorSpace< DataType, DIM> const * in_space,
             VectorSpace< DataType, DIM> const * out_space,
             bool related_meshes)
  {
    assert (in_space != nullptr);
    assert (out_space != nullptr);
    
    std::vector<size_t> in_fe_inds;  
    std::vector<size_t> out_fe_inds;
    number_range<size_t>(0, 1, in_space->nb_fe(), in_fe_inds);
    number_range<size_t>(0, 1, out_space->nb_fe(), out_fe_inds);
    
    this->init(in_space, out_space, related_meshes, in_fe_inds, out_fe_inds);
  }
  
  void init (VectorSpace< DataType, DIM> const * in_space,
             VectorSpace< DataType, DIM> const * out_space,
             bool related_meshes,
             const std::vector<size_t>& in_fe_inds, 
             const std::vector<size_t>& out_fe_inds)
  {
    assert (in_space != nullptr);
    assert (out_space != nullptr);
    
    NodalCellInterpolator * cell_inter = new NodalCellInterpolator(*out_space);
      
    this->init_with_linear_map_without_comm (cell_inter,
                                             in_space, 
                                             out_space,
                                             related_meshes,
                                             in_fe_inds,
                                             out_fe_inds);
                                            
    delete cell_inter;
  }

  void interpolate (const Vector& in_vec, Vector& out_vec) const
  {
    //this->interpolate_with_linear_map_without_comm(in_vec, out_vec);
    this->interpolate_with_matrix_without_comm(in_vec, out_vec);
  }
  
  void interpolate_transpose (const Vector& in_vec, Vector& out_vec) const
  {
    this->interpolate_trans_with_matrix_without_comm(in_vec, out_vec);
  }
};

template < class LAD, int DIM> 
class FeInterMapRedNodal : public FeInterMapBase<LAD, DIM > 
{
  typedef typename LAD::DataType DataType;
  typedef typename LAD::VectorType Vector;
  typedef FeInterCellNodal<DataType, DIM, FeEvalLocal< DataType, DIM > >  NodalCellInterpolator;
  
public:

  FeInterMapRedNodal()
  : FeInterMapBase<LAD, DIM >()
  {
  }

  ~FeInterMapRedNodal()
  {
  }

  void init (VectorSpace< DataType, DIM> const * in_space,
             VectorSpace< DataType, DIM> const * out_space)
  {
    assert (in_space != nullptr);
    assert (out_space != nullptr);
    
    std::vector<size_t> in_fe_inds;  
    std::vector<size_t> out_fe_inds;
    number_range<size_t>(0, 1, in_space->nb_fe(), in_fe_inds);
    number_range<size_t>(0, 1, out_space->nb_fe(), out_fe_inds);
    
    this->init(in_space, out_space, in_fe_inds, out_fe_inds);
  }
  
  void init (VectorSpace< DataType, DIM> const * in_space,
             VectorSpace< DataType, DIM> const * out_space,
             const std::vector<size_t>& in_fe_inds, 
             const std::vector<size_t>& out_fe_inds)
  {
    assert (in_space != nullptr);
    assert (out_space != nullptr);
    
    this->init_without_linear_map_without_comm(in_space, 
                                               out_space, 
                                               in_fe_inds,
                                               out_fe_inds);
  }

  void interpolate (const Vector& in_vec, Vector& out_vec) const
  {
    NodalCellInterpolator * cell_inter = new NodalCellInterpolator(*this->out_space_);
    
    this->interpolate_without_linear_map_without_comm(cell_inter, in_vec, out_vec);
    delete cell_inter;
  }
  
};

////////////////////////////////////////////////////
///////////// FeInterMap ///////////////////////////
////////////////////////////////////////////////////

template < class LAD, int DIM >
FeInterMapBase<LAD, DIM>::FeInterMapBase()
: in_space_(nullptr), out_space_(nullptr), initialized_(false), parcom_(nullptr), 
  diag_(nullptr),
  diagT_(nullptr),
  odiag_(nullptr),
  odiagT_(nullptr)
{
}

template < class LAD, int DIM >
void FeInterMapBase<LAD, DIM>::clear()
{
  this->inter_patterns_.clear();
  this->out_2_pattern_.clear();
  this->weights_.clear();
  this->in_dof_ids_.clear();
  this->out_dof_ids_.clear();
  this->cell_map_.clear();
  this->in_fe_inds_.clear();
  this->out_fe_inds_.clear();
  this->in_space_ = nullptr;
  this->out_space_ = nullptr;
  this->initialized_ = false;
  if (this->parcom_ != nullptr)
  {
    delete this->parcom_;
    this->parcom_ = nullptr;
  }
  if (this->diag_ != nullptr)
  {
    delete this->diag_;
    this->diag_ = nullptr;
  }
  if (this->odiag_ != nullptr)
  {
    delete this->odiag_;
    this->odiag_ = nullptr;
  }
  if (this->diagT_ != nullptr)
  {
    delete this->diagT_;
    this->diagT_ = nullptr;
  }
  if (this->odiagT_ != nullptr)
  {
    delete this->odiagT_;
    this->odiagT_ = nullptr;
  }
}

template < class DataType >
bool same_weights (const std::vector< std::vector< std::vector< DataType > > >& wA,
                   const std::vector< std::vector< std::vector< DataType > > >& wB,
                   DataType tol)
{
  for (int l=0; l != wA.size(); ++l)
  {
    for (int i = 0; i!= wA[l].size(); ++i)
    {
      for (int j = 0; j!= wA[l][i].size(); ++j)
      {
        if (std::abs(wA[l][i][j] - wB[l][i][j]) > tol)
        {
          return false;
        }
      }
    }
  }
  return true;

/*
  std::cout << " out " << out_index << std::endl;
      for (int l=0; l != this->weights_[out_index].size(); ++l)
      {
        for (int i = 0; i!= this->weights_[out_index][l].size(); ++i)
        {
          for (int j = 0; j!= this->weights_[out_index][l][i].size(); ++j)
          {
            std::cout << " " << this->weights_[out_index][l][i][j];
          }
          std::cout << std::endl;
        }
      }
*/
}

template < class LAD, int DIM >
template < class CellInterpolator >
void FeInterMapBase<LAD, DIM>::init_with_linear_map_without_comm (CellInterpolator * cell_inter,
                                                                  VectorSpace< DataType, DIM> const * in_space,
                                                                  VectorSpace< DataType, DIM> const * out_space,
                                                                  bool related_meshes,
                                                                  const std::vector<size_t>& in_fe_inds, 
                                                                  const std::vector<size_t>& out_fe_inds)
{
  Timer timer;
  timer.start();

  assert (cell_inter != nullptr);
  assert (in_space != nullptr);
  assert (out_space != nullptr);
  assert (in_fe_inds.size() > 0);
  assert (out_fe_inds.size() > 0);
  assert (in_fe_inds.size() == out_fe_inds.size());
  
#ifndef NDEBUG
  for (size_t l=0; l<in_fe_inds.size(); ++l)
  {
    const size_t in_fe_ind = in_fe_inds[l];
    const size_t out_fe_ind = out_fe_inds[l];
    assert(in_fe_ind < in_space->nb_fe());
    assert(out_fe_ind < out_space->nb_fe());
  }
#endif

  this->clear();
  this->parcom_ = new ParCom(in_space->get_mpi_comm());
   
  this->in_space_ = in_space;
  this->out_space_ = out_space;
  
  this->in_fe_inds_ = in_fe_inds;
  this->out_fe_inds_ = out_fe_inds;

#ifdef NO_RELATED_MESH_OPTIM
  related_meshes = false;
#endif

  // create map to obtain for K of out_mesh all adjacent K' of in_mesh
  //LOG_INFO("init linear map", " create cell map ");
  if (related_meshes)
  {
    mesh::find_adjacent_cells_related<DataType, DIM> (out_space->meshPtr(), in_space->meshPtr(), this->cell_map_);
  }
  else
  {
    mesh::find_adjacent_cells<DataType, DIM> (out_space->meshPtr(), in_space->meshPtr(), this->cell_map_);
  }
  
  //this->parcom_->barrier();
  timer.stop();
  LOG_INFO("FE inter", "cell mapping took " << timer.get_duration() << " sec");
  timer.reset();
  timer.start();

  const int tdim = out_space->meshPtr()->tdim();
  const CellIndex out_num_cell = out_space->meshPtr()->num_entities(tdim);

  this->weights_.clear();
  this->weights_.resize(out_num_cell);
  this->out_dof_ids_.resize(out_num_cell);
  this->in_dof_ids_.resize(out_num_cell);

  // TODO: put vector objects into class member 
  std::vector< std::vector< std::vector< std::vector<DataType> > > >coeff;
  std::vector< gDofId > in_gl_ids;
  std::vector< gDofId > out_gl_ids;

  std::vector< BasisEvalLocal<DataType, DIM> > in_basis_eval (in_fe_inds.size());

  // loop over all cells in out_mesh
  //LOG_INFO("init linear map", " compute coefficients ");
  
  assert (in_space->meshPtr()->num_entities(DIM) == in_space->fe_manager().fe_tank_size());
  assert (out_space->meshPtr()->num_entities(DIM) == out_space->fe_manager().fe_tank_size());

  this->out_2_pattern_.clear();
  this->out_2_pattern_.resize(out_num_cell, -1);
  this->inter_patterns_.clear();
  this->inter_patterns_.reserve(out_num_cell);
  int computed_patterns = 0;
  int different_patterns = 0;
  int different_cell_patterns = 0;

  // compute maximal number of in-cell
  // -> used to identify boundary patterns in structured grids
  int max_in_cells = 0;
  for (CellIndex out_index = 0; out_index < out_num_cell; ++out_index)
  {
    if (this->cell_map_[out_index].size() > max_in_cells)
    {
      max_in_cells = this->cell_map_[out_index].size();
    }
  }

  for (CellIndex out_index = 0; out_index < out_num_cell; ++out_index)
  {
    mesh::Entity out_cell (this->out_space_->meshPtr(), tdim, out_index);

    const int num_patterns = this->inter_patterns_.size();
    int pattern_ind = -1;
    int similar_out_index = -1;
    bool found_pattern = false;
    bool full_basis_init = true;

    // create interpolation pattern
    InterpolationPattern<DataType, DIM> cur_pattern;
    cur_pattern.init(in_space,
                     out_space,
                     out_index,
                     this->cell_map_[out_index], 
                     in_fe_inds, 
                     out_fe_inds);

#ifndef NO_EXACT_MATCH_OPTIM
    bool cell_match = cur_pattern.is_cell_match();
#else 
    bool cell_match = false;
#endif

#ifndef NO_PATTERN_OPTIM
    // check if interpolation pattern already exists

    // always compute boundary / irregular patterns 
    //if (this->cell_map_[out_index].size() == max_in_cells)
    {
      for (int k = 0; k!=num_patterns; ++k)
      {
        if (cur_pattern == this->inter_patterns_[k])
        {
          found_pattern = true;
          pattern_ind = k;
          break;
        }
      }
    }
    if (!found_pattern)
    {
      this->inter_patterns_.push_back(cur_pattern);
      pattern_ind = this->inter_patterns_.size() -1;
      different_cell_patterns++;
    }
    this->out_2_pattern_[out_index] = pattern_ind;
    
    // determine out cell index with same pattern
    if (found_pattern)
    {
      for (int k=0; k!=out_num_cell; ++k)
      {
        if (this->out_2_pattern_[k] == pattern_ind)
        {
          similar_out_index = k;
          break;
        }
      }
    }
#endif

    LOG_DEBUG(2, " out cell " << out_index << " : found_pattern " << found_pattern << " with id " << pattern_ind);
    LOG_DEBUG(2, " out cell " << out_index << " : similar out cell " << similar_out_index );
    LOG_DEBUG(2, " out cell " << out_index << " : num cells " << this->cell_map_[out_index].size());

    const int out_num_unique_fe = cur_pattern.out_num_unique();
    const int in_num_unique_fe = cur_pattern.in_num_unique();
    bool applied_fe_reduction = (in_num_unique_fe < this->in_fe_inds_.size());

#ifdef DBG_PATTERN
    if (false)
#else
    if (found_pattern)
#endif
    {
      // if pattern found -> no need to recompute interpolation weights  
      assert (similar_out_index >= 0);
      assert (this->weights_[similar_out_index].size() > 0);
      this->weights_[out_index] = this->weights_[similar_out_index];
      full_basis_init = false;
    }
    else 
    {
      coeff.clear();
      coeff.resize(out_num_unique_fe);
      for (int l = 0; l!=out_num_unique_fe; ++l)
      {
        coeff[l].resize(in_num_unique_fe);
      }

      if (cell_match)
      {  
        // coeff is identity matrix
        full_basis_init = false;
      }
      else
      {
        // pattern has not been processed so far
      }
    }

    // cheap initialization of BasisObject -> sufficient for calling get_basis_ids()
    for (size_t l=0; l<this->in_fe_inds_.size(); ++l)
    {
      LOG_DEBUG(2, "   create basis eval object for " << this->cell_map_[out_index].size() << " cells ");
      if (applied_fe_reduction)
      {
        // multiple instances of same FE pairs are present
        // -> we can reuse the corresponding computed weights
        // -> we only need to fully initialize basis_eval once, see below
        in_basis_eval[l].init( *this->in_space_, in_fe_inds[l], this->cell_map_[out_index], false);
      }
      else 
      {
        // we need to compute weights for each single fe-pair
        // -> fully initialize, if necessary (in accordance with cell pattern)
        in_basis_eval[l].init( *this->in_space_, in_fe_inds[l], this->cell_map_[out_index], full_basis_init);
      }
    }

    // loop over all considered variables 
    for (size_t l=0; l<this->in_fe_inds_.size(); ++l)
    {
      const size_t in_fe_ind = this->in_fe_inds_[l];
      const size_t out_fe_ind = this->out_fe_inds_[l];

      const int in_cur_unique_fe_ind = cur_pattern.in_ind_2_unique(l);
      const int out_cur_unique_fe_ind = cur_pattern.out_ind_2_unique(l);

      const size_t nb_dofs_on_out_cell = this->out_space_->fe_manager().get_fe(out_index, out_fe_ind)->nb_dof_on_cell();
            
      // get global dof ids of in_space
      in_gl_ids.clear();
      in_basis_eval[in_fe_ind].get_basis_ids(in_gl_ids);
      assert (in_gl_ids.size() > 0);
      
      // get global dof ids of out cell
      out_gl_ids.clear();
      out_gl_ids.resize(nb_dofs_on_out_cell);
      this->out_space_->get_dof_indices(out_fe_ind, out_index, out_gl_ids);


#ifdef DBG_PATTERN
      if (true)
#else
      if (!found_pattern)
#endif
      {
#ifdef NO_FE_OPTIM
        coeff[out_cur_unique_fe_ind][in_cur_unique_fe_ind].clear();
#endif
        // evaluate in_basis at out_dofs
        if (coeff[out_cur_unique_fe_ind][in_cur_unique_fe_ind].size() == 0)
        {
          if (cell_match)
          {
            // out_cell matches in_cell -> identity matrix
            assert (in_gl_ids.size() == out_gl_ids.size());
            assert (this->cell_map_[out_index].size() == 1);
            assert (out_cur_unique_fe_ind == in_cur_unique_fe_ind);

            // get in global indices from cell (needed, because in_gl_ids from basis_eval might be shuffled)
            const CellIndex in_index = *(this->cell_map_[out_index].begin());
            const size_t nb_dofs_on_in_cell = this->in_space_->fe_manager().get_fe(in_index, in_fe_ind)->nb_dof_on_cell();

            in_gl_ids.clear();
            in_gl_ids.resize(nb_dofs_on_in_cell);
            this->in_space_->get_dof_indices(in_fe_ind, in_index, in_gl_ids);

            assert (nb_dofs_on_in_cell == nb_dofs_on_out_cell);
            coeff[out_cur_unique_fe_ind][in_cur_unique_fe_ind].resize(nb_dofs_on_in_cell);

            for (size_t k = 0; k != nb_dofs_on_in_cell; ++k)
            {
              set_to_unitvec_i(nb_dofs_on_in_cell, k, coeff[out_cur_unique_fe_ind][in_cur_unique_fe_ind][k]);
            }
          }
          else 
          {
            // this FE pair has not been processed so far 
            assert (cur_pattern.in_unique_2_ind(in_cur_unique_fe_ind).size() > 0);
            assert (cur_pattern.out_unique_2_ind(out_cur_unique_fe_ind).size() > 0);

            // need full initialization of fe basis object 
            const int repr_fe_ind = cur_pattern.in_unique_2_ind(in_cur_unique_fe_ind)[0];
            if (applied_fe_reduction)
            {
              // see comment above
              in_basis_eval[repr_fe_ind].init( *this->in_space_, in_fe_inds[repr_fe_ind], this->cell_map_[out_index], true);
            }
            else 
            {
              assert (full_basis_init);
            }

            // pass evaluable object to cell interpolator
            cell_inter->set_function(&in_basis_eval[repr_fe_ind]);

            // evaluate cell interpolator
            LOG_DEBUG(2, "   evaluate cell interpolator ");
            cell_inter->compute_fe_coeff (&out_cell, out_fe_inds[cur_pattern.out_unique_2_ind(out_cur_unique_fe_ind)[0]], coeff[out_cur_unique_fe_ind][in_cur_unique_fe_ind]); 
            computed_patterns++;
          }
        }
        LOG_DEBUG(2, "   data handling ");
        this->weights_[out_index].push_back(coeff[out_cur_unique_fe_ind][in_cur_unique_fe_ind]);
      }

      this->in_dof_ids_[out_index].push_back(in_gl_ids);
      this->out_dof_ids_[out_index].push_back(out_gl_ids);

      assert ( found_pattern || (coeff[out_cur_unique_fe_ind][in_cur_unique_fe_ind][0].size() == in_gl_ids.size()) );
      assert ( found_pattern || (coeff[out_cur_unique_fe_ind][in_cur_unique_fe_ind].size() == nb_dofs_on_out_cell));
      //std::cout << " out cell " << out_index << " : " << coeff.size() << " " << in_gl_ids.size() << std::endl;
      //log_2d_array(coeff, std::cout, 2);
    }

#ifdef DBG_PATTERN
    if (found_pattern)
    {
      assert (similar_out_index >= 0);
      if (!same_weights(weights_[out_index], weights_[similar_out_index], 1e-12))
      {
        LOG_DEBUG(0, "        out index " << out_index 
                  << " in indices " << string_from_range(this->cell_map_[out_index].begin(), this->cell_map_[out_index].end()));
        LOG_DEBUG(0, "similar out index " << similar_out_index 
                  << " in indices " << string_from_range(this->cell_map_[similar_out_index].begin(), this->cell_map_[similar_out_index].end()));
      }
    }
#endif
#if 0
    // determine number of different weight patterns
    bool found_match = false;
    for (int c=0; c < out_index; ++c)
    {  
      if (same_weights(weights_[out_index], weights_[c], 1e-12))
      {
        found_match = true;
        break;
      }
    }
    if (!found_match)
    {
      different_patterns++;
    }
#endif
  }

  //this->parcom_->barrier();
  timer.stop();
  LOG_INFO("FE inter", " found " << different_patterns << " different weights for " << out_num_cell << " cells ");
  LOG_INFO("FE inter", " found " << different_cell_patterns << " different interpolation patterns for " << out_num_cell << " cells ");
  LOG_INFO("FE inter", " computed " << computed_patterns << " different interpolation patterns for " << out_num_cell << " cells ");
  LOG_INFO("FE inter", " weight computation took " << timer.get_duration() << " sec");
  timer.reset();
  timer.start();

  this->build_interpolation_matrix();
  
  //this->parcom_->barrier();
  timer.stop();
  LOG_INFO("FE inter", " matrix computation took " << timer.get_duration() << " sec");

  this->initialized_ = true;
}

template < class LAD, int DIM >
void FeInterMapBase<LAD, DIM>::init_without_linear_map_without_comm (VectorSpace< DataType, DIM> const * in_space,
                                                                     VectorSpace< DataType, DIM> const * out_space,
                                                                     const std::vector<size_t>& in_fe_inds, 
                                                                     const std::vector<size_t>& out_fe_inds)
{
  assert (in_space != nullptr);
  assert (out_space != nullptr);
  assert (in_fe_inds.size() > 0);
  assert (out_fe_inds.size() > 0);
  assert (in_fe_inds.size() == out_fe_inds.size());
  
#ifndef NDEBUG
  for (size_t l=0; l<in_fe_inds.size(); ++l)
  {
    const size_t in_fe_ind = in_fe_inds[l];
    const size_t out_fe_ind = out_fe_inds[l];
    assert(in_fe_ind < in_space->nb_fe());
    assert(out_fe_ind < out_space->nb_fe());
  }
#endif

  this->clear();
  this->parcom_ = new ParCom(in_space->get_mpi_comm());
   
  this->in_space_ = in_space;
  this->out_space_ = out_space;
  
  this->in_fe_inds_ = in_fe_inds;
  this->out_fe_inds_ = out_fe_inds;

  // create map to obtain for K of out_mesh all adjacent K' of in_mesh
  mesh::find_adjacent_cells<DataType, DIM> (out_space->meshPtr(), in_space->meshPtr(), this->cell_map_);
     
  this->initialized_ = true;
}

template < class LAD, int DIM >
void FeInterMapBase<LAD, DIM>::build_interpolation_matrix()
{
  // interpolation weights below eps are considered to be zero
  const DataType eps = 1e-16;
  const int tdim = this->out_space_->meshPtr()->tdim();
  const CellIndex out_num_cell = this->out_space_->meshPtr()->num_entities(tdim);
  const size_t num_fe = this->in_fe_inds_.size();
  const auto my_rank = this->parcom_->rank();

  const auto nb_out_local = this->out_space_->la_couplings().nb_dofs(my_rank);
  const auto nb_out_ghost = this->out_space_->la_couplings().size_ghost();
  const auto nb_in_local = this->in_space_->la_couplings().nb_dofs(my_rank);
  const auto nb_in_ghost = this->in_space_->la_couplings().size_ghost();

  const lDofId diag_nrow = nb_out_local;    // out_vec.size_local();
  const lDofId diag_ncol = nb_in_local;     // in_vec.size_local();
  
  const lDofId odiag_nrow = nb_out_local;   // out_vec.size_local();
  const lDofId odiag_ncol = nb_in_ghost;    // in_vec.size_local_ghost();

  const lDofId diagT_nrow = nb_in_local;    //  in_vec.size_local();
  const lDofId diagT_ncol = nb_out_local;   //  out_vec.size_local();
  
  const lDofId odiagT_nrow = nb_in_local;   //in_vec.size_local();
  const lDofId odiagT_ncol = nb_out_ghost;  //out_vec.size_local_ghost();
    
  const lDofId mat_diag_nnz_est = diag_nrow * 20;
  const lDofId mat_odiag_nnz_est = odiag_nrow * 20;
  const lDofId matT_diag_nnz_est = diagT_nrow * 20;
  const lDofId matT_odiag_nnz_est = odiagT_nrow * 20;
  
  // TODO: make member variables for performance reasons 
  SortedArray<gDofId> visited_i;
  
  std::vector<lDofId> diag_coo_i;
  std::vector<lDofId> diag_coo_j;
  std::vector<DataType> diag_coo_v;

  std::vector<lDofId> diagT_coo_i;
  std::vector<lDofId> diagT_coo_j;
  std::vector<DataType> diagT_coo_v;

  std::vector<lDofId> odiag_coo_i;
  std::vector<lDofId> odiag_coo_j;
  std::vector<DataType> odiag_coo_v;

  std::vector<lDofId> odiagT_coo_i;
  std::vector<lDofId> odiagT_coo_j;
  std::vector<DataType> odiagT_coo_v;

  // diag and offdiagonal part of interpolation matrix
  diag_coo_i.reserve(mat_diag_nnz_est);
  diag_coo_j.reserve(mat_diag_nnz_est);
  diag_coo_v.reserve(mat_diag_nnz_est);

  odiag_coo_i.reserve(mat_odiag_nnz_est);
  odiag_coo_j.reserve(mat_odiag_nnz_est);
  odiag_coo_v.reserve(mat_odiag_nnz_est);

  // diag and offdiagonal part of transposed interpolation matrix
  diagT_coo_i.reserve(matT_diag_nnz_est);
  diagT_coo_j.reserve(matT_diag_nnz_est);
  diagT_coo_v.reserve(matT_diag_nnz_est);

  odiagT_coo_i.reserve(matT_odiag_nnz_est);
  odiagT_coo_j.reserve(matT_odiag_nnz_est);
  odiagT_coo_v.reserve(matT_odiag_nnz_est);
  
  //SortedArray<gDofId> out_ids;
  //SortedArray<gDofId> in_ids;
  
  //out_ids.reserve(2*num_row);
  //in_ids.reserve(2*num_col);
  
  std::vector<lDofId> tmp_loc_in_ids;
  std::vector<lDofId> tmp_gh_in_ids;
  std::vector<lDofId> tmp_loc_out_ids;
  std::vector<lDofId> tmp_gh_out_ids;
      
  std::vector< DataType > dof_factors;
  
  //this->in_ids_.clear();
  //this->in_ids_.resize(num_col);
  //this->out_ids_.clear();
  //this->out_ids_.resize(num_row);
  
  // compute interpolation matrices in COO form
  
  // loop over all out cell
  for (size_t out_index = 0; out_index < out_num_cell; ++out_index)
  {
    dof_factors.clear();
    this->out_space_->dof().get_dof_factors_on_cell(out_index, dof_factors);

    // loop over all fe's
    for (size_t l=0; l<num_fe; ++l)
    {
      const size_t out_fe_ind   = this->out_fe_inds_[l];
      const size_t num_out_dofs = this->out_dof_ids_[out_index][l].size();
      const size_t num_in_dofs  = this->in_dof_ids_[out_index][l].size();

      // for indexing dof_factors
      size_t out_start_dof_on_cell = 0;
      for (size_t k=0; k<out_fe_ind; ++k)
      { 
        out_start_dof_on_cell += this->out_space_->dof().nb_dofs_on_cell(k, out_index);
      }
      
      // input: get local and ghost dof ids for current cell and fe 
      tmp_loc_in_ids.clear();
      tmp_loc_in_ids.reserve(num_in_dofs);
      tmp_gh_in_ids.clear();
      tmp_gh_in_ids.reserve(num_in_dofs);
      
      this->in_space_->global_2_local_and_ghost(this->in_dof_ids_[out_index][l],
                                                tmp_loc_in_ids,
                                                tmp_gh_in_ids);
                        
      // output: get local and ghost dof ids for current cell and fe
      tmp_loc_out_ids.clear();
      tmp_loc_out_ids.reserve(num_out_dofs);
      tmp_gh_out_ids.clear();
      tmp_gh_out_ids.reserve(num_out_dofs);
      
      this->out_space_->global_2_local_and_ghost(this->out_dof_ids_[out_index][l],
                                                 tmp_loc_out_ids,
                                                 tmp_gh_out_ids);
      
      assert (num_out_dofs > 0);
      assert (num_in_dofs > 0);

      // loop over output dofs
      for (size_t i=0; i!= num_out_dofs; ++i)
      {
        // get global out dof id
        const gDofId gl_i = this->out_dof_ids_[out_index][l][i];
               
        // visit all out ids only once
        bool found_i = visited_i.find_insert(gl_i);
        if (found_i)
        {
          continue;
        }
        
        // get local and ghost out id
        const lDofId loc_i = tmp_loc_out_ids[i];
        const lDofId gh_i = tmp_gh_out_ids[i];
  
        assert (loc_i >= 0 || gh_i >= 0);
        assert (loc_i < 0 || gh_i < 0);
        
        const bool diag_i = (gh_i == -1);
          
        //this->out_ids_[loc_i] = gl_i;
        //out_ids.insert(loc_i);
        
        // compare VectorSpace::insert_dof_values()
        const DataType factor_i = 1. / dof_factors[out_start_dof_on_cell + i];
                
        // loop over in dofs
        for (size_t j=0; j!=num_in_dofs; ++j)
        {
          const lDofId loc_j = tmp_loc_in_ids[j];
          const lDofId gh_j = tmp_gh_in_ids[j];
          
          assert (loc_j >= 0 || gh_j >= 0);
          assert (loc_j < 0 || gh_j < 0);
          const bool diag_j = (gh_j == -1);
          
          DataType val = this->weights_[out_index][l][i][j] * factor_i;
          
          // skip entries close to zero
          if (std::abs(val) < eps)
          {
            continue;
          }
          
          // correct rounding errors if val is close to -1, 1
          if (std::abs(val-1.) < eps)
          {
            val = 1.;
          }
          else if (std::abs(val+1.) < eps)
          {
            val = -1.;
          }
          
          if (diag_i)
          {
            // out id is in diagonal part
            if (diag_j)
            {
              // in id is in diagonal part
              assert (loc_i < diag_nrow);
              assert (loc_j < diag_ncol);
              
              diag_coo_i.push_back(loc_i);
              diag_coo_j.push_back(loc_j);
              diag_coo_v.push_back(val);

              assert (loc_j < diagT_nrow);
              assert (loc_i < diagT_ncol);
              
              diagT_coo_i.push_back(loc_j);
              diagT_coo_j.push_back(loc_i);
              diagT_coo_v.push_back(val);
            }
            else
            {
              // in id is in offdiagonal part
              assert (loc_i < odiag_nrow);
              assert (gh_j < odiag_ncol);
              
              odiag_coo_i.push_back(loc_i);
              odiag_coo_j.push_back(gh_j);
              odiag_coo_v.push_back(val);
            }
          }
          else
          {
            // out id is in offdiagonal part
            if (diag_j)
            {
              // in id is in diagonal part
              assert (loc_j < odiagT_nrow);
              assert (gh_i < odiagT_ncol);
              
              odiagT_coo_i.push_back(loc_j);
              odiagT_coo_j.push_back(gh_i);
              odiagT_coo_v.push_back(val);
            }
          }
        }
      }
    }
  }
  
  const lDofId diag_nnz   = diag_coo_i.size();
  const lDofId odiag_nnz  = odiag_coo_i.size();
  const lDofId diagT_nnz  = diagT_coo_i.size();
  const lDofId odiagT_nnz = odiagT_coo_i.size();
  
  assert (diag_nnz > 0);
  assert (diagT_nnz > 0);
  
#ifdef nWITH_MKL
  this->diag_   = la::init_matrix<DataType>(diag_nnz,  diag_nrow,  diag_ncol,  "inter_diag",  la::CPU, la::MKL, la::CSR);
  this->diagT_  = la::init_matrix<DataType>(diagT_nnz, diagT_nrow, diagT_ncol, "inter_diagT", la::CPU, la::MKL, la::CSR);
  this->odiag_  = la::init_matrix<DataType>(odiag_nnz, odiag_nrow, odiag_ncol, "inter_odiag", la::CPU, la::MKL, la::CSR);
  this->odiagT_ = la::init_matrix<DataType>(odiagT_nnz,odiagT_nrow,odiagT_ncol,"inter_odiagT",la::CPU, la::MKL, la::CSR);
#else
  this->diag_   = la::init_matrix<DataType>(diag_nnz,  diag_nrow,  diag_ncol,  "inter_diag",  la::CPU, la::NAIVE, la::CSR);
  this->diagT_  = la::init_matrix<DataType>(diagT_nnz, diagT_nrow, diagT_ncol, "inter_diagT", la::CPU, la::NAIVE, la::CSR);
  this->odiag_  = la::init_matrix<DataType>(odiag_nnz, odiag_nrow, odiag_ncol, "inter_odiag", la::CPU, la::NAIVE, la::CSR);
  this->odiagT_ = la::init_matrix<DataType>(odiagT_nnz,odiagT_nrow,odiagT_ncol,"inter_odiagT",la::CPU, la::NAIVE, la::CSR);
#endif
  
  dynamic_cast<CPU_CSR_lMatrix <DataType> * >(this->diag_)  ->TransformFromCOO(&(diag_coo_i[0]),  &(diag_coo_j[0]),  &(diag_coo_v[0]),  diag_nrow,  diag_ncol,  diag_nnz); 
  dynamic_cast<CPU_CSR_lMatrix <DataType> * >(this->diagT_) ->TransformFromCOO(&(diagT_coo_i[0]), &(diagT_coo_j[0]), &(diagT_coo_v[0]), diagT_nrow, diagT_ncol, diagT_nnz); 
  dynamic_cast<CPU_CSR_lMatrix <DataType> * >(this->odiag_) ->TransformFromCOO(&(odiag_coo_i[0]), &(odiag_coo_j[0]), &(odiag_coo_v[0]), odiag_nrow, odiag_ncol, odiag_nnz); 
  dynamic_cast<CPU_CSR_lMatrix <DataType> * >(this->odiagT_)->TransformFromCOO(&(odiagT_coo_i[0]),&(odiagT_coo_j[0]),&(odiagT_coo_v[0]),odiagT_nrow,odiagT_ncol,odiagT_nnz); 
}

template < class LAD, int DIM >
void FeInterMapBase<LAD, DIM>::interpolate_with_matrix_without_comm (const Vector& in_vec, 
                                                                     Vector& out_vec) const
{
  assert (this->initialized_);  
  assert (this->diag_ != nullptr);
  assert (this->odiag_ != nullptr);
  assert (in_vec.interior().get_size() == this->diag_->get_num_col());
  assert (out_vec.interior().get_size() == this->diag_->get_num_row());
  assert (in_vec.ghost().get_size() == this->odiag_->get_num_col());
  assert (out_vec.interior().get_size() == this->odiag_->get_num_row());
  
  this->diag_->VectorMult(in_vec.interior(), &(out_vec.interior()));
  this->odiag_->VectorMultAdd(in_vec.ghost(), &(out_vec.interior()));
  out_vec.store_interior();
  out_vec.Update();
}

template < class LAD, int DIM >
void FeInterMapBase<LAD, DIM>::interpolate_trans_with_matrix_without_comm (const Vector& out_vec, 
                                                                           Vector& in_vec) const
{
  assert (this->initialized_);  
  assert (this->diagT_ != nullptr);
  assert (this->odiagT_ != nullptr);
  
  this->diagT_->VectorMult(out_vec.interior(), &(in_vec.interior()));

  this->odiagT_->VectorMultAdd(out_vec.ghost(), &(in_vec.interior()));

  in_vec.store_interior();
  in_vec.Update();
}

template < class LAD, int DIM >
void FeInterMapBase<LAD, DIM>::interpolate_with_linear_map_without_comm (const Vector& in_vec, 
                                                                         Vector& out_vec) const
{
  const int tdim = this->out_space_->meshPtr()->tdim();
  const size_t out_num_cell = this->out_space_->meshPtr()->num_entities(tdim);
  const size_t num_fe = this->in_fe_inds_.size();

  assert (this->initialized_);  
  assert (this->weights_.size() == out_num_cell);
  assert (this->out_dof_ids_.size() == out_num_cell);
  assert (this->in_dof_ids_.size() == out_num_cell);
  
  // loop over all cells in out_mesh
  for (size_t out_index = 0; out_index < out_num_cell; ++out_index)
  {
    assert (this->weights_[out_index].size() == num_fe);
    assert (this->in_dof_ids_[out_index].size() == num_fe);
    assert (this->out_dof_ids_[out_index].size() == num_fe);
          
    // loop over all considered variables 
    for (size_t l=0; l<num_fe; ++l)
    {
      const size_t out_fe_ind = this->out_fe_inds_[l];
      const size_t num_dofs_on_out_cell = this->out_dof_ids_[out_index][l].size();
      const size_t num_dofs_on_in_cell = this->in_dof_ids_[out_index][l].size();
        
      // get values from in_vector
      this->in_vals_.clear();
      this->in_vals_.resize(num_dofs_on_in_cell, 0.);
      in_vec.GetValues (&this->in_dof_ids_[out_index][l][0], num_dofs_on_in_cell, &in_vals_[0]);
      
      assert (in_vals_.size() == num_dofs_on_in_cell);
      assert (this->weights_[out_index][l].size() == num_dofs_on_out_cell);

      this->out_dof_values_.clear();
      this->out_dof_values_.resize(num_dofs_on_out_cell, 0.);

      // loop over all dofs on out_cell
      for (size_t i=0; i<num_dofs_on_out_cell; ++i)
      {
        const gDofId gl_i = this->out_dof_ids_[out_index][l][i];
        
        assert (num_dofs_on_in_cell == this->weights_[out_index][l][i].size());
        
        // interpolate only locally owned dofs
        if (this->parcom_->rank() != this->out_space_->dof().owner_of_dof(gl_i))
        {
          continue;
        }
        
        DataType val_i = 0.;
                
        // loop over all dofs on in_cell
        for (size_t j=0; j<num_dofs_on_in_cell; ++j)
        {
          val_i += in_vals_[j] * this->weights_[out_index][l][i][j];
        }
        
        out_dof_values_[i] = val_i;
        
        //std::cout << out_index << " " << l << " : " << gl_i << " " << val_i << " " << in_vec.GetValue(gl_i) << std::endl;
        //out_vec.SetValue(gl_i, val_i);
      }
      this->out_space_->insert_dof_values(out_fe_ind, out_index, out_vec, out_dof_values_);
    }
  }
  out_vec.Update();
}

template < class LAD, int DIM >
template < class CellInterpolator >
void FeInterMapBase<LAD, DIM>::interpolate_without_linear_map_without_comm (CellInterpolator * cell_inter,
                                                                            const Vector& in_vec, 
                                                                            Vector& out_vec) const
{
  const int tdim = this->out_space_->meshPtr()->tdim();
  const size_t out_num_cell = this->out_space_->meshPtr()->num_entities(tdim);
  const size_t num_fe = this->in_fe_inds_.size();
  std::vector< std::vector<DataType> > cell_coeff;
  std::vector<DataType> dof_values;
  
  assert (this->initialized_);  

  // create Fe evaluator objects
  std::vector< FeEvalLocal<DataType, DIM>* > in_fe_evals (num_fe);
  for (size_t l=0; l<num_fe; ++l)
  {
    in_fe_evals[l] = new FeEvalLocal<DataType, DIM>(*this->in_space_, in_vec, this->in_fe_inds_[l]);
  }
  
  // loop over all cells in out_mesh
  for (int out_index = 0; out_index < out_num_cell; ++out_index)
  {         
    mesh::Entity out_cell = this->out_space_->meshPtr()->get_entity(tdim, out_index);
                            
    // loop over all considered variables 
    for (size_t l=0; l<num_fe; ++l)
    {
      const size_t in_fe_ind = this->in_fe_inds_[l];
      const size_t out_fe_ind = this->out_fe_inds_[l];

      // set trial cells for accelerating point search
      in_fe_evals[l]->set_trial_cells(this->cell_map_.at(out_index));

      // evaluate interpolation on current cell
      cell_inter->set_function(in_fe_evals[l]);
      cell_inter->compute_fe_coeff (&out_cell, out_fe_ind, cell_coeff);

      assert (cell_coeff.size() == this->out_space_->nb_dof_on_cell(out_fe_ind, out_index) );
      assert (cell_coeff[0].size() == 1);
    
      // insert values into vector sol
      dof_values.resize(cell_coeff.size(), 0.); 
      for (size_t i=0; i<dof_values.size(); ++i)
      {
        dof_values[i] = cell_coeff[i][0];
      }
      this->out_space_->insert_dof_values(out_fe_ind, out_index, out_vec, dof_values);
    }
  }
  out_vec.Update();
  
  for (size_t l=0; l<num_fe; ++l)
  {
    delete in_fe_evals[l];
  }
}

template < class DataType, int DIM >
void InterpolationPattern<DataType, DIM>::init (VectorSpace< DataType, DIM> const * in_space,
                                                VectorSpace< DataType, DIM> const * out_space,
                                                int out_index,
                                                const std::set<int>& in_cell_indices, 
                                                const std::vector<size_t>& in_fe_inds, 
                                                const std::vector<size_t>& out_fe_inds)
{
  this->in_space_ = in_space;
  this->out_space_ = out_space;
  this->out_cell_index_ = out_index;
  this->in_cell_indices_ = in_cell_indices;

  this->out_ind_2_unique_.clear();
  this->in_ind_2_unique_.clear();
  this->out_unique_2_ind_.clear();
  this->in_unique_2_ind_.clear();
  this->out_unique_fe_.clear();
  this->in_unique_fe_.clear();
  this->in_fe_inds_ = in_fe_inds;
  this->out_fe_inds_ = out_fe_inds;
    

  // put equal fe types into groups
  this->create_fe_sets (out_index, out_space, out_fe_inds, 
                        this->out_ind_2_unique_, this->out_unique_2_ind_, this->out_unique_fe_);

  if (in_cell_indices.size() == 1)
  {
    this->create_fe_sets (*(in_cell_indices.begin()), in_space, in_fe_inds, 
                          this->in_ind_2_unique_, this->in_unique_2_ind_,  this->in_unique_fe_);
  }
  else
  {
    this->in_unique_2_ind_.resize(in_fe_inds.size());
    for (int l=0; l!=in_fe_inds.size(); ++l)
    {
      this->in_ind_2_unique_.push_back(l);
      this->in_unique_2_ind_[l].push_back(l);
    }
  }


  /*
  if (out_space->fe_manager().same_fe_on_all_cells())
  {
    if (out_index == 0)
    {
      this->create_fe_sets (out_index, out_space, out_fe_inds, 
                            this->out_ind_2_unique_, this->out_unique_2_ind_, this->out_unique_fe_);
    }
  }
  else
  {
    this->create_fe_sets (out_index, out_space, out_fe_inds, 
                          this->out_ind_2_unique_, this->out_unique_2_ind_, this->out_unique_fe_);
  }
  if (in_space->fe_manager().same_fe_on_all_cells())
  {
    this->create_fe_sets (*(in_cell_indices.begin()), in_space, in_fe_inds, 
                          this->in_ind_2_unique_, this->in_unique_2_ind_,  this->in_unique_fe_);
  }
  else
  {
    if (in_cell_indices.size() == 1)
    {
      this->create_fe_sets (*(in_cell_indices.begin()), in_space, in_fe_inds, 
                            this->in_ind_2_unique_, this->in_unique_2_ind_,  this->in_unique_fe_);
    }
    else
    {
      this->in_unique_2_ind_.resize(in_fe_inds.size());
      for (int l=0; l!=in_fe_inds.size(); ++l)
      {
        this->in_ind_2_unique_.push_back(l);
        this->in_unique_2_ind_[l].push_back(l);
      }
    }
  }
  */
}

template < class DataType, int DIM >
void InterpolationPattern<DataType, DIM>::create_fe_sets (int cell_index,
                                                          VectorSpace< DataType, DIM> const * space,
                                                          const std::vector<size_t>& fe_inds, 
                                                          std::vector< int >& ind_2_unique,
                                                          std::vector< std::vector< int > >& unique_2_ind,
                                                          std::vector< doffem::RefElement< DataType, DIM > const * >& unique_fe) const
{
  std::vector< doffem::RefElement< DataType, DIM > const * > ref_fe;

  for (int l = 0; l!=fe_inds.size(); ++l)
  {
    size_t fe_ind = fe_inds[l];
    ref_fe.push_back(space->fe_manager().get_fe(cell_index, fe_ind).get());
  }

  create_unique_objects_mapping< const doffem::RefElement< DataType, DIM > > (ref_fe,
                                                                              ind_2_unique,
                                                                              unique_2_ind,
                                                                              unique_fe);
}

template < class DataType, int DIM >
bool InterpolationPattern<DataType, DIM>::same_fe_pattern (const InterpolationPattern<DataType, DIM>& rhs) const 
{
  if (this->in_ind_2_unique_ != rhs.in_ind_2_unique_)
  {
    return false;
  }
  if (this->out_ind_2_unique_ != rhs.out_ind_2_unique_)
  {
    return false;
  }
  if (this->in_unique_fe_.size() != rhs.in_unique_fe_.size())
  {
    return false;
  }
  if (this->out_unique_fe_.size() != rhs.out_unique_fe_.size())
  {
    return false;
  }
  for (int i = 0, e_i = this->in_unique_fe_.size(); i != e_i; ++i)
  {
    if ( !(*this->in_unique_fe_[i] == *rhs.in_unique_fe_[i]) )
    {
      return false;
    }
  }
  for (int i = 0, e_i = this->out_unique_fe_.size(); i != e_i; ++i)
  {
    if ( !(*this->out_unique_fe_[i] == *rhs.out_unique_fe_[i]) )
    {
      return false;
    }
  }
  return true;
}

template < class DataType, int DIM >
bool InterpolationPattern<DataType, DIM>::same_cell_pattern (const InterpolationPattern<DataType, DIM>& rhs) const 
{
  return this->same_cell_pattern_impl (this->in_space_,
                                  this->out_space_,
                                  this->out_cell_index_,
                                  rhs.out_cell_index_,
                                  this->in_cell_indices_,
                                  rhs.in_cell_indices_);
}

template < class DataType, int DIM >
bool InterpolationPattern<DataType, DIM>::is_cell_match() const
{
  boost::intrusive_ptr< const mesh::MeshDbView > in_mesh_dbview =
    boost::static_pointer_cast< const mesh::MeshDbView >(this->in_space_->meshPtr());

  boost::intrusive_ptr< const mesh::MeshDbView > out_mesh_dbview =
    boost::static_pointer_cast< const mesh::MeshDbView >(this->out_space_->meshPtr());

  if (out_mesh_dbview == 0 || in_mesh_dbview == 0)
  {
    return false;
  }

  if (in_mesh_dbview->get_db() != out_mesh_dbview->get_db())
  {
    return false;
  }

  if (this->in_cell_indices_.size() != 1)
  {
    return false;
  }

  if (this->in_fe_inds_ != this->out_fe_inds_)
  {
    return false;
  }

  const int tdim = this->out_space_->meshPtr()->tdim();
  mesh::Id out_cell_id = this->out_space_->meshPtr()->get_id(tdim, this->out_cell_index_);
  mesh::Id in_cell_id = this->in_space_->meshPtr()->get_id(tdim, *(this->in_cell_indices_.begin()));

  return out_cell_id == in_cell_id;  
}

template < class DataType, int DIM >
bool InterpolationPattern<DataType, DIM>::same_cell_pattern_impl (VectorSpace< DataType, DIM> const * in_space,
                                                                  VectorSpace< DataType, DIM> const * out_space,
                                                                  int out_index_A,
                                                                  int out_index_B,
                                                                  const std::set<int>& in_indices_A,
                                                                  const std::set<int>& in_indices_B) const
{
  assert (in_space != nullptr);
  assert (out_space != nullptr);
  assert (out_index_A >= 0);
  assert (out_index_B >= 0);
  
  // step 0: check if both cells are identical 
  if (out_index_A == out_index_B)
  {
    if (in_space->meshPtr().get() == out_space->meshPtr().get())
    {
      // same mesh
      return true;
    }
  }

  // step 1: check if both out_cells are covered by the same number of in_cells
  if (in_indices_A.size() != in_indices_B.size())
  {
    // different number of covering in_cells
    return false;
  }
  const int num_in_cells = in_indices_A.size();

  // step 2: check if out_cell_A and out_cell_B only differ by translation
  auto trafo_outA = out_space->get_cell_transformation(out_index_A);
  auto trafo_outB = out_space->get_cell_transformation(out_index_B);
  
  if (!trafo_outA->differs_by_translation_from(trafo_outB))
  {
    // they differ more than just by translation
    return false;
  }

  // compute displacement between out cell A and B
  vec trans_out_AB = trafo_outA->get_coordinate(0) - trafo_outB->get_coordinate(0);

  // step 3: check if a all in_cells differ only by translation and if this displacement vector coincides 
  // with the out dis vector

  auto it_A = in_indices_A.begin();
  auto it_B = in_indices_B.begin();

  for (int c=0; c != num_in_cells; ++c)
  {
    auto trafo_inA = in_space->get_cell_transformation(*it_A);
    auto trafo_inB = in_space->get_cell_transformation(*it_B);

    if (!trafo_inA->differs_by_translation_from(trafo_inB))
    {
      // found in-cell pair that differs more than just by translation
      return false;
    }

    vec trans_in_AB = trafo_inA->get_coordinate(0) - trafo_inB->get_coordinate(0); 
    
    if (trans_in_AB != trans_out_AB)
    {
      return false;
    }
    
    it_A++;
    it_B++;
  } 
  return true;
}


} // namespace hiflow
#endif
