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

#include "space/basis_evaluation.h"

#include "common/permutation.h"
#include "fem/fe_manager.h"
#include "fem/fe_reference.h"
#include "fem/cell_trafo/cell_transformation.h"
#include "linear_algebra/vector.h"
#include "mesh/entity.h"
#include "mesh/mesh.h"
#include "mesh/geometric_search.h"
#include "mesh/geometric_tools.h"
#include "space/vector_space.h"
#include "space/element.h"
#include <boost/function.hpp>
#include <set>

namespace hiflow {

///////////////////////////////////////////////////////////////////
/////////////// BasisEvalLocal ////////////////////////////////////
///////////////////////////////////////////////////////////////////


template < class DataType, int DIM >
BasisEvalLocal<DataType, DIM>::BasisEvalLocal(const VectorSpace< DataType, DIM > &space, 
                                              size_t fe_ind)
: space_(&space), fe_ind_(fe_ind)
{
  std::set<BasisId> tmp_ids;
  const int tdim = this->space_->meshPtr()->tdim();
  std::vector< BasisId > gl_dofs_on_cell;

  for (mesh::EntityIterator cell = this->space_->meshPtr()->begin(tdim); 
       cell != this->space_->meshPtr()->end(tdim); ++cell) 
  {
    gl_dofs_on_cell.clear();  
    this->space_->get_dof_indices(fe_ind, cell->index(), gl_dofs_on_cell);
    tmp_ids.insert(gl_dofs_on_cell.begin(), gl_dofs_on_cell.end());
  }
  this->basis_ids_.clear();
  
  auto e_it = tmp_ids.end();
  for (auto it = tmp_ids.begin(); it != e_it; ++it)
  {
    this->basis_ids_.push_back(*it);
  }

  this->sort_basis_ids();
  this->setup_for_evaluation();
}

template < class DataType, int DIM >
BasisEvalLocal<DataType, DIM>::BasisEvalLocal(const VectorSpace< DataType, DIM > &space, 
                                              size_t fe_ind,
                                              CellIndex cell_index)
: space_(&space), fe_ind_(fe_ind)
{
  this->space_->get_dof_indices(this->fe_ind_, cell_index, this->basis_ids_);

  this->sort_basis_ids();
  this->setup_for_evaluation();
}

template < class DataType, int DIM >
BasisEvalLocal<DataType, DIM>::BasisEvalLocal(const VectorSpace< DataType, DIM > &space, 
                                              size_t fe_ind,
                                              const std::vector<CellIndex>& cell_indices)
: space_(&space), fe_ind_(fe_ind)
{
  std::set<BasisId> all_ids;
  std::vector<BasisId> tmp_ids;
  for (size_t l=0; l<cell_indices.size(); ++l)
  {
    const CellIndex c = cell_indices[l];
    
    tmp_ids.clear();
    assert (c >= 0);
    assert (c < this->space_->meshPtr()->num_entities(DIM));
    assert (c < this->space_->fe_manager().fe_tank_size());
    
    this->space_->get_dof_indices(this->fe_ind_, c, tmp_ids);
    all_ids.insert(tmp_ids.begin(), tmp_ids.end());
  }
  
  auto e_it = all_ids.end();
  for (auto it = all_ids.begin(); it != e_it; ++it)
  {
    this->basis_ids_.push_back(*it);
  }

  this->sort_basis_ids();
  this->setup_for_evaluation();
}

template < class DataType, int DIM >
BasisEvalLocal<DataType, DIM>::BasisEvalLocal(const VectorSpace< DataType, DIM > &space, 
                                              size_t fe_ind,
                                              const std::set<CellIndex>& cell_indices)
: space_(&space), fe_ind_(fe_ind)
{
  std::set<BasisId> all_ids;
  auto e_it = cell_indices.end();
  std::vector<BasisId> tmp_ids;

  for (auto it = cell_indices.begin(); it != e_it; ++it)
  {
    assert (*it >= 0);
    assert (*it < this->space_->meshPtr()->num_entities(DIM));
    assert (*it < this->space_->fe_manager().fe_tank_size());
    tmp_ids.clear();
    this->space_->get_dof_indices(this->fe_ind_, *it, tmp_ids);
    all_ids.insert(tmp_ids.begin(), tmp_ids.end());
  }
  
  auto e_it2 = all_ids.end();
  for (auto it = all_ids.begin(); it != e_it2; ++it)
  {
    this->basis_ids_.push_back(*it);
  }

  this->sort_basis_ids();
  this->setup_for_evaluation();
}

#ifdef ALLOW_BASIS_PICKING
template < class DataType, int DIM >
BasisEvalLocal<DataType, DIM>::BasisEvalLocal(const VectorSpace< DataType, DIM > &space, 
                                              size_t fe_ind, bool dummy,
                                              const std::vector<BasisId>& global_ids )
: space_(&space), basis_ids_(global_ids), fe_ind_(fe_ind)
{
  this->sort_basis_ids();
  this->setup_for_evaluation();
}
#endif

template < class DataType, int DIM >
void BasisEvalLocal<DataType, DIM>::clear_init()
{
  space_ = nullptr;
  fe_ind_ = 0;

  basis_ids_.clear();  
}

template < class DataType, int DIM >
void BasisEvalLocal<DataType, DIM>::clear_setup()
{
  nb_func_ = 0;
  nb_comp_ = 0;
  weight_size_ = 0;

  //inv_basis_ids_.clear();
  active_cells_.clear();
  inv_basis_support_.clear();
  dof_factors_.clear();
  global_2_cell_.clear();
  pt_multiplicity_.clear();
  is_pt_in_cell_.clear();
  pt_in_cell_.clear();
  ref_pts_.clear();
  cur_weights_.clear();
  cell_coord_.clear();
  double_coord_.clear();
}

template < class DataType, int DIM >
void BasisEvalLocal<DataType, DIM>::init(const VectorSpace< DataType, DIM > &space, 
                                         size_t fe_ind,
                                         const std::set<CellIndex>& cell_indices,
                                         bool full_init)
{
  this->clear_init();

  space_ = &space;
  fe_ind_ = fe_ind;
  this->basis_ids_.reserve(cell_indices.size() * space.nb_dof_on_cell(fe_ind, *cell_indices.begin()));

  auto e_it = cell_indices.end();

  for (auto it = cell_indices.begin(); it != e_it; ++it)
  {
    assert (*it >= 0);
    assert (*it < this->space_->meshPtr()->num_entities(DIM));
    assert (*it < this->space_->fe_manager().fe_tank_size());
    mut_tmp_ids_.clear();
    this->space_->get_dof_indices(this->fe_ind_, *it, mut_tmp_ids_);
    this->basis_ids_.insert(this->basis_ids_.end(), mut_tmp_ids_.begin(), mut_tmp_ids_.end());
  }
  
  this->sort_basis_ids();

  if (full_init)
  {
    this->setup_for_evaluation();
  }
}

template < class DataType, int DIM >
void BasisEvalLocal<DataType, DIM>::setup_for_evaluation( )
{
  this->clear_setup();

  this->mut_vars_ = this->space_->fe_2_var (this->fe_ind_);
  this->nb_func_ = this->basis_ids_.size();
  this->nb_comp_ = mut_vars_.size();
  this->weight_size_ = this->nb_func_ * this->nb_comp_;
  const CellDofIt nb_dof_per_cell = this->space_->nb_dof_on_cell(this->fe_ind_, 0);
 
  CellDofIt capacity = this->basis_ids_.size() * 10;
  this->active_cells_.reserve(capacity);
  auto active_insert_pos = this->active_cells_.begin();
  
  //std::cout << "actv capa " << this->basis_ids_.size() * 10 << std::endl;
  //this->inv_basis_ids_.reserve(this->nb_func_);

  for (auto c_it = this->basis_ids_sorted_.begin(), 
       e_it = this->basis_ids_sorted_.end(); 
       c_it != e_it; ++c_it )
  {
#ifdef USE_STL_MAP
    const BasisId i = c_it->first;
#else
    const BasisId i = *c_it;
#endif

    //this->inv_basis_ids_[i] = j;
    //std::cout << j << " / " << this->nb_func_ << " size " << this->active_cells_.size() 
    //          << " it dist = " << std::distance(this->active_cells_.begin(),active_insert_pos) << std::endl;
    auto new_insert_pos = active_insert_pos;

    assert (this->active_cells_.size() < capacity);
    this->space_->dof().global2cells (i, active_insert_pos, this->active_cells_, new_insert_pos);
    active_insert_pos = new_insert_pos;
  }

  sort_and_erase_duplicates<int>(this->active_cells_);
  const CellIt num_cells = this->active_cells_.size();
  this->inv_basis_support_.resize(num_cells);

  std::vector< BasisId > gl_dofs_on_cell;
  for (BasisIt k=0; k!=num_cells; ++k)
  {
    this->inv_basis_support_[k].reserve(nb_dof_per_cell);
    CellIndex c = this->active_cells_[k];

    this->space_->dof().get_dofs_on_cell(this->fe_ind_, c, gl_dofs_on_cell); 
  
    const CellDofIt num_dof = gl_dofs_on_cell.size();
    for (CellDofIt l = 0; l!= num_dof; ++l)
    {
      const BasisId i = gl_dofs_on_cell[l];
      if (this->valid_basis(i))
      {
        this->inv_basis_support_[k].push_back(i);
      }
    }
  }

  for (auto it = this->inv_basis_support_.begin(), 
       e_it = this->inv_basis_support_.end();
       it != e_it; ++it)
  {
    sort_and_erase_duplicates<int>(*it);
  }

  this->dof_factors_.resize(this->active_cells_.size());
  this->global_2_cell_.resize(this->active_cells_.size());

  for (CellIt k=0, e_k =this->active_cells_.size(); k != e_k; ++k)
  {
    const CellIndex c = this->active_cells_[k];
    this->dof_factors_[k].resize(this->nb_func_, 0.);
    this->global_2_cell_[k].resize(this->nb_func_, 0);

    // get dof factors
    size_t start_dof = 0;
    for (size_t f=0; f<this->fe_ind_; ++f)
    { 
      start_dof += this->space_->fe_manager().get_fe(c, f)->dim();
    }
  
    this->mut_all_gl_dof_ids_on_cell_.clear();
    this->mut_cur_dof_factors_.clear();
    
    this->space_->get_dof_indices(this->fe_ind_, c, this->mut_all_gl_dof_ids_on_cell_);
    this->space_->dof().get_dof_factors_on_cell(c, this->mut_cur_dof_factors_);
    
    const size_t nb_dof_fe = this->mut_all_gl_dof_ids_on_cell_.size();
    for (CellDofIt l = 0; l < nb_dof_fe; ++l)
    {
      const BasisId j = this->mut_all_gl_dof_ids_on_cell_[l];
      const BasisIt j_it = this->basis_id_2_it(j);

      if (j_it >= 0)
      {
        assert (j_it < this->nb_func_);
        this->dof_factors_[k][j_it] = this->mut_cur_dof_factors_[start_dof + l];
        this->global_2_cell_[k][j_it] = l;
      }
    }
  }
}

template < class DataType, int DIM >
void BasisEvalLocal<DataType, DIM>::evaluate (const Coord& pt, std::vector<DataType>& vals) const
{
  std::vector<Coord> tmp_pts(1, pt);
  std::vector< std::vector< DataType> > tmp_vals;
  this->evaluate(tmp_pts, tmp_vals);
  assert (tmp_vals.size() == 1);
  vals = tmp_vals[0];
}

template < class DataType, int DIM >
void BasisEvalLocal<DataType, DIM>::evaluate (const std::vector<Coord>& pts, 
                                              std::vector< std::vector<DataType> >& vals) const
{
  const size_t num_pt = pts.size();
  const size_t num_cell = this->active_cells_.size();
  const int tdim = this->space_->meshPtr()->tdim();
  
  vals.resize(num_pt);
  
  // setup data structures for current points
  this->search_pts(pts);
  
  this->cur_weights_.clear();
  
  // loop over all points
  for (PtIt p=0; p<num_pt; ++p)
  {
    vals[p].clear();
    vals[p].resize(this->weight_size_, 0.);
    //const Coord pt = pts[p];
    const DataType mult_factor = 1. / static_cast<DataType> (this->pt_multiplicity_[p]);
    
    // loop over all active cells (= Union of support of all considered basis functions)
    // that contain the current point
    for (size_t l=0; l<pt_in_cell_[p].size(); ++l)
    {
      // cell K_c
      const CellIt k = this->pt_in_cell_[p][l];
      const CellIndex c = this->active_cells_[k]; 
      const Coord ref_pt = this->ref_pts_[p][l];

      /*
      if (this->print_)
      {
        std::cout << "BasisEval at " << pt << " = ref pt " << ref_pt << " on cell " << c << std::endl; 
      }
      */
      
      assert (this->pt_in_cell_[p].size() == this->ref_pts_[p].size());
      assert (this->is_pt_in_cell_[p][k]);
      assert (this->active_cells_[k] < this->space_->meshPtr()->num_entities(tdim));
            
      // evaluate mapped FE basis on current cell at current point
      Element<DataType, DIM> elem (*this->space_, c);
      this->ref_fe_ = elem.get_fe(this->fe_ind_);
          
      const size_t cur_weight_size = elem.get_fe(this->fe_ind_)->weight_size();
      const size_t nb_dof_fe = elem.nb_dof(this->fe_ind_);
      
      cur_weights_.clear();
      cur_weights_.resize (cur_weight_size, 0.);
      elem.N_fe(ref_pt, this->fe_ind_, cur_weights_);
  
      // loop over all basis functions, whose respective support has a non-empty intersection with K_c
      auto end_i_it = this->inv_basis_support_[k].end();
      assert (end_i_it != this->inv_basis_support_[k].begin());
      
      for (auto i_it = this->inv_basis_support_[k].begin(); i_it != end_i_it; ++i_it)
      {
        assert (this->valid_basis(*i_it));
        const BasisIt j = this->basis_id_2_it(*i_it);
        assert (j >= 0);
        assert (j < this->nb_func_);

        //assert (this->dof_factors_[c].find(*i_it) != this->dof_factors_[c].end());
        //assert (this->global_2_cell_[k].find(*i_it) != this->global_2_cell_[k].end());
        //assert (this->pt_multiplicity_[p].find(*i_it) != this->pt_multiplicity_[p].end());
        
        const CellDofIt loc_i = this->global_2_cell_[k][j];
        assert (loc_i >= 0);
        assert (loc_i < nb_dof_fe);
        const DataType dof_factor = this->dof_factors_[k][j];
//      const DataType mult_factor = 1. / static_cast<DataType> (this->pt_multiplicity_[p].at(*i_it));

        /*        
        if (this->print_)
        {
          std::cout << " cell " << c << " dof id " << *i_it 
          *         << " dof factor " << dof_factor << " mult_factor " << mult_factor << std::endl;
        }
        */
        
        // loop over components of FE
        for (size_t cp = 0; cp < this->nb_comp_; ++cp)
        {
          vals[p][this->iv2ind(j, cp)] += dof_factor 
                                        * mult_factor 
                                        * cur_weights_[this->ref_fe_->iv2ind(loc_i, cp)];
        }
      }
    }
  }
}
  
template < class DataType, int DIM >
void BasisEvalLocal<DataType, DIM>::search_pts(const std::vector<Coord>& pts) const
{
  const size_t num_pt = pts.size();
  const size_t num_cell = this->active_cells_.size();
  const int tdim = this->space_->meshPtr()->tdim();
  
  // todo: for performance reasons: try to avoid clear / resize
  this->pt_multiplicity_.clear();
  this->is_pt_in_cell_.clear();
  this->pt_in_cell_.clear();
  this->ref_pts_.clear();
  
  this->pt_multiplicity_.resize(num_pt, 0);
  this->is_pt_in_cell_.resize(num_pt);
  this->pt_in_cell_.resize(num_pt);
  this->ref_pts_.resize(num_pt);
  
  for (PtIt p = 0; p<num_pt; ++p)
  {
    this->is_pt_in_cell_[p].resize(num_cell, false);
  }
  
  // loop over active cells
  for (CellIt k=0; k<num_cell; ++k)
  {
    const CellIndex c = this->active_cells_[k];
    
    //assert (this->inv_basis_support_.find(c) != this->inv_basis_support_.end());
    
#if 1
    // get cell transformation
    auto c_trafo = this->space_->get_cell_transformation(c);

#else
    // get vertex coordinates

    double_coord_.clear();
    
    this->space_->meshPtr()->get_coordinates(tdim, c, double_coord_);
    
    cell_coord_.clear();
    
    double_2_datatype(double_coord_, cell_coord_);
#endif       
    // loop over points
    for (PtIt p = 0; p<num_pt; ++p)
    {
      const Coord pt = pts[p];
      Coord ref_pt;
      
      // check if pt is contained in current cell 
#if 1
      bool found = c_trafo->contains_physical_point(pt, ref_pt);
#else
      bool found = mesh::point_inside_cell<DataType, DIM>(pt, cell_coord_, ref_pt);
#endif
      this->is_pt_in_cell_[p][k] = found;
      if (found)
      {
        this->pt_in_cell_[p].push_back(k);
        this->ref_pts_[p].push_back(ref_pt);
        this->pt_multiplicity_[p]++;
        
/*
        auto e_it = this->inv_basis_support_.at(c).end();
        for (auto i=this->inv_basis_support_.at(c).begin(); i!=e_it; ++i)
        {
          assert (*i >= 0);
          if (this->pt_multiplicity_[p].find(*i) == this->pt_multiplicity_[p].end())
          {
            this->pt_multiplicity_[p][*i] = 1;
          }
          else
          {
            this->pt_multiplicity_[p][*i] += 1;
          }
        }
*/
      }
    }
  }
}

template class BasisEvalLocal <float, 1>;
template class BasisEvalLocal <float, 2>;
template class BasisEvalLocal <float, 3>;
template class BasisEvalLocal <double, 1>;
template class BasisEvalLocal <double, 2>;
template class BasisEvalLocal <double, 3>;


///////////////////////////////////////////////////////////////////
/////////////// FeEvalLocal ///////////////////////////////////////
///////////////////////////////////////////////////////////////////

template < class DataType, int DIM >
FeEvalBasisLocal<DataType, DIM>::FeEvalBasisLocal(const VectorSpace< DataType, DIM > &space, 
                                                  const la::Vector<DataType> &coeff, 
                                                  size_t fe_ind )
: space_(space), fe_ind_(fe_ind), coeff_(coeff)
{
  this->nb_comp_ = space.fe_2_var(fe_ind).size();
  this->nb_func_ = 1;
  this->weight_size_ = this->nb_comp_;
  this->basis_eval_ = new BasisEvalLocal<DataType, DIM>(space, fe_ind);

  assert (this->basis_eval_->nb_comp() == this->nb_comp());
}

template < class DataType, int DIM >
FeEvalBasisLocal<DataType, DIM>::~FeEvalBasisLocal()
{
  if (this->basis_eval_ != nullptr)
  {
    delete this->basis_eval_;
  }
}

template < class DataType, int DIM >
bool FeEvalBasisLocal<DataType, DIM>::evaluate ( const Coord& pt, 
                                                 DataType& value ) const 
{
  assert (this->weight_size() == 1);
  std::vector< std::vector< DataType> > tmp_val;
  std::vector< Coord > tmp_pt (1, pt);
  std::vector<bool> found = this->evaluate_impl (tmp_pt, tmp_val);
  assert (tmp_val.size() == 1);
  value = tmp_val[0][0];
  return found[0];
}

template < class DataType, int DIM >
bool FeEvalBasisLocal<DataType, DIM>::evaluate ( const Coord& pt, 
                                                 std::vector< DataType >& vals ) const 
{
  std::vector< std::vector< DataType> > tmp_val;
  std::vector< Coord > tmp_pt (1, pt);
  std::vector<bool> found = this->evaluate_impl (tmp_pt, tmp_val);
  assert (tmp_val.size() == 1);
  vals = tmp_val[0];
  return found[0];
}

template < class DataType, int DIM >
std::vector<bool> FeEvalBasisLocal<DataType, DIM>::evaluate ( const std::vector<Coord>& pts, 
                                                              std::vector<std::vector<DataType> >& vals ) const 
{
  return this->evaluate_impl(pts, vals); 
}

template < class DataType, int DIM >
std::vector<bool> FeEvalBasisLocal<DataType, DIM>::evaluate_impl ( const std::vector<Coord>& pts, 
                                                                   std::vector<std::vector<DataType> >& vals ) const 
{
  std::vector<bool> success (pts.size(), true);
  
  if (vals.size() != pts.size())
  {
    vals.resize(pts.size());
  }

  /*
  if (this->print_)
  {  
    for (size_t p=0; p<pts.size(); ++p)
    {
      std::cout << "FeEvalBasisLocal at " << pts[p] << std::endl;
    }  
  }
  */
  
  mesh::MeshPtr meshptr = this->space_.meshPtr();
  
  const size_t w_size = this->weight_size();
  
  std::vector< std::vector<DataType> > basis_vals;
  std::vector< gDofId > basis_ids;

  //this->basis_eval_->set_print(this->print_);
  this->basis_eval_->evaluate(pts, basis_vals);
  this->basis_eval_->get_basis_ids(basis_ids);
  
  //std::cout << string_from_range(basis_ids.begin(), basis_ids.end()) << std::endl;
  
  const size_t num_basis = basis_ids.size();
  
  std::vector< DataType > coeff_val (num_basis, 0.);
  this->coeff_.GetValues(&basis_ids[0], num_basis, &coeff_val[0]);
  
  for (size_t p=0; p<pts.size(); ++p)
  {
    vals[p].clear();
    vals[p].resize(w_size, 0.);
    
    for (size_t j=0; j<num_basis; ++j)
    {
      for (size_t cp = 0; cp<this->nb_comp_; ++cp)
      {
        vals[p][this->iv2ind(0,cp)] += coeff_val[j] * basis_vals[p][basis_eval_->iv2ind(j,cp)];
      }
    }
  }
  return success;
}


template class FeEvalBasisLocal <float, 1>;
template class FeEvalBasisLocal <float, 2>;
template class FeEvalBasisLocal <float, 3>;
template class FeEvalBasisLocal <double, 1>;
template class FeEvalBasisLocal <double, 2>;
template class FeEvalBasisLocal <double, 3>;


} // namespace hiflow
