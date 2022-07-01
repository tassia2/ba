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

#ifndef __FEM_FEMANAGER_H_
#define __FEM_FEMANAGER_H_


#include <cassert>
#include <vector>

#include "common/pointers.h"
#include "common/macros.h"
#include "common/vector_algebra_descriptor.h"
#include "dof/dof_fem_types.h"
#include "fem/fe_instance.h"

namespace hiflow {

namespace mesh {
class Mesh;
}

namespace doffem {

template <class DataType, int DIM> class RefElement;
template <class DataType, int DIM> class CellTransformation;

///
/// \class FEManager femanager.h
/// \brief Ancestor Manager of information about the finite element ansatz
/// \author Michael Schick<br>Martin Baumann<br>Philipp Gerstner
///

template < class DataType, int DIM >
class FEManager {
public:
  using Coord = typename StaticLA<DIM, DIM, DataType>::ColVectorType;

  /// Constructor with parameters geometrical dimension and nb of dof variables
  explicit FEManager();

  /// Destructor
  virtual ~FEManager();

  /// Initialize reference to given mesh and automatically storing cell
  /// transformations
  void set_mesh(const mesh::Mesh &mesh);
  
  inline bool is_initialized() const;

  inline size_t tdim() const;

  /// Get total number of variables
  inline size_t nb_var() const;

  /// Get number of independent variables (<= get_nb_var() with "=" in case of pure tensor product FE space)
  inline size_t nb_fe() const;

  /// Get number of initialized Finite Elements
  size_t nb_fe_inst() const; 

  inline size_t var_2_fe(size_t var) const;

  /// Get the internal FE component of a physical variable (!=0 only in case of vector valued FE) 
  inline size_t var_2_comp(size_t var) const;

  inline std::vector<size_t> fe_2_var(size_t fe_ind) const;

  size_t nb_dof_on_cell (int cell_index) const;
  size_t nb_dof_on_cell (int cell_index, size_t fe_ind) const;
  
  /// Information if a fully continuous, tangential continuous, normal continuous or discontinuous ansatz is used
  /// for a specific variable
  inline FEConformity fe_conformity (size_t fe_ind) const;

  inline bool is_dg (size_t fe_ind) const;
  
  inline bool contain_only_lagrange_fe() const;
  
  inline int fe_tank_size() const 
  {
    return this->fe_tank_.size();
  }

  void get_status() const;
  
  void clear_cell_transformation();

  /// Information about the used FEType on a given mesh cell for a specific variable
  inline std::vector< CRefElementSPtr< DataType, DIM > > get_fe(int cell_index) const;
  inline CRefElementSPtr< DataType, DIM >                get_fe(int cell_index, size_t fe_ind) const; 
  inline CRefElementSPtr< DataType, DIM >                get_fe_for_var(int cell_index, size_t var) const; 

  int get_fe_index (CRefElementSPtr< DataType, DIM >& ref_fe, int cell_index) const;

  FEType fe_type_for_var(int cell_index, size_t var) const; 
  FEType fe_type(int cell_index, size_t fe_ind) const; 

  /// Get cell transformation for a specific mesh cell
  inline CellTrafoSPtr<DataType,DIM> get_cell_transformation(int cell_index) const;
  
  //std::shared_ptr<CellTransformation< DataType, DIM > > get_cell_transformation(int cell_index, size_t fe_ind) const;

  /// \brief Initialize FE Tank by given fe ansatz and parameters for all variables (all cells get same ansatz and parameters)
  /// param[fe][cell] : set of parameters on given cell for given fe index
  virtual void init ( const std::vector<FEType> &fe_types, 
                      const std::vector<bool> & is_dg,
                      const std::vector< std::vector< std::vector< int > > > &param);
  
  int nb_nonlinear_trafos() const 
  {
    return this->fe_inst_.nb_nonlinear_trafos();
  }

  int nb_cell_trafos() const 
  {
    return this->cell_transformation_.size();
  }
  
  inline bool same_fe_on_all_cells() const 
  {
    return this->same_fe_on_all_cells_;
  }

protected:

  /// topological Dimension
  size_t tdim_;

  /// Number of variables
  size_t nb_var_;

  /// Number of different FiniteElements (<= nb_var_)
  size_t nb_fe_;

  std::vector<size_t> var_2_comp_;
  std::vector<size_t> var_2_fe_;
  std::vector< std::vector<size_t> >fe_2_var_;
  
  /// This vector stores for every variable if fe_tank was initialized
  std::vector< bool > initialized_;
  bool fe_tank_initialized_;

  /// Fully Continuous (true) or (discontinuous, partially discontinuous)  ansatz for each variable
  std::vector< FEConformity > fe_conformity_;
  std::vector< bool > is_dg_;

  /// Const pointer to given mesh
  const mesh::Mesh *mesh_;

  /// Number of cells on the mesh
  int num_entities_;

  /// flag whether same element is used on all cells wihtin each single component 
  /// (if different components use different elements, this flag can still be true)
  bool same_fe_on_all_cells_;

  /// FE Tank, which holds pointers to all RefElement instances for every cell 
  std::vector< std::vector<CRefElementSPtr< DataType, DIM > > > fe_tank_;

  /// Instance holder for every needed FEType
  FEInstance< DataType, DIM > fe_inst_;

  /// Cell transformations for every mesh cell for every element
  std::vector<CellTrafoSPtr<DataType,DIM> >  cell_transformation_;
  //std::vector< std::vector<std::shared_ptr<CellTransformation< DataType, DIM > > > > cell_transformation_;
};

//------------ INLINE FUNCTIONS FOR FEMANAGER ---------------
template < class DataType, int DIM >
inline bool FEManager< DataType, DIM >::is_initialized() const 
{
  return this->fe_tank_initialized_;
}

template < class DataType, int DIM >
inline size_t FEManager< DataType, DIM >::tdim() const 
{
  return this->tdim_;
}

template < class DataType, int DIM >
inline size_t FEManager< DataType, DIM >::nb_var() const 
{
  return this->nb_var_;
}

template < class DataType, int DIM >
inline size_t FEManager< DataType, DIM >::nb_fe() const 
{
  return this->nb_fe_;
}

template < class DataType, int DIM >
inline size_t FEManager< DataType, DIM >::nb_fe_inst() const 
{
  return this->fe_inst_.nb_fe_inst();
}

template < class DataType, int DIM >
inline FEConformity FEManager< DataType, DIM >::fe_conformity(size_t fe_ind) const 
{
  interminable_assert(fe_ind < nb_fe_);
  return fe_conformity_[fe_ind];
}

template < class DataType, int DIM >
inline bool FEManager< DataType, DIM >::is_dg(size_t fe_ind) const 
{
  interminable_assert(fe_ind < nb_fe_);
  return is_dg_[fe_ind];
}

template < class DataType, int DIM >
inline CellTrafoSPtr<DataType,DIM> FEManager< DataType, DIM >::get_cell_transformation(int cell_index) const 
{
  //assert (this->cell_transformation_.size() == this->mesh_->num_entities(this->tdim_));
  assert (cell_index >= 0);
  assert (cell_index < cell_transformation_.size());
  return cell_transformation_[cell_index];
}

template < class DataType, int DIM >
inline size_t FEManager< DataType, DIM >::var_2_comp(size_t var) const 
{
  assert (var < var_2_comp_.size());
  return this->var_2_comp_[var];
}

template < class DataType, int DIM >
inline size_t FEManager< DataType, DIM >::var_2_fe(size_t var) const 
{
  assert (var < var_2_fe_.size());
  return this->var_2_fe_[var];
}

template < class DataType, int DIM >
inline std::vector<size_t> FEManager< DataType, DIM >::fe_2_var(size_t fe_ind) const 
{
  assert (fe_ind < fe_2_var_.size());
  return this->fe_2_var_[fe_ind];
}

template < class DataType, int DIM >
inline std::vector< CRefElementSPtr< DataType, DIM > > FEManager< DataType, DIM >::get_fe(int cell_index) const 
{
  assert (cell_index >= 0);
  assert (cell_index < this->fe_tank_.size());
  assert (this->fe_tank_.size() == this->num_entities_);

  std::vector< CRefElementSPtr< DataType, DIM > > tank (this->fe_tank_[cell_index].size());
  for (size_t l=0; l<tank.size(); ++l)
  {
    tank[l] = this->fe_tank_[cell_index][l];
  }
  return tank;
}

template < class DataType, int DIM >
inline CRefElementSPtr< DataType, DIM > FEManager< DataType, DIM >::get_fe(int cell_index, size_t fe_ind) const 
{
  assert (cell_index >= 0);
  assert (cell_index < this->fe_tank_.size());
  assert (fe_ind < this->fe_tank_[cell_index].size());
  assert (this->fe_tank_.size() == this->num_entities_);
 
  return this->fe_tank_[cell_index][fe_ind];
}

template < class DataType, int DIM >
inline CRefElementSPtr< DataType, DIM > FEManager< DataType, DIM >::get_fe_for_var(int cell_index, size_t var) const 
{
  assert (var < this->nb_var_);
  return this->get_fe(cell_index, this->var_2_fe(var));
}

template < class DataType, int DIM >
inline bool FEManager< DataType, DIM >::contain_only_lagrange_fe() const 
{
  return this->fe_inst_.contain_only_lagrange_fe();
}

} // namespace doffem
} // namespace hiflow
#endif
