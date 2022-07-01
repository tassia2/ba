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

#ifndef HIFLOW_ELEMENT_H
#define HIFLOW_ELEMENT_H

#include <functional>
#include "common/vector_algebra_descriptor.h"
#include "common/log.h"
#include "dof/dof_fem_types.h"
#include "fem/fe_manager.h"
#include "mesh/types.h"
#include "mesh/entity.h"
#include "space/vector_space.h"


/// @author Staffan Ronnas, Philipp Gerstner

namespace hiflow {

namespace mesh {
class Mesh;
//class Entity;
}

namespace doffem {
template <class DataType, int DIM> class RefElement;
template <class DataType, int DIM> class FEManager;
template <class DataType, int DIM> class CellTransformation;
template <class DataType, int DIM> class RefCell;
}

template < class T > class Quadrature;
template <class DataType, int DIM> class VectorSpace;

/// \brief Class representing one (physical) element of a finite element space. 
/// In case of vector valued spaces consisting of tensor products of different finite elements, 
/// this class acts as a vector valued function space.
///
/// Important: it is assumed that all individuals RefElements live on the same reference cell
///
/// \details This class provides a local view of one element of
/// the finite element space, and brings together information from
/// the mesh, dof and fem data structures.

template < class DataType, int DIM > 
class Element 
{
public:
  using Coord = typename StaticLA<DIM, DIM, DataType>::ColVectorType;
  using vec = typename StaticLA<DIM, DIM, DataType>::ColVectorType;
  using mat = typename StaticLA<DIM, DIM, DataType>::MatrixType;
  
  typedef std::function< size_t (size_t, size_t) > IndexFunction;

  /// \brief Construct an element on a given cell in a space.
  ///
  /// \param[in] space         finite element space to which element belongs
  /// \param[in] cell_index    index of cell for which element should be created

  Element(const VectorSpace< DataType, DIM > &space, int cell_index);

  // TODO: assert (initialized_)
  Element() = default;
  
  ~Element()
  {}

  void init (const VectorSpace< DataType, DIM > &space, int cell_index);

  /////////////////////////// Simple informations ////////////////////////////////////////

  /// \return the number of variables in the finite element space
  inline size_t nb_var() const
  {
    return this->space_->fe_manager().nb_var();
  }

  /// \return the number of finite elements in the finite element space.
  /// This value might differ from nb_var() if vector valued elements are used that are 
  /// not obtained by tensor product of single elements.
  inline size_t nb_fe() const
  {
    return this->space_->fe_manager().nb_fe();
  }

  inline size_t var_2_fe (size_t var) const
  {
    return this->space_->fe_manager().var_2_fe(var);
  }

  inline size_t var_2_comp (size_t var) const
  {
    return this->space_->fe_manager().var_2_comp(var);
  }

  inline std::vector<size_t> fe_2_var(size_t fe_ind) const
  {
    return this->space_->fe_manager().fe_2_var(fe_ind);
  }
  
  /// \brief Access the number of dofs for a given variable associated with the
  /// element. 
  /// \param[in]  var   the number of the variable 
  /// \return the number of dofs associated with variable var
  inline size_t nb_dof(size_t fe_ind) const
  {
    auto fe_type = this->get_fe(fe_ind).get();
    assert(fe_type != 0);
    return fe_type->nb_dof_on_cell();
  }

  inline size_t nb_comp(size_t fe_ind) const
  {
    return this->space_->fe_manager().get_fe(this->cell_index_, fe_ind)->nb_comp();
  }
  
  inline size_t dof_offset (size_t fe_ind) const
  {
    return this->dim_offsets_[fe_ind];
  }
  
  /// \return the cell index of the element.
  inline mesh::EntityNumber cell_index() const
  { 
    return this->cell_index_; 
  }
  
  /// \return true if and only if the element belongs to the local subdomain.
  bool is_local() const;
  
  /// \return true if and only if the element is adjacent to the boundary of the
  /// mesh.
  bool is_boundary() const;

  /// \return the local facet numbers of the element which lie on the boundary.
  std::vector< int > boundary_facet_numbers() const;
  
  /////////////////////////// Get pointers to complex objects ////////////////////////////
  /// \return a reference to the finite element space to which the element belongs.
  inline const VectorSpace< DataType, DIM > &get_space() const 
  { 
    return *this->space_; 
  }

  /// \return a reference to the cell entity associated with the element.
  inline const mesh::Entity &get_cell() const 
  { 
    return this->cell_;
  }

  /// \return the cell transformation associated with the element
  inline doffem::CellTrafoSPtr<DataType, DIM> get_cell_transformation(/*size_t fe_ind=0*/) const
  {
    return this->space_->fe_manager().get_cell_transformation(this->cell_.index()/*, fe_ind*/);
  }

  /// \brief Accesss the finite element ansatz for a variable on the cell.
  /// \param[in] var    the number of the variable
  /// \return a pointer to the finite element ansatz for variable var on the
  /// cell.
  inline doffem::CRefElementSPtr< DataType, DIM > get_fe(size_t fe_ind) const
  {
    return this->space_->fe_manager().get_fe(this->cell_.index(), fe_ind);
  }

  inline doffem::CRefElementSPtr< DataType, DIM > get_fe_for_var(size_t var) const
  {
    return this->space_->fe_manager().get_fe_for_var(this->cell_index_, var);
  }

  inline doffem::CRefCellSPtr<DataType, DIM> ref_cell() const
  {
    return this->space_->fe_manager().get_fe(this->cell_index_, 0)->ref_cell();
  }

  /////////////////////////// Dof index handling /////////////////////////////////////////
  /// \brief Access the dof indices for a given variables associated with the
  /// element. \param[in]  var       number of the variable \param[out] indices
  /// vector of dof indices for variable @p var associated with the element.
  inline void get_dof_indices(size_t fe_ind, std::vector< doffem::gDofId > &indices) const
  {
    this->space_->get_dof_indices(fe_ind, this->cell_index_, indices);
  }

  /// \brief Access the dof indices associated with the element.
  /// \param[out] indices   vector of dof indices associated with the element.
  inline void get_dof_indices(std::vector< doffem::gDofId > &indices) const
  {
    this->space_->get_dof_indices(this->cell_index_, indices);
  }

  /// \brief Access the dof indices on the boundary for a given variable
  /// associated with the element. \param[in]  var       number of the variable
  /// \param[out] indices   vector of dof indices for variable @p var associated
  /// with the element.
  inline void get_dof_indices_on_subentity(size_t fe_ind, int tdim, int sindex,
                                    std::vector< doffem::gDofId > &indices) const
  {
    this->space_->get_dof_indices_on_subentity(fe_ind, this->cell_index_, tdim, sindex, indices);
  }

  /// \brief Access the dof indices on the boundary associated with the element.
  /// \param[out] indices   vector of dof indices associated with the element.
  inline void get_dof_indices_on_subentity(int tdim, int sindex,
                                    std::vector< doffem::gDofId > &indices) const
  {
    this->space_->get_dof_indices_on_subentity(this->cell_index_, tdim, sindex, indices);
  }

  /////////////////////////// Evaluation of basis functions //////////////////////////////
  /// indexing of return array of routines N(pt, weight), grad_N(ot, weight) and hessians_N(pt, weight)
  /// @param[in] var considered (physical) variable
  /// @param[in] i index of basis function in function space associated to given variable
  /// Note: 0 <= i < nb_dof( var_2_fe(var) ) 
  /// \return index in array
  size_t iv2ind (size_t i, size_t var) const;

  /// evaluate all components of all (mapped) basis functions associated to specified finite element of index fe_ind
  /// -> return.size = sum_{refelements fe} fe.nb_comp * fe.dim
  /// indexing the return array is possible with routine iv2ind()
  /// important: for performance reasons, the user has tp provide the coordinate on the 
  /// corresponding reference cell
  
  void N(const Coord &ref_pt,   
         std::vector< DataType > &vals) const;
         
  void grad_N(const Coord &ref_pt, 
              std::vector< vec > &gradients) const;
              
  void hessian_N(const Coord &ref_pt, 
                 std::vector< mat > &hessians) const;

  /// evaluate all components of all (mapped) basis functions associated to specified finite element of index fe_ind
  /// -> return.size = fe(fe_ind).nb_comp * fe(fe_ind).dim
  /// indexing the return array is possible with routine get_fe(fe_ind)->iv2ind()
  /// important: for performance reasons, the user has tp provide the coordinate on the 
  /// corresponding reference cell

  void N_fe(const Coord &ref_pt, 
            size_t fe_ind, 
            std::vector< DataType > &vals) const;
  
  void grad_N_fe(const Coord &ref_pt, 
                 size_t fe_ind, 
                 std::vector< vec > &gradients) const;
  
  void N_and_grad_N_fe (const Coord &ref_pt, 
                        size_t fe_ind, 
                        std::vector< DataType > &vals,
                        std::vector< vec > &gradients) const;
                                              
  void hessian_N_fe(const Coord &ref_pt, 
                    size_t fe_ind, 
                    std::vector< mat > &hessians) const;

  /// evaluate all (mapped) basis functions associated to specified variable
  /// -> return.size = get_fe(var_2_fe(var)).dim
  /// important: for performance reasons, the user has tp provide the coordinate on the 
  /// corresponding reference cell

  void N_var(const Coord &ref_pt, 
             size_t var, 
             std::vector< DataType > &vals) const;
             
  void grad_N_var(const Coord &ref_pt, 
                  size_t var, 
                  std::vector< vec > &gradients) const;
                  
  void hessian_N_var(const Coord &ref_pt, 
                     size_t var, 
                     std::vector< mat > &hessians) const;

  void print_lagrange_dof_coords(size_t fe_ind) const;
  
private:

  std::vector<size_t> weight_offsets_;
  std::vector<size_t> dim_offsets_;
  size_t nb_comp_;
  size_t weight_size_;
  size_t dim_;
  mesh::EntityNumber cell_index_ = -1;
  
  size_t active_fe_ind_;
  std::vector<DataType> active_dof_values_;
  bool mapping_eval_;
  
  const VectorSpace< DataType, DIM > *space_;
  mesh::Entity cell_;

  mutable std::vector< DataType> cur_weights_;
  mutable std::vector< vec > cur_grads_;
  mutable std::vector< mat > cur_hessians_;

  mutable std::vector<DataType> shape_vals_;
  mutable std::vector< vec > shape_grads_;
  mutable std::vector< mat > shape_hessians_;
  mutable std::vector< DataType > mapped_vals_;
  mutable std::vector< vec > mapped_grads_;



};



} // namespace hiflow

#endif
