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

#ifndef __FEM_FE_TRANSFORMATION_H_
#define __FEM_FE_TRANSFORMATION_H_

#include <cassert>
#include <iostream>
#include <string>
#include <vector>
#include <functional>
#include <cmath>

#include "common/macros.h"
#include "common/vector_algebra_descriptor.h"
#include "common/log.h"

/// \author Philipp Gerstner

namespace hiflow {
namespace mesh {
class Entity;
}

namespace doffem {

template <class DataType, int DIM> class CellTransformation;
template <class DataType, int DIM> class RefElement;
///
/// \class FETransformation fe_transformation.h
/// \brief Abstract base class of different types of operators that map a FE defined on the reference cell to a FE on the physical cell
/// \author Philipp Gerstner


template < class DataType, int DIM > 
class FETransformation 
{
public:
  typedef std::function< size_t (size_t, size_t) > IndexFunction;
  using Coord = typename StaticLA<DIM, DIM, DataType>::ColVectorType;
  using mat = typename StaticLA<DIM, DIM, DataType>::MatrixType;

  /// Default Constructor
  FETransformation()
  : index_func_set_(false) 
  {} 
  
  virtual ~FETransformation()
  {
  }
  
  /// For a given cell transformation and a given physical point, map physical function values to reference function values
  virtual void inverse_map_shape_function_values (const CellTransformation<DataType, DIM>& cell_trafo,
                                                  const Coord& ref_pt,
                                                  size_t func_offset, size_t num_func, size_t nb_comp,
                                                  IndexFunction ind_func,
                                                  const std::vector<DataType> & mapped_vals,
                                                  std::vector<DataType>& shape_vals) const = 0; 
                                    
  /// \brief map shape functions, evaluated on reference cell, to physical cell
  /// @param[in] shape_vals values of shape functions evaluated at specific point on reference cell
  ///
  /// shape_vals = [phi_hat_{comp = 0, j = 0}(q), .... , phi_hat_{comp = 0, j = n}(q), phi_hat_{comp = 1, j = 0}(q), .... ] 
  ///
  /// @param[in] func_offset map values starting from this function index
  /// @param[in] num_func number of functions to map
  /// @param[in] nb_comp number of components for each function 
  /// @param[in] detJ determinant of JAcobian of cell transformation
  /// @param[in J jacobian of cell transformation
  /// @param[out] mapped_vals mapped values are stored here starting from func_offset, must have same structure as shape_vals
  virtual void map_shape_function_values (DataType const * detJ, 
                                          mat const * J,
                                          mat const * JinvT,
                                          size_t func_offset, size_t num_func, size_t nb_comp,
                                          const RefElement<DataType, DIM>& fe,
                                          const std::vector<DataType> & shape_vals, 
                                          std::vector<DataType>& mapped_vals) const = 0;
                                          
  /// \brief same function as above, however, cell transformation dependent values (J, detJ, ...) are computed within this function.
  /// Then, the previous routine is called. Why do we have both? For efficiency: in the local assembler, cell transformation quantities
  /// have to be computed anyway, whereas for the evaluation of FE functions, e.g. in Element, those values are not available a priori.
  /// Thus they have to be computed, but not all FE transformatons require the same set of quantities (J, detJ, ...).
  virtual void map_shape_function_values (const CellTransformation<DataType, DIM>& cell_trafo,
                                          const Coord& ref_pt,
                                          size_t func_offset, size_t num_func, size_t nb_comp,
                                          IndexFunction ind_func,
                                          const std::vector<DataType> & shape_vals,
                                          std::vector<DataType>& mapped_vals) const = 0; 

  virtual void map_shape_function_values (const CellTransformation<DataType, DIM>& cell_trafo,
                                          const Coord& ref_pt,
                                          size_t func_offset, size_t num_func, size_t nb_comp,
                                          const RefElement<DataType, DIM>& fe,
                                          const std::vector<DataType> & shape_vals,
                                          std::vector<DataType>& mapped_vals) const = 0; 

  /// \brief map gradient of shape functions, evaluated on reference cell, to physical cell
  /// @param[in] shape_grads gradients of shape functions evaluated at specific point on reference cell
  /// @param[in] shape_vals values of shape functions evaluated at specific point on reference cell

  /// shape_gradients = [grad phi_hat_{comp = 0, j = 0}(q), .... , grad phi_hat_{comp = 0, j = n}(q), grad phi_hat_{comp = 1, j = 0}(q), .... ] 

  /// @param[in] func_offset map values starting from this function index
  /// @param[in] num_func number of functions to map
  /// @param[in] nb_comp number of components for each function 
  /// @param[in] detJ determinant of JAcobian of cell transformation
  /// @param[in] J jacobian of cell transformation


  /// @param[out] mapped_grads mapped gradients are stored here starting from func_offset, must have same structure as shape_grads
  virtual void map_shape_function_gradients (DataType const * detJ, 
                                             Coord const * grad_inv_detJ, 
                                             mat const * J,
                                             mat const * Jinv,
                                             mat const * JinvT,
                                             std::vector< mat > const * H,
                                             size_t func_offset, size_t num_func, size_t nb_comp,
                                             const RefElement<DataType, DIM>& fe,
                                             const std::vector< DataType> & shape_vals, 
                                             const std::vector< Coord > & shape_grads,
                                             const std::vector< DataType> & mapped_vals,
                                             std::vector< Coord > & mapped_grads) const = 0; 

  /// \brief same function as above, however, cell transformation dependent values are computed within this function
  virtual void map_shape_function_gradients (const CellTransformation<DataType, DIM>& cell_trafo,
                                             const Coord& ref_pt,
                                             size_t func_offset, size_t num_func, size_t nb_comp,
                                             const RefElement<DataType, DIM>& fe,
                                             const std::vector< DataType> & shape_vals,
                                             const std::vector< Coord > & shape_grads,
                                             const std::vector< DataType> & mapped_vals,
                                             std::vector< Coord > & mapped_grads) const = 0; 
                                             

  /// \brief map hessian of shape functions, evaluated on reference cell, to physical cell
  /// @param[in] shape_hessians hessians of shape functions evaluated at specific point on reference cell
  /// @param[in] shape_grads gradients of shape functions evaluated at specific point on reference cell

  /// shape_hessians = [hessian phi_hat_{comp = 0, j = 0}(q), .... , hessian phi_hat_{comp = 0, j = n}(q), hessian phi_hat_{comp = 1, j = 0}(q), .... ] 

  /// @param[in] func_offset map values starting from this function index
  /// @param[in] num_func number of functions to map
  /// @param[in] nb_comp number of components for each function 

  /// @param[out] mapped_hessians mapped hessians are stored here starting from func_offset, must have same structure as shape_hessians
  virtual void map_shape_function_hessians (mat const * JinvT,
                                            std::vector< mat > const * H,
                                            size_t func_offset, size_t num_func, size_t nb_comp,
                                            const RefElement<DataType, DIM>& fe,
                                            const std::vector< Coord > & shape_grads,
                                            const std::vector< mat > & shape_hessians,
                                            const std::vector< Coord > & mapped_grads, 
                                            std::vector< mat > & mapped_hessians) const = 0;

  /// \brief same function as above, however, cell transformation dependent values are computed within this function
  virtual void map_shape_function_hessians (const CellTransformation<DataType, DIM>& cell_trafo,
                                            const Coord& ref_pt, 
                                            size_t func_offset, size_t num_func, size_t nb_comp,
                                            const RefElement<DataType, DIM>& fe,
                                            const std::vector< Coord > & shape_grads, 
                                            const std::vector< mat > & shape_hessians,
                                            const std::vector< Coord > & mapped_grads,
                                            std::vector< mat > & mapped_hessians) const = 0;
                                             
  virtual bool need_cell_trafo_hessian_for_hessians() const = 0;
  virtual bool need_cell_trafo_hessian_for_gradients() const = 0;
  virtual bool need_cell_trafo_grad_inv_detJ_for_gradients() const = 0;
  virtual bool need_cell_trafo_grad_inv_detJ_for_hessians() const = 0;
  
protected:

  mutable IndexFunction ind_func_;
  mutable bool index_func_set_;


};

/// Standard FE Transformation: phi(x) = phi_hat(x_hat), x_hat = F^{-1}(x)
/// with cell transformation F: ref_cell -> phys_cell
/// This transformation is typically used for H¹-conforming elements

template < class DataType, int DIM > 
class FETransformationStandard : public virtual FETransformation<DataType, DIM>
{
public:
  typedef std::function< size_t (size_t, size_t) > IndexFunction;
  using Coord = typename StaticLA<DIM, DIM, DataType>::ColVectorType;
  using mat = typename StaticLA<DIM, DIM, DataType>::MatrixType;

  /// Default Constructor
  FETransformationStandard()
  : FETransformation<DataType, DIM>()
  {} 
   
  ~FETransformationStandard()
  {
  }

  inline bool need_cell_trafo_hessian_for_hessians() const
  {
    return true;
  }
  
  inline bool need_cell_trafo_hessian_for_gradients() const
  {
    return false;
  }
   
  inline bool need_cell_trafo_grad_inv_detJ_for_gradients() const
  {
    return false;
  }
  
  // TODO überprüfen
  inline bool need_cell_trafo_grad_inv_detJ_for_hessians() const
  {
    return false;
  }
  
  
  void inverse_map_shape_function_values (const CellTransformation<DataType, DIM>& cell_trafo,
                                          const Coord& ref_pt,
                                          size_t func_offset, size_t num_func, size_t nb_comp,
                                          IndexFunction ind_func,
                                          const std::vector<DataType> & mapped_vals,
                                          std::vector<DataType>& shape_vals) const;
  
  void map_shape_function_values (DataType const * detJ, 
                                  mat const * J,
                                  mat const * JinvT,
                                  size_t func_offset, size_t num_func, size_t nb_comp,
                                  const RefElement<DataType, DIM>& fe,
                                  const std::vector<DataType> & shape_vals, 
                                  std::vector<DataType>& mapped_vals) const;
 
  void map_shape_function_values (const CellTransformation<DataType, DIM>& cell_trafo,
                                  const Coord& ref_pt,
                                  size_t func_offset, size_t num_func, size_t nb_comp,
                                  IndexFunction ind_func,
                                  const std::vector<DataType> & shape_vals,
                                  std::vector<DataType>& mapped_vals) const;

  void map_shape_function_values (const CellTransformation<DataType, DIM>& cell_trafo,
                                  const Coord& ref_pt,
                                  size_t func_offset, size_t num_func, size_t nb_comp,
                                  const RefElement<DataType, DIM>& fe,
                                  const std::vector<DataType> & shape_vals,
                                  std::vector<DataType>& mapped_vals) const;

  void map_shape_function_gradients (DataType const * detJ, 
                                     Coord const * grad_inv_detJ, 
                                     mat const * J,
                                     mat const * Jinv,
                                     mat const * JinvT,
                                     std::vector< mat > const * H,
                                     size_t func_offset, size_t num_func, size_t nb_comp,
                                     const RefElement<DataType, DIM>& fe,
                                     const std::vector< DataType> & shape_vals, 
                                     const std::vector< Coord > & shape_grads,
                                     const std::vector< DataType> & mapped_vals,
                                     std::vector< Coord > & mapped_grads) const;

  void map_shape_function_gradients (const CellTransformation<DataType, DIM>& cell_trafo,
                                     const Coord& ref_pt,
                                     size_t func_offset, size_t num_func, size_t nb_comp,
                                     const RefElement<DataType, DIM>& fe,
                                     const std::vector< DataType> & shape_vals,
                                     const std::vector< Coord > & shape_grads,
                                     const std::vector< DataType> & mapped_vals,
                                     std::vector< Coord > & mapped_grads) const;

  void map_shape_function_hessians (mat const * JinvT,
                                    std::vector< mat > const * H,
                                    size_t func_offset, size_t num_func, size_t nb_comp,
                                    const RefElement<DataType, DIM>& fe,
                                    const std::vector< Coord > & shape_grads,
                                    const std::vector< mat > & shape_hessians,
                                    const std::vector< Coord > & mapped_grads, 
                                    std::vector< mat > & mapped_hessians) const;
  
  void map_shape_function_hessians (const CellTransformation<DataType, DIM>& cell_trafo,
                                    const Coord& ref_pt, 
                                    size_t func_offset, size_t num_func, size_t nb_comp,
                                    const RefElement<DataType, DIM>& fe,
                                    const std::vector< Coord > & shape_grads, 
                                    const std::vector< mat > & shape_hessians,
                                    const std::vector< Coord > & mapped_grads,
                                    std::vector< mat > & mapped_hessians) const;
};

/// Contravariant Piola Transformation: phi(x) = 1. / detJ(x_hat) * J(x_hat) * phi_hat(x_hat), x_hat = F^{-1}(x)
/// with cell transformation F: ref_cell -> phys_cell, J := DF
/// This transformation is typically used for H(div)-conforming elements

template < class DataType, int DIM > 
class FETransformationContraPiola : public virtual FETransformation<DataType, DIM>
{
public:
  typedef std::function< size_t (size_t, size_t) > IndexFunction;
  using Coord = typename StaticLA<DIM, DIM, DataType>::ColVectorType;
  using mat = typename StaticLA<DIM, DIM, DataType>::MatrixType;

  /// Default Constructor
  FETransformationContraPiola()
  : FETransformation<DataType, DIM>()
  {} 
   
  ~FETransformationContraPiola()
  {
  }

  inline bool need_cell_trafo_hessian_for_hessians() const
  {
    return true;
  }
  
  inline bool need_cell_trafo_hessian_for_gradients() const
  {
    return false;
  }
   
  inline bool need_cell_trafo_grad_inv_detJ_for_gradients() const
  {
    return false;
  }

  // TODO: überprüfen
  inline bool need_cell_trafo_grad_inv_detJ_for_hessians() const
  {
    return false;
  }
    
  void inverse_map_shape_function_values (const CellTransformation<DataType, DIM>& cell_trafo,
                                          const Coord& ref_pt,
                                          size_t func_offset, size_t num_func, size_t nb_comp,
                                          IndexFunction ind_func,
                                          const std::vector<DataType> & mapped_vals,
                                          std::vector<DataType>& shape_vals) const;
  
  void map_shape_function_values (DataType const * detJ, 
                                  mat const * J,
                                  mat const * JinvT,
                                  size_t func_offset, size_t num_func, size_t nb_comp,
                                  const RefElement<DataType, DIM>& fe,
                                  const std::vector<DataType> & shape_vals, 
                                  std::vector<DataType>& mapped_vals) const;
 
  void map_shape_function_values (const CellTransformation<DataType, DIM>& cell_trafo,
                                  const Coord& ref_pt,
                                  size_t func_offset, size_t num_func, size_t nb_comp,
                                  IndexFunction ind_func,
                                  const std::vector<DataType> & shape_vals,
                                  std::vector<DataType>& mapped_vals) const;

  void map_shape_function_values (const CellTransformation<DataType, DIM>& cell_trafo,
                                  const Coord& ref_pt,
                                  size_t func_offset, size_t num_func, size_t nb_comp,
                                  const RefElement<DataType, DIM>& fe,
                                  const std::vector<DataType> & shape_vals,
                                  std::vector<DataType>& mapped_vals) const;
  /// 
  /// grad(phi_k) = J^{-T} * [ (J phi_hat)_k * g + 1. / detJ * (H^(k) * phi_hat + D (phi_hat)^{T} * J^{T}_{:,k})]
  /// with
  /// phi_k = k-th component of basis function phi, phi_hat_k = k-th component of reference basis function phi_hat
  ///
  /// phi_hat = (phi_hat_1, ..., phi_hat_DIM)^{T}
  /// 
  /// g = grad (1. / detJ)
  ///
  /// H^(k) = cell_trafo.hessian(k)
   
  void map_shape_function_gradients (DataType const * detJ, 
                                     Coord const * grad_inv_detJ, 
                                     mat const * J,
                                     mat const * Jinv,
                                     mat const * JinvT,
                                     std::vector< mat > const * H,
                                     size_t func_offset, size_t num_func, size_t nb_comp,
                                     const RefElement<DataType, DIM>& fe,
                                     const std::vector< DataType> & shape_vals, 
                                     const std::vector< Coord > & shape_grads,
                                     const std::vector< DataType> & mapped_vals,
                                     std::vector< Coord > & mapped_grads) const;

  void map_shape_function_gradients (const CellTransformation<DataType, DIM>& cell_trafo,
                                     const Coord& ref_pt,
                                     size_t func_offset, size_t num_func, size_t nb_comp,
                                     const RefElement<DataType, DIM>& fe,
                                     const std::vector< DataType> & shape_vals,
                                     const std::vector< Coord > & shape_grads,
                                     const std::vector< DataType> & mapped_vals,
                                     std::vector< Coord > & mapped_grads) const;
  
  void map_shape_function_hessians (mat const * JinvT,
                                    std::vector< mat > const * H,
                                    size_t func_offset, size_t num_func, size_t nb_comp,
                                    const RefElement<DataType, DIM>& fe,
                                    const std::vector< Coord > & shape_grads,
                                    const std::vector< mat > & shape_hessians,
                                    const std::vector< Coord > & mapped_grads, 
                                    std::vector< mat > & mapped_hessians) const;
  
  void map_shape_function_hessians (const CellTransformation<DataType, DIM>& cell_trafo,
                                    const Coord& ref_pt, 
                                    size_t func_offset, size_t num_func, size_t nb_comp,
                                    const RefElement<DataType, DIM>& fe,
                                    const std::vector< Coord > & shape_grads,
                                    const std::vector< mat > & shape_hessians,
                                    const std::vector< Coord > & mapped_grads,
                                    std::vector< mat > & mapped_hessians) const;
};

/// Covariant Piola Transformation: phi(x) = J(x_hat)^{-T} * phi_hat(x_hat), x_hat = F^{-1}(x)
/// with cell transformation F: ref_cell -> phys_cell, J := DF
/// This transformation is typically used for H(curl)-conforming elements

template < class DataType, int DIM > 
class FETransformationCoPiola : public virtual FETransformation<DataType, DIM>
{
public:
  typedef std::function< size_t (size_t, size_t) > IndexFunction;
  using Coord = typename StaticLA<DIM, DIM, DataType>::ColVectorType;
  using mat = typename StaticLA<DIM, DIM, DataType>::MatrixType;
  
  /// Default Constructor
  FETransformationCoPiola()
  : FETransformation<DataType, DIM>()
  {} 
   
  ~FETransformationCoPiola()
  {
  }

  inline bool need_cell_trafo_hessian_for_hessians() const
  {
    return true;
  }
  
  inline bool need_cell_trafo_hessian_for_gradients() const
  {
    return true;
  }
   
  inline bool need_cell_trafo_grad_inv_detJ_for_gradients() const
  {
    return false;
  }
  
  // TODO überprüfen
  inline bool need_cell_trafo_grad_inv_detJ_for_hessians() const
  {
    return false;
  }
  
  void inverse_map_shape_function_values (const CellTransformation<DataType, DIM>& cell_trafo,
                                          const Coord& ref_pt,
                                          size_t func_offset, size_t num_func, size_t nb_comp,
                                          IndexFunction ind_func,
                                          const std::vector<DataType> & mapped_vals,
                                          std::vector<DataType>& shape_vals) const;
  
  void map_shape_function_values (DataType const * detJ, 
                                  mat const * J,
                                  mat const * JinvT,
                                  size_t func_offset, size_t num_func, size_t nb_comp,
                                  const RefElement<DataType, DIM>& fe,
                                  const std::vector<DataType> & shape_vals, 
                                  std::vector<DataType>& mapped_vals) const;
 
  void map_shape_function_values (const CellTransformation<DataType, DIM>& cell_trafo,
                                  const Coord& ref_pt,
                                  size_t func_offset, size_t num_func, size_t nb_comp,
                                  IndexFunction ind_func,
                                  const std::vector<DataType> & shape_vals,
                                  std::vector<DataType>& mapped_vals) const;

  void map_shape_function_values (const CellTransformation<DataType, DIM>& cell_trafo,
                                  const Coord& ref_pt,
                                  size_t func_offset, size_t num_func, size_t nb_comp,
                                  const RefElement<DataType, DIM>& fe,
                                  const std::vector<DataType> & shape_vals,
                                  std::vector<DataType>& mapped_vals) const;
  /// 
  /// grad(phi_k) = J^{-T}_{k,:} * [D(phi_hat) - sum_l H[l] * phi_l  ] * J^{-1}
  /// with
  /// phi_k = k-th component of basis function phi, phi_hat_k = k-th component of reference basis function phi_hat
  ///
  /// phi_hat = (phi_hat_1, ..., phi_hat_DIM)^{T}
  /// 
  /// g = grad (1. / detJ)
  ///
  /// H^(k) = cell_trafo.hessian(k)
   
  void map_shape_function_gradients (DataType const * detJ, 
                                     Coord const * grad_inv_detJ, 
                                     mat const * J,
                                     mat const * Jinv,
                                     mat const * JinvT,
                                     std::vector< mat > const * H,
                                     size_t func_offset, size_t num_func, size_t nb_comp,
                                     const RefElement<DataType, DIM>& fe,
                                     const std::vector< DataType> & shape_vals, 
                                     const std::vector< Coord > & shape_grads,
                                     const std::vector< DataType> & mapped_vals,
                                     std::vector< Coord > & mapped_grads) const;

  void map_shape_function_gradients (const CellTransformation<DataType, DIM>& cell_trafo,
                                     const Coord& ref_pt,
                                     size_t func_offset, size_t num_func, size_t nb_comp,
                                     const RefElement<DataType, DIM>& fe,
                                     const std::vector< DataType> & shape_vals,
                                     const std::vector< Coord > & shape_grads,
                                     const std::vector< DataType> & mapped_vals,
                                     std::vector< Coord > & mapped_grads) const;
  
  void map_shape_function_hessians (mat const * JinvT,
                                    std::vector< mat > const * H,
                                    size_t func_offset, size_t num_func, size_t nb_comp,
                                    const RefElement<DataType, DIM>& fe,
                                    const std::vector< Coord > & shape_grads,
                                    const std::vector< mat > & shape_hessians,
                                    const std::vector< Coord > & mapped_grads, 
                                    std::vector< mat > & mapped_hessians) const;
  
  void map_shape_function_hessians (const CellTransformation<DataType, DIM>& cell_trafo,
                                    const Coord& ref_pt, 
                                    size_t func_offset, size_t num_func, size_t nb_comp,
                                    const RefElement<DataType, DIM>& fe,
                                    const std::vector< Coord > & shape_grads,
                                    const std::vector< mat > & shape_hessians,
                                    const std::vector< Coord > & mapped_grads,
                                    std::vector< mat > & mapped_hessians) const;
};


} // namespace doffem
} // namespace hiflow
#endif
