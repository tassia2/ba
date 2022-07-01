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

/// \author Staffan Ronnas, Simon Gawlok, Philipp Gerstner

#ifndef _ASSEMBLY_ASSISTANT_VALUES_H_
#define _ASSEMBLY_ASSISTANT_VALUES_H_

#include "assembly/function_values.h"
#include "common/log.h"
#include "common/pointers.h"
#include "common/vector_algebra.h"
#include "common/simd_vector.h"
#include "common/vector_algebra_descriptor.h"
#include "fem/cell_trafo/cell_transformation.h"
#include "fem/fe_reference.h"
#include "mesh/types.h"
#include "quadrature/quadrature.h"
#include "space/element.h"

#include <algorithm>
#include <cmath>
#include <numeric>
//#include "boost/bind/bind.hpp"

namespace hiflow {

//using namespace boost::placeholders;
////////////////////////////////////////////////////////////////////////
//////////////// Helper functions for AssemblyAssistant ////////////////
////////////////////////////////////////////////////////////////////////

// A large part of the functionality of the AssemblyAssistant is
// implemented using the FunctionValues class evaluated with
// different functions. The definition of these functions follows below.

////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////

/// \brief Function that evaluates shape functions
///
/// \details Evaluates all shape functions for a provided set of
/// FEType objects. Takes as input a set of points and returns for
/// each point, a std::vector<double> with the values of all the
/// shape functions at that point. This is used to compute the
/// shape function values on the reference cell.
/// Here, a reording is performed in such a way, that 
///
/// shape_function_values[fe_type][comp*num_dof+i] 
///
/// denotes the comp_-th component of shape function i of fe_type-th reference finite element

template < int DIM, class DataType > 
class EvalShapeFunctions 
{
  using Mat = typename StaticLA<DIM, DIM, DataType>::MatrixType;
  using RMat = typename StaticLA<DIM, DIM-1, DataType>::MatrixType;
  using Vec = typename StaticLA<DIM, DIM, DataType>::ColVectorType;
  using RefElement = const doffem::RefElement<DataType, DIM>;
public:
  EvalShapeFunctions(const std::vector< RefElement*  > &fe_types)
      : fe_types_(fe_types) {}

  // eval functon for given quad point
  inline void operator()(int i, 
                         const Vec& pt,
                         std::vector< std::vector<DataType > > &shape_function_values) const 
  {
    typedef typename std::vector< RefElement*  >::const_iterator Iterator;

    shape_function_values.clear();
    shape_function_values.resize(fe_types_.size());
    size_t fe_type = 0;

    // Loop over RefElements 
    for (Iterator it = fe_types_.begin(), e_it = fe_types_.end(); it != e_it; ++it) 
    {
      const size_t weight_size = (*it)->weight_size();
      
      shape_function_values[fe_type].resize((*it)->weight_size(), 0.);

      (*it)->N(pt, shape_function_values[fe_type]);
      ++fe_type;
    }
  }

private:
  const std::vector< RefElement*  > &fe_types_;
};

/// \brief Function that evaluates shape function gradients
///
/// \details Evaluates the gradients of all shape functions for a
/// provided set of FEType objects. Takes as input a set of points
/// and returns for each point, a std::vector< Vec<DIM> > with the
/// values of all the shape function gradients at that point. This
/// is used to compute the shape function gradients on the
/// reference cell.

template < int DIM, class DataType > 
class EvalShapeFunctionGradients 
{
  using Mat = typename StaticLA<DIM, DIM, DataType>::MatrixType;
  using RMat = typename StaticLA<DIM, DIM-1, DataType>::MatrixType;
  using Vec = typename StaticLA<DIM, DIM, DataType>::ColVectorType;
  using RefElement = const doffem::RefElement<DataType, DIM>;

  public:
  EvalShapeFunctionGradients( const std::vector< RefElement*  > &fe_types)
      : fe_types_(fe_types) {}

  inline void operator()( int i, 
                          const Vec& pt,
                          std::vector< std::vector< Vec > > &shape_function_gradients) const 
  {
    shape_function_gradients.clear();
    shape_function_gradients.resize(fe_types_.size());
    size_t fe_type = 0;

    for (auto it = fe_types_.begin(), e_it = fe_types_.end(); it != e_it; ++it) 
    {
      const size_t weight_size = (*it)->weight_size();
      
      shape_function_gradients[fe_type].resize (weight_size);
 
      (*it)->grad_N(pt, shape_function_gradients[fe_type]);

      fe_type++;
    }
  }

private:
  const std::vector<  RefElement*  > &fe_types_;
};

template < typename InType, typename OutType > 
class ConvertVector 
{
  public:
  ConvertVector( )
  {}

  inline void operator()( int i, 
                          const std::vector< std::vector< InType > > &std_vec,
                          std::vector< std::vector< OutType > > &simd_vec) const 
  {
    simd_vec.resize(std_vec.size());
    for (int i=0, e_i = std_vec.size(); i != e_i; ++i)
    {
      simd_vec[i].resize(std_vec[i].size());
      for (int j=0, e_j = std_vec[i].size(); j != e_j; ++j)
      {
        simd_vec[i][j] = std_vec[i][j];
      }
    }
  }

private:
 
};

/// \brief Function that evaluates shape function hessians.
///
/// \details Evaluates the hessians of all shape functions for a
/// provided set of FEType objects. Takes as input a set of points
/// and returns for each point, a std::vector< Mat<DIM, DIM> > with the
/// values of all the shape function hessians at that point. This
/// is used to compute the shape function hessians on the
/// reference cell.

template < int DIM, class DataType > 
class EvalShapeFunctionHessians 
{
  using Mat = typename StaticLA<DIM, DIM, DataType>::MatrixType;
  using RMat = typename StaticLA<DIM, DIM-1, DataType>::MatrixType;
  using Vec = typename StaticLA<DIM, DIM, DataType>::ColVectorType;
  using RefElement = const doffem::RefElement<DataType, DIM>;

public:
  EvalShapeFunctionHessians(const std::vector<  RefElement*  > &fe_types)
      : fe_types_(fe_types) {}

  inline void operator()( int i, 
                          const Vec& pt,
                          std::vector< std::vector< Mat > > &shape_function_hessians) const 
  {
    shape_function_hessians.clear();
    shape_function_hessians.resize(fe_types_.size());
    size_t fe_type = 0;

    for (auto it = fe_types_.begin(), e_it = fe_types_.end(); it != e_it; ++it) 
    {
      const size_t weight_size = (*it)->weight_size();
      
      shape_function_hessians[fe_type].resize (weight_size);
      
      (*it)->hessian_N(pt, shape_function_hessians[fe_type]); 

      fe_type++;
    }
  }

private:
  const std::vector< RefElement*  > &fe_types_;
};

template < int DIM, class DataType > 
class EvalMappedShapeFunctions 
{
  using Mat = typename StaticLA<DIM, DIM, DataType>::MatrixType;
  using RMat = typename StaticLA<DIM, DIM-1, DataType>::MatrixType;
  using Vec = typename StaticLA<DIM, DIM, DataType>::ColVectorType;
  using RefElement = const doffem::RefElement<DataType, DIM>;

public:
  typedef std::function< size_t (size_t, size_t) > IndexFunction;

  EvalMappedShapeFunctions( const std::vector<  RefElement*  > &fe_types,
                            const FunctionValues< std::vector<std::vector< DataType > > > &phi_hat)
      : phi_hat_(phi_hat), 
        fe_types_(fe_types) 
  {}

  inline void operator()(int q, 
                         const DataType detJ,
                         const Mat &J,
                         const Mat &JinvT,
                         std::vector< std::vector<DataType> > &mapped_shape_functions) const 
  { 
    assert (q < phi_hat_.size());
    assert (phi_hat_[q].size() == fe_types_.size());
    
    mapped_shape_functions.clear();
    mapped_shape_functions.resize(fe_types_.size());

    // Loop over all RefElements
    for (size_t i = 0, e = fe_types_.size(); i != e; ++i) 
    {
      const size_t nb_comp = fe_types_[i]->nb_comp();
      const size_t dim = fe_types_[i]->dim();
      
      mapped_shape_functions[i].resize(fe_types_[i]->weight_size(), 0.);

      //IndexFunction ind_fun = boost::bind ( &doffem::RefElement<DataType, DIM>::iv2ind, fe_types_[i], _1, _2);
      //auto ind_fun = [this, i] (size_t _j, size_t _var) { return this->fe_types_[i]->iv2ind(_j, _var);};

      fe_types_[i]->fe_trafo()->map_shape_function_values (&detJ, &J, &JinvT, 0, dim, nb_comp, *fe_types_[i], 
                                                           phi_hat_[q][i], mapped_shape_functions[i]); 
    }
  }

private:
  const FunctionValues< std::vector<std::vector< DataType > > > &phi_hat_;
  const std::vector< RefElement*  > &fe_types_;
};

/// \brief Function for computing mapped shape function gradients.
///
/// \details Evaluates the gradients on the physical element by
/// applying the provided set of inverse transpose:s of the
/// jacobians of the cell transformations to each vector in the
/// set of shape function gradients on the reference cell
/// (grad_phi_hat). The returned vectors for each matrix JinvT are
/// in the same order as those in grad_phi_hat.

template < int DIM, class DataType > 
class EvalMappedShapeFunctionGradients 
{
  using Mat = typename StaticLA<DIM, DIM, DataType>::MatrixType;
  using RMat = typename StaticLA<DIM, DIM-1, DataType>::MatrixType;
  using Vec = typename StaticLA<DIM, DIM, DataType>::ColVectorType;
  using RefElement = const doffem::RefElement<DataType, DIM>;

public:
  typedef std::function< size_t (size_t, size_t) > IndexFunction;
  EvalMappedShapeFunctionGradients( const std::vector< RefElement*  > &fe_types,
                                    const FunctionValues< std::vector<std::vector< DataType > > > &phi,
                                    const FunctionValues< std::vector<std::vector< DataType > > > &phi_hat,
                                    const FunctionValues< std::vector<std::vector< Vec > > > &grad_phi_hat)
      : phi_(phi), 
        phi_hat_(phi_hat), 
        grad_phi_hat_(grad_phi_hat), 
        fe_types_(fe_types) 
  {}

  inline void operator()(int q, 
                         const DataType detJ,
                         const Vec grad_inv_detJ,
                         const Mat &J,
                         const Mat &Jinv,
                         const Mat &JinvT,
                         const std::vector<Mat >&H,
                         std::vector< std::vector<Vec > > &mapped_shape_function_gradients) const 
  {
    assert (q < phi_.size());
    assert (q < phi_hat_.size());
    assert (q < grad_phi_hat_.size());
    assert (phi_[q].size() == fe_types_.size());
    assert (phi_hat_[q].size() == fe_types_.size());
    assert (grad_phi_hat_[q].size() == fe_types_.size());
    
    mapped_shape_function_gradients.clear();
    mapped_shape_function_gradients.resize(fe_types_.size());

    // Loop over all RefElements
    for (size_t i = 0, e = fe_types_.size(); i != e; ++i)
    {
      const size_t nb_comp = fe_types_[i]->nb_comp();
      const size_t dim = fe_types_[i]->dim();
      mapped_shape_function_gradients[i].resize(fe_types_[i]->weight_size());

      //IndexFunction ind_fun = boost::bind ( &doffem::RefElement<DataType, DIM>::iv2ind, fe_types_[i], _1, _2);
      //auto ind_fun = [this, i] (size_t _j, size_t _var) { return this->fe_types_[i]->iv2ind(_j, _var);};

      fe_types_[i]->fe_trafo()->map_shape_function_gradients (&detJ, &grad_inv_detJ, &J, &Jinv, &JinvT, &H,
                                                              0, dim, nb_comp, *fe_types_[i],
                                                              phi_hat_[q][i], grad_phi_hat_[q][i], phi_[q][i],
                                                              mapped_shape_function_gradients[i]); 
    }
  }

private:
  const FunctionValues< std::vector< std::vector<DataType> > > &phi_;
  const FunctionValues< std::vector< std::vector<DataType> > > &phi_hat_;
  const FunctionValues< std::vector< std::vector<Vec > > > &grad_phi_hat_;
  const std::vector< RefElement*  > &fe_types_;
};

/// \brief Function for computing mapped shape function hessians.
///
/// \details Evaluates the hessians of the shape functions on the
/// physical element through the relation
/// \f$H_{\phi} = J^{-T}(H_{\hat{\phi}} - H_F\nabla{\phi})J^{-1}\f$.

template < int DIM, class DataType > 
class EvalMappedShapeFunctionHessians 
{
  using Mat = typename StaticLA<DIM, DIM, DataType>::MatrixType;
  using RMat = typename StaticLA<DIM, DIM-1, DataType>::MatrixType;
  using Vec = typename StaticLA<DIM, DIM, DataType>::ColVectorType;
  using RefElement = const doffem::RefElement<DataType, DIM>;

public:
  typedef std::function< size_t (size_t, size_t) > IndexFunction;
  EvalMappedShapeFunctionHessians(const std::vector<  RefElement*  > &fe_types,
                                  const FunctionValues< std::vector< std::vector< Vec > > > &grad_phi,
                                  const FunctionValues< std::vector< std::vector< Vec > > > &grad_phi_hat,
                                  const FunctionValues< std::vector< std::vector< Mat > > > &H_phi_hat)
      : H_phi_hat_(H_phi_hat), 
        grad_phi_hat_(grad_phi_hat), 
        grad_phi_(grad_phi), 
        fe_types_(fe_types)  
  {}

  inline void operator()(int q, 
                         const Mat &JinvT,
                         const std::vector<Mat >&H,
                         std::vector< std::vector<Mat > > &mapped_shape_function_hessians) const 
  {
    mapped_shape_function_hessians.clear();
    mapped_shape_function_hessians.resize(fe_types_.size());

    // Loop over all RefElements
    for (size_t i = 0, e = fe_types_.size(); i != e; ++i)
    {
      const size_t nb_comp = fe_types_[i]->nb_comp();
      const size_t dim = fe_types_[i]->dim();
      mapped_shape_function_hessians[i].resize(fe_types_[i]->weight_size());

      //IndexFunction ind_fun = boost::bind ( &doffem::RefElement<DataType, DIM>::iv2ind, fe_types_[i], _1, _2);
      //auto ind_fun = [this, i] (size_t _j, size_t _var) { return this->fe_types_[i]->iv2ind(_j, _var);};

      fe_types_[i]->fe_trafo()->map_shape_function_hessians (&JinvT, &H, 
                                                             0, dim, nb_comp, *fe_types_[i],
                                                             grad_phi_hat_[q][i], H_phi_hat_[q][i],grad_phi_[q][i], 
                                                             mapped_shape_function_hessians[i]); 
    }
  }

private:
  const FunctionValues< std::vector< std::vector<Mat > > > &H_phi_hat_;
  const FunctionValues< std::vector< std::vector<Vec > > > &grad_phi_hat_;
  const FunctionValues< std::vector< std::vector<Vec > > > &grad_phi_;
  const std::vector<  RefElement*  > &fe_types_;
};

template < int DIM, class DataType, class T > 
class ReorderMappedShapeT 
{
  using Mat = typename StaticLA<DIM, DIM, DataType>::MatrixType;
  using RMat = typename StaticLA<DIM, DIM-1, DataType>::MatrixType;
  using Vec = typename StaticLA<DIM, DIM, DataType>::ColVectorType;
  using RefElement = const doffem::RefElement<DataType, DIM>;

public:
  typedef std::function< size_t (size_t, size_t) > IndexFunction;
  ReorderMappedShapeT ( const std::vector< RefElement*  > &fe_types,
                        const std::vector< size_t >& var_2_fe_type,
                        const std::vector< size_t >& var_2_comp,
                        const FunctionValues< std::vector<std::vector< T > > > &T_phi,
                        const std::vector<size_t>& phi_offset_for_var)
      : T_phi_(T_phi), 
        fe_types_(fe_types),
        var_2_fe_type_(var_2_fe_type),
        var_2_comp_(var_2_comp),
        phi_offset_for_var_(phi_offset_for_var)
  {
    nb_var_ = phi_offset_for_var_.size()-1;
  }

  inline void operator()(int q,
                         const DataType detJ,
                         std::vector< T >  &reordered_shape_T) const 
  {
    if (reordered_shape_T.size() != this->phi_offset_for_var_[nb_var_])
    {
      reordered_shape_T.resize(this->phi_offset_for_var_[nb_var_]);
    }
    
    // Loop over all variables
    for (size_t var = 0; var < this->nb_var_; ++var) 
    {
      const size_t i = this->var_2_fe_type_[var];
      const size_t c = this->var_2_comp_[var];
      const size_t dim = this->fe_types_[i]->dim();
      const size_t offset = this->phi_offset_for_var_[var];
      assert (offset + dim == this->phi_offset_for_var_[var+1]);
      
      //IndexFunction ind_fun = boost::bind ( &doffem::RefElement<DataType, DIM>::iv2ind, this->fe_types_[i], _1, _2);
      auto ind_fun = [this, i] (size_t _j, size_t _var) { return this->fe_types_[i]->iv2ind(_j, _var);};

      for (size_t s=0; s<dim; ++s)
      {         
        reordered_shape_T[offset + s] = T_phi_[q][i][ind_fun(s,c)];
      }
    }
  }

private:
  const FunctionValues< std::vector<std::vector< T > > > &T_phi_;
  const std::vector< RefElement*  > &fe_types_;
  const std::vector< size_t >& phi_offset_for_var_;
  const std::vector< size_t >& var_2_fe_type_;
  const std::vector< size_t >& var_2_comp_;
  size_t nb_var_;
};

template < int DIM, class DataType > 
class EvalLaplacian {
public:
  EvalLaplacian ( size_t nb_var,
                  const FunctionValues< std::vector< Mat<DIM, DIM, DataType > > >&H_phi)
      : H_phi_(H_phi), 
        nb_var_(nb_var)
  {
  }

  inline void operator()(int q,
                         std::vector< DataType >  &laplacian) const 
  {
    laplacian.resize(nb_var_);

    // Loop over all variables
    for (size_t var = 0; var != this->nb_var_; ++var) 
    {
      laplacian[var] = trace(H_phi_[q][var]);
    }
  }

private:
  const FunctionValues< std::vector< Mat<DIM, DIM, DataType > > >& H_phi_;
  size_t nb_var_;
};

/// \brief Maps a normal of a reference element to the
/// physical element.

template < int DIM, class DataType > class EvalMappedNormal 
{
  using Mat = typename StaticLA<DIM, DIM, DataType>::MatrixType;
  using RMat = typename StaticLA<DIM, DIM-1, DataType>::MatrixType;
  using Vec = typename StaticLA<DIM, DIM, DataType>::ColVectorType;

public:
  EvalMappedNormal(const Vec &n) : n_(n) {}

  inline void operator()(int q, 
                         const Mat &JinvT,
                         Vec &mapped_n) const 
  {
    // mapped_n = JinvT * n_;
    JinvT.VectorMult(n_, mapped_n);
    mapped_n /= norm(mapped_n);
  }

private:
  const Vec &n_;
};
/// \brief Computes the coordinates of a set of points on the physical element.
///
/// \details Transforms a set of points on the reference element to
/// the physical element, using the provided CellTransform
/// object. This is used to compute the coordinates of the
/// quadrature points on the physical element.

template < int DIM, class DataType > 
class EvalPhysicalPoint 
{
  using Mat = typename StaticLA<DIM, DIM, DataType>::MatrixType;
  using RMat = typename StaticLA<DIM, DIM-1, DataType>::MatrixType;
  using Vec = typename StaticLA<DIM, DIM, DataType>::ColVectorType;


public:
  EvalPhysicalPoint(const doffem::CellTransformation< DataType, DIM > &transform)
      : transform_(transform) 
  {
    assert(DIM > 0);
    assert(DIM <= 3);
  }

  inline void operator()(int i, 
                         const Vec& pt,
                         Vec &mapped_point) const 
  {
    transform_.transform(pt, mapped_point);
  }

private:
  const doffem::CellTransformation< DataType, DIM > &transform_;
};

/// \brief Computes the jacobian of the element transformation at
/// a set of points on the reference element.
///
/// \details This is used to compute the jacobian of the element
/// transformation at the quadrature points.

template < int DIM, class DataType > class EvalPhysicalJacobian 
{
  using Mat = typename StaticLA<DIM, DIM, DataType>::MatrixType;
  using RMat = typename StaticLA<DIM, DIM-1, DataType>::MatrixType;
  using Vec = typename StaticLA<DIM, DIM, DataType>::ColVectorType;

public:
  EvalPhysicalJacobian(const doffem::CellTransformation< DataType, DIM > &transform)
      : transform_(transform) {
    assert(DIM > 0);
    assert(DIM <= 3);
  }

  inline void operator()(int i, 
                         const Vec& pt,
                         Mat &jacobian) const 
  {
    transform_.J(pt, jacobian);
  }

private:
  const doffem::CellTransformation< DataType, DIM > &transform_;
};

/// \brief Function that evaluates the hessians of the cell
/// transformation.
///
/// \details Evaluates the hessians of the cell transformation at
/// a set of points. Returns an array Mat<DIM, DIM>[DIM] with the
/// values \f$\partial_j\partial_k\f$ of the hessian of each
/// component \f$F_i\f$ of the cell transformation at that point.

template < int DIM, class DataType > 
class EvalCellTransformationHessian 
{
  using Mat = typename StaticLA<DIM, DIM, DataType>::MatrixType;
  using RMat = typename StaticLA<DIM, DIM-1, DataType>::MatrixType;
  using Vec = typename StaticLA<DIM, DIM, DataType>::ColVectorType;

public:
  EvalCellTransformationHessian(const doffem::CellTransformation< DataType, DIM > &transform)
      : transform_(transform) {}

  inline void
  operator()(int i, 
             const Vec& pt,
             std::vector< Mat > &hessian) const 
  {

    hessian.resize(DIM);
    for (size_t d =0; d<DIM; ++d)
    {
      transform_.H(pt, d, hessian[d]);
    }
  }

private:
  const doffem::CellTransformation< DataType, DIM > &transform_;
};



/// \brief Computes the determinant of a set of matrices.
///
/// \details Each evaluation computes the determinant of a
/// matrix. This is used to compute the determinants of the
/// jacobian matrices at the quadrature points.

template < int DIM, class DataType > 
struct EvalDeterminant 
{
  using Mat = typename StaticLA<DIM, DIM, DataType>::MatrixType;
  using RMat = typename StaticLA<DIM, DIM-1, DataType>::MatrixType;
  using Vec = typename StaticLA<DIM, DIM, DataType>::ColVectorType;

  inline void operator()(int i, 
                         const Mat &matrix,
                         DataType &determinant) const 
  {
    determinant = det(matrix);
  }
};

template < int DIM, class DataType > 
struct EvalAbsDeterminant 
{
  using Mat = typename StaticLA<DIM, DIM, DataType>::MatrixType;
  using RMat = typename StaticLA<DIM, DIM-1, DataType>::MatrixType;
  using Vec = typename StaticLA<DIM, DIM, DataType>::ColVectorType;

  inline void operator()(int i, 
                         const Mat &matrix,
                         DataType &determinant) const 
  {
    determinant = std::abs(det(matrix));
  }
};

template < int DIM, class DataType > 
class EvalGradInvDeterminantCellTrafo 
{
  using Mat = typename StaticLA<DIM, DIM, DataType>::MatrixType;
  using RMat = typename StaticLA<DIM, DIM-1, DataType>::MatrixType;
  using Vec = typename StaticLA<DIM, DIM, DataType>::ColVectorType;

  public:
  EvalGradInvDeterminantCellTrafo(const doffem::CellTransformation< DataType, DIM > &transform)
      : transform_(transform) {}
      
  inline void operator()(int i, 
                         const Vec& pt,
                         Vec &grad) const 
  {
    this->transform_.grad_inv_detJ(pt, grad);
  }
  
  private:
  const doffem::CellTransformation< DataType, DIM > &transform_;
};

/// \brief Computes the inverse transpose of a set of matrices.
///
/// \details Each evaluation computes the inverse transpose of a
/// matrix. This is used to compute the inverse transposes of the
/// jacobian matrices at the quadrature points.

template < int DIM, class DataType > 
struct EvalInvTranspose 
{
  using Mat = typename StaticLA<DIM, DIM, DataType>::MatrixType;
  using RMat = typename StaticLA<DIM, DIM-1, DataType>::MatrixType;
  using Vec = typename StaticLA<DIM, DIM, DataType>::ColVectorType;

  void operator()(int i, 
                  const Mat &matrix,
                  Mat &inv_T) const 
  {
    invTransp(matrix, inv_T);
  }
};

/// \brief Computes the inverse of a set of matrices.
///
/// \details Each evaluation computes the inverse of a
/// matrix. This is used to compute the inverses of the
/// jacobian matrices at the quadrature points.

template < int DIM, class DataType > 
struct EvalInverse 
{
  using Mat = typename StaticLA<DIM, DIM, DataType>::MatrixType;
  using RMat = typename StaticLA<DIM, DIM-1, DataType>::MatrixType;
  using Vec = typename StaticLA<DIM, DIM, DataType>::ColVectorType;

  void operator()(int i, 
                  const Mat &matrix,
                  Mat &inv) const 
  {
    // TODO: check why inv(matrix, inv) does not work
    Mat tmp;
    invTransp(matrix, tmp);
    trans (tmp, inv);
  }
};

/// \brief Computes the transpse of a set of matrices.
///
/// \details Each evaluation computes the transpose of a
/// matrix. This is used to compute the transposes of the
/// jacobian matrices at the quadrature points.

template < int DIM, class DataType > 
struct EvalTranspose
{
  using Mat = typename StaticLA<DIM, DIM, DataType>::MatrixType;
  using RMat = typename StaticLA<DIM, DIM-1, DataType>::MatrixType;
  using Vec = typename StaticLA<DIM, DIM, DataType>::ColVectorType;

  void operator()(int i, 
                  const Mat &matrix,
                  Mat &T) const 
  {
    trans(matrix, T);
  }
};

/// \brief Computes the trace of a set of matrices.
///
/// \details

template < int DIM, class DataType > 
struct EvalTrace
{
  void operator()(int i, 
                  const Mat< DIM, DIM, DataType > &matrix,
                  DataType &T) const 
  {
    T = trace(matrix);
  }
};

/// \brief Multiplies matrices on the right with a given matrix.
///
/// \details Each evaluation computes B = A*R. The dimensions of
/// the matrices are as follows: A -> MxN, R -> NxP, B->MxP.

template < int M, int N, int P, class DataType > 
struct EvalRightMatrixMult 
{
  using MatNP = typename StaticLA<N, P, DataType>::MatrixType;
  using MatMP = typename StaticLA<M, P, DataType>::MatrixType;
  using MatMN = typename StaticLA<M, N, DataType>::MatrixType;

  EvalRightMatrixMult(const MatNP &R) : R_(R) {}

  inline void operator()(int i, 
                         const MatMN &A,
                         MatMP &B) const 
  {
    // B = A * R_;
    MatrixMatrixMult(B, A, R_);
  }

private:
  const MatNP &R_;
};

/// \brief Evaluate surface element.
///
/// \brief Given Jacobian matrix Jf of mapping R^{D-1} -> R^D, computes
/// the surface element ds = \sqrt(det(Jf^T * Jf)).

template < int DIM, class DataType > 
struct EvalSurfaceElement 
{
  using Mat = typename StaticLA<DIM-1, DIM-1, DataType>::MatrixType;
  using PMat = typename StaticLA<DIM-1, DIM, DataType>::MatrixType;
  using RMat = typename StaticLA<DIM, DIM-1, DataType>::MatrixType;
  using Vec = typename StaticLA<DIM, DIM, DataType>::ColVectorType;

  inline void operator()(int i, 
                         const RMat &Jf,
                         DataType &ds) const 
  {
    PMat JfT;
    trans(Jf, JfT);

    Mat JfTJf;
    MatrixMatrixMult(JfTJf, JfT, Jf);

    const DataType detJfTJf = det(JfTJf);

    assert(detJfTJf > 0.);

    ds = std::sqrt(detJfTJf);
  }
};

/// \brief Evaluates a finite element function defined through the
/// values of its degrees of freedoms for different sets of shape
/// function values, typically corresponding to different points.
///
/// \details EvalFiniteElementFunction takes as input a set of
/// shape function values {\phi^(k)_i}, where k is indexing the list fe_types 
/// and i denotes the basis function index. 
/// Evaluates \sum{u_i * \phi^(fe_type)_i}. The values of u_i are given in the variable
/// local_coefficients. The index set of i is offset by the
/// variable fe_offset, so that i goes from
///
/// [fe_offset, ..., fe_offset + local_coefficients.size() [ .
///
/// The offset makes it possible to compute the values only for an
/// isolated variable. This function is used in the context of
/// AssemblyAssistant::evaluate_fe_function().

template < int DIM, class DataType > 
class EvalFiniteElementFunction 
{
  using Mat = typename StaticLA<DIM, DIM, DataType>::MatrixType;
  using RMat = typename StaticLA<DIM, DIM-1, DataType>::MatrixType;
  using Vec = typename StaticLA<DIM, DIM, DataType>::ColVectorType;

  public:
  EvalFiniteElementFunction(const size_t var,
                            const std::vector<size_t>& phi_offset_for_var,
                            const std::vector< DataType > &local_coefficients)
      : var_(var),
        phi_offset_for_var_(phi_offset_for_var),
        local_coefficients_(local_coefficients) 
  {
    offset_ = phi_offset_for_var[var];
  }

  inline void operator()(int q, 
                         const std::vector< DataType > &phi_values,
                         DataType &u_value) const 
  {
    const size_t num_dofs = local_coefficients_.size();
   
    DataType res = 0.;
//PRAGMA_LOOP_VEC
    // TODO_VECTORIZE
    for (size_t i = 0; i < num_dofs; ++i) 
    {
      res += phi_values[offset_+i] * local_coefficients_[i];
    }
    u_value = res;
  }

private:
  const size_t var_;
  size_t offset_;
  const std::vector<size_t>& phi_offset_for_var_;
  const std::vector< DataType > &local_coefficients_;
};


/// \brief Evaluates the gradient and Hessian of a finite element function
/// defined through the values of its degrees of freedoms for
/// different sets of shape function values, typically
/// corresponding to different points.
///
/// \details Does the same as \see EvalFiniteElementFunction in
/// principle, but computes \sum{u_i * \nabla{phi}_i} instead. This
/// is used in the context of
/// AssemblyAssistant::evaluate_fe_function_gradients().

template < int DIM, class DataType, class T > 
class EvalFiniteElementFunctionT 
{
  using Mat = typename StaticLA<DIM, DIM, DataType>::MatrixType;
  using RMat = typename StaticLA<DIM, DIM-1, DataType>::MatrixType;
  using Vec = typename StaticLA<DIM, DIM, DataType>::ColVectorType;

  public:
  EvalFiniteElementFunctionT(const size_t var,
                            const std::vector<size_t>& phi_offset_for_var,
                            const std::vector< DataType > &local_coefficients)
      : var_(var),
        phi_offset_for_var_(phi_offset_for_var),
        local_coefficients_(local_coefficients) 
  {
    offset_ = phi_offset_for_var[var];
  }

  inline void operator()(int q, 
                         const std::vector< T > &T_phi_values,
                         T &T_u_value) const 
  {
    const size_t num_dofs = local_coefficients_.size();

    T_u_value = T();    

//PRAGMA_LOOP_VEC
    // TODO_VECTORIZE
    for (size_t i = 0; i < num_dofs; ++i) 
    {
      T_u_value.Axpy(T_phi_values[offset_+i], local_coefficients_[i]);
    }
  }

private:
  const size_t var_;
  size_t offset_;
  const std::vector<size_t>& phi_offset_for_var_;
  const std::vector< DataType > &local_coefficients_;
};


template < int DIM, class DataType > 
class EvalFiniteElementFunctionGradientLagrange
{
  using Mat = typename StaticLA<DIM, DIM, DataType>::MatrixType;
  using RMat = typename StaticLA<DIM, DIM-1, DataType>::MatrixType;
  using Vec = typename StaticLA<DIM, DIM, DataType>::ColVectorType;
  using RefElement = const doffem::RefElement<DataType, DIM>;

  public:
  EvalFiniteElementFunctionGradientLagrange(const size_t var,
                                            const std::vector< RefElement*  > &fe_types,
                                            const std::vector< size_t >& var_2_fe_type,
                                            const std::vector< size_t >& var_2_comp,
                                            const std::vector< DataType > &local_coefficients)
      : var_(var),
        fe_ind_(var_2_fe_type[var]),
        fe_type_(fe_types[var_2_fe_type[var]]), 
        comp_ (var_2_comp[var]),
        dim_(fe_types[var_2_fe_type[var]]->dim()),
        local_coefficients_(local_coefficients),
        num_dofs_(local_coefficients.size())
        //R_(num_dofs_ % CHUNK),
        //I_(num_dofs_ / CHUNK),
        //J_(I_ * CHUNK) 
  {
    assert (dim_ == num_dofs_);
  }

  inline void operator()(int q, 
                         const Mat &JinvT,
                         const std::vector<std::vector< Vec > > &grad_phi_hat,
                         Vec &grad_u) const 
  {
    Vec tmp_grad;
    //Vec tmp_grad_2;
    
    for (size_t i = 0; i < num_dofs_; ++i) 
    //for (size_t i = 0; i+1 < num_dofs_; i += CHUNK) 
    {
      tmp_grad.Axpy(grad_phi_hat[fe_ind_][this->fe_type_->iv2ind(i,comp_)], local_coefficients_[i]);
      //tmp_grad_2.Axpy(grad_phi_hat[fe_ind_][this->fe_type_->iv2ind(i+1,comp_)], local_coefficients_[i+1]);
    }

    //for (size_t i = J_; i < num_dofs_; ++i) 
    //{
    //  tmp_grad.Axpy(grad_phi_hat[fe_ind_][this->fe_type_->iv2ind(i,comp_)], local_coefficients_[i]);
    //}

    //tmp_grad += tmp_grad_2;
    JinvT.VectorMult(tmp_grad, grad_u);
  }

private:
  //static constexpr size_t CHUNK = 2;
  //const size_t R_;
  //const size_t I_;
  //const size_t J_;
  const size_t var_;
  const size_t fe_ind_;
  const size_t comp_;
  const size_t dim_;
  const size_t num_dofs_;
  const doffem::RefElement< DataType, DIM >* fe_type_;
  const std::vector< DataType > &local_coefficients_;  
};

} // namespace hiflow

#endif /* _ASSEMBLY_ASSISTANT_VALUES_H_ */
