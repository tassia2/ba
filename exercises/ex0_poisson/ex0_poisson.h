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

/// \author Philipp Gerstner

// System includes.
#include "hiflow.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include <mpi.h>
#include <string>
#include <vector>

// TODO exercise: uncomment the following line for preparing sub exercise d)
#define nSUBEX_D

// All names are imported for simplicity.
using namespace hiflow;
using namespace hiflow::doffem;
using namespace hiflow::la;
using namespace hiflow::mesh;

// Shorten some datatypes with typedefs.
typedef LADescriptorCoupledD LAD;
typedef LAD::DataType DataType;
typedef LAD::VectorType VectorType;
typedef LAD::MatrixType MatrixType;

// DIM of the problem.
const int DIM = 2;

typedef Vec<DIM, DataType> Coord;

// Rank of the master process.
const int MASTER_RANK = 0;

// Parameters M, N and O in solution. These decide the period in x-, y- and
// z-direction respectively.
const int M = 2;
const int N = 2;
const int O = 1;

// Functor to evaluate the exact solution u of the Poisson problem
// with Dirichlet BC, and its gradient \grad u.

struct ExactSol {
  size_t nb_comp() const {
    return 1;
  }
  
  size_t nb_func() const {
    return 1;
  }

  size_t weight_size() const 
  {
    return nb_func() * nb_comp();
  }
  
  size_t iv2ind(size_t j, size_t v) const {
    return v;
  }
  
  // wrapper needed to make ExactSol compatible with FE interpolation
  void evaluate(const mesh::Entity &cell,
                const Vec< DIM, DataType > &pt, 
                std::vector<DataType>& vals) const 
  {
    vals.clear();
    vals.resize(1,0.);
    vals[0] = this->operator()(pt);
  }
  
  // evaluate analytical function at given point
  DataType operator()(const Vec< DIM, DataType > &pt) const 
  {
    const DataType x = pt[0];
    const DataType y = (DIM > 1) ? pt[1] : 0;
    const DataType z = (DIM > 2) ? pt[2] : 0;
    const DataType pi = M_PI;
    DataType solution;

    switch (DIM) 
    {
      case 2: {
        solution = 10.0 * std::sin(2. * M * pi * x) * std::sin(2. * N * pi * y);
        break;
    }
    case 3: 
    {
      solution = 10.0 * std::sin(2. * M * pi * x) * std::sin(2. * N * pi * y) *
                 std::sin(2. * O * pi * z);
      break;
    }

    default:
      assert(0);
    }
    return solution;
  }

  // evaluate gradient of analytical function at given point
  Vec< DIM, DataType > eval_grad(const Vec< DIM, DataType > &pt) const 
  {
    Vec< DIM, DataType > grad;
    const DataType x = pt[0];
    const DataType y = (DIM > 1) ? pt[1] : 0;
    const DataType z = (DIM > 2) ? pt[2] : 0;
    const DataType pi = M_PI;

    switch (DIM) 
    {
      case 2: 
      {
        grad.set(0, 20. * M * pi * std::cos(2. * M * pi * x) * std::sin(2. * N * pi * y));
        grad.set(1, 20. * N * pi * std::sin(2. * M * pi * x) * std::cos(2. * N * pi * y));
        break;
      }
      case 3: 
      {
        grad.set(0, 20. * M * pi * std::cos(2. * M * pi * x) * std::sin(2. * N * pi * y) * std::sin(2. * O * pi * z));
        grad.set(1, 20. * N * pi * std::sin(2. * M * pi * x) * std::cos(2. * N * pi * y) * std::sin(2. * O * pi * z));
        grad.set(2, 20. * O * pi * std::sin(2. * M * pi * x) * std::sin(2. * N * pi * y) * std::cos(2. * O * pi * z));
        break;
      }
      default:
        assert(0);
    }
    return grad;
  }
};

// Functor used to impose u(x) = c on the boundary.
struct DirichletConstant 
{
  DirichletConstant(DataType c)
  : c_ (c)
  {}
  
  void evaluate(const mesh::Entity &face,
                const Vec< DIM, DataType> &coords_on_face, 
                std::vector<DataType> &vals) const 
  {
    // return array with Dirichlet values for dof:s on boundary face
    vals = std::vector< DataType >(1, c_);
  }
  
  size_t nb_comp() const {
    return 1;
  }
  
  size_t nb_func() const {
    return 1;
  }
  
  size_t iv2ind(size_t j, size_t v) const {
    return v;
  }
  
  DataType c_;
};

// Functor used for the local assembly of the stiffness matrix and load vector.

template < class ExactSol >
class LocalPoissonAssembler : private AssemblyAssistant< DIM, DataType > 
{
public:
  // compute local matrix 
  // [in]  element:    contains information about current cell
  // [in]  quadrature: quadrature rule to be used for approximating the integrals
  // [out] lm: contribution of the current cell to the global system matrix
  
  void operator()(const Element< DataType, DIM > &element,
                  const Quadrature< DataType > &quadrature, 
                  LocalMatrix &lm) 
  {
    const bool need_basis_hessians = false;
    
    // AssemblyAssistant sets up the local FE basis functions for the current cell
    AssemblyAssistant< DIM, DataType >::initialize_for_element(element, quadrature, need_basis_hessians);

    // number of degrees of freedom on current cell
    const size_t num_dof = this->num_dofs_total();
    
    // number of quadrature points
    const int num_q = this->num_quadrature_points();
    
    // loop over quadrature points
    for (int q = 0; q < num_q; ++q) 
    {
      // quadrature weight
      const DataType wq = w(q);
      
      // volume element of cell transformation
      const DataType dJ = std::abs(this->detJ(q)); 
      
      // loop over test DOFs <-> test function v
      for (int i = 0; i < num_dof; ++i) 
      { 
        // loop over trrial DOFs <-> trial function u 
        for (int j = 0; j < num_dof; ++j) 
        {
          // grad_Phi(i,q,var): gradient of local basis function i of variable var, evaluated at quadrature point q
          // for scalar problems (like Poisson: var = 0)
          // for vector-valued problems (like Navier-Stokes in 2d: var = 0,1 (velocity), var=2 (pressure))

#ifdef SUBEX_D
          // ********************************
          // TODO exercise d)
          
          lm(i, j) += wq * 
                    (1. )
                   * dJ;
          // ********************************
#else
          lm(i, j) += wq * 
                    1.
                   * dJ;
#endif
        }
      }
    }
  }

  // compute local right hand side vector 
  // [in]  element:    contains information about current cell
  // [in]  quadrature: quadrature rule to be used for approximating the integrals
  // [out] lv: contribution of the current cell to the global system right hand side
  void operator()(const Element< DataType, DIM > &element,
                  const Quadrature< DataType > &quadrature, 
                  LocalVector &lv) 
  {
    const bool need_basis_hessians = false;
    
    // AssemblyAssistant sets up the local FE basis functions for the current cell
    AssemblyAssistant< DIM, DataType >::initialize_for_element(element, quadrature, need_basis_hessians);

    // number of degrees of freedom on current cell
    const size_t num_dof = this->num_dofs_total();
    
    // number of quadrature points
    const int num_q = this->num_quadrature_points();
    
    // loop over quadrature points
    for (int q = 0; q < num_q; ++q) 
    {
      // quadrature weight
      const DataType wq = w(q);
      
      // volume element of cell transformation
      const DataType dJ = std::abs(this->detJ(q)); 
      
      // loop over test DOFs <-> test function v
      for (int i = 0; i < num_dof; ++i) 
      { 
        // x(q): physical coordinates of quadrature point q
        lv[i] += wq 
                * f(this->x(q)) 
                * this->Phi(i, q, 0) 
                * dJ;
      } 
    }
  }

  DataType f(Vec< DIM, DataType > pt) 
  {
    ExactSol sol;
    DataType rhs_sol;

#ifdef SUBEX_D
    // ************************************************
    // TODO: exercise d)
    return 1.;
    // ************************************************
#else
    switch (DIM) 
    {
      case 2: 
      {
        rhs_sol = 4. * M_PI * M_PI * (M * M + N * N) * sol(pt);
        break;
      }
      case 3: 
      {
        rhs_sol = 4. * M_PI * M_PI * (M * M + N * N + O * O) * sol(pt);
        break;
      }
      default:
        assert(0);
    }
#endif
    return rhs_sol;
  }
};

// Functor used for the local evaluation of the square of the L2-norm of the
// error on each element.

template < class ExactSol >
class L2ErrorIntegrator : private AssemblyAssistant< DIM, DataType > 
{
public:
  L2ErrorIntegrator(VectorType &pp_sol) : pp_sol_(pp_sol) {}

  void operator()(const Element< DataType, DIM > &element,
                  const Quadrature< DataType > &quadrature, 
                  DataType &value) 
  {
    const bool need_basis_hessians = false;
    
    // AssemblyAssistant sets up the local FE basis functions for the current cell
    AssemblyAssistant< DIM, DataType >::initialize_for_element(element, quadrature, need_basis_hessians);

    // Evaluate the computed solution at all quadrature points.
    evaluate_fe_function(pp_sol_, 0, approx_sol_);

    const int num_q = num_quadrature_points();
    for (int q = 0; q < num_q; ++q) 
    {
      const DataType wq = w(q);
      const DataType dJ = std::abs(detJ(q));
      const DataType delta = sol_(x(q)) - approx_sol_[q];
      
      value += wq * delta * delta * dJ;
    }
  }

private:
  // coefficients of the computed solution
  const VectorType &pp_sol_;
  
  // functor to evaluate exact solution
  ExactSol sol_;
  
  // vector with values of computed solution evaluated at each quadrature point
  FunctionValues< DataType > approx_sol_;
};

// Functor used for the local evaluation of the square of the H1-norm of the
// error on each element.

template < class ExactSol >
class H1ErrorIntegrator : private AssemblyAssistant< DIM, DataType > 
{
public:
  H1ErrorIntegrator(VectorType &pp_sol) : pp_sol_(pp_sol) {}

  void operator()(const Element< DataType, DIM > &element,
                  const Quadrature< DataType > &quadrature, 
                  DataType &value) 
  {
    const bool need_basis_hessians = false;
    
    // AssemblyAssistant sets up the local FE basis functions for the current cell
    AssemblyAssistant< DIM, DataType >::initialize_for_element(element, quadrature, need_basis_hessians);

    // Evaluate the gradient of the computed solution at all quadrature points.
    evaluate_fe_function_gradients(pp_sol_, 0, approx_grad_u_);

    const int num_q = num_quadrature_points();
    for (int q = 0; q < num_q; ++q) 
    {
      const DataType wq = w(q);
      const DataType dJ = std::abs(detJ(q));
      const Vec< DIM, DataType > grad_u = sol_.eval_grad(x(q));
      
      value += wq
             * dot(grad_u-approx_grad_u_[q], grad_u-approx_grad_u_[q]) 
             * dJ;
    }
  }

private:
  // coefficients of the computed solution
  const VectorType &pp_sol_;
  
  // functor to evaluate exact solution
  ExactSol sol_;
  
  // gradient of computed solution evaluated at each quadrature point
  FunctionValues< Vec< DIM, DataType > > approx_grad_u_;
};
