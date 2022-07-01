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

// Functor used to impose u(x) = c on the boundary.
struct DirichletBC
{
  DirichletBC(int dir_mat_number)
  : dir_mat_num_ (dir_mat_number)
  {}
  
  void evaluate(const mesh::Entity &face,
                const Vec< DIM, DataType> &pt_coord, 
                std::vector<DataType> &vals) const 
  {
    const int material_number = face.get_material_number();
    vals.clear();
    
    // TODO exercise D
    // impose dirichlet boundary conditions on left boundary only
    if (material_number == dir_mat_num_)
    {
      DataType dir_value;
      if (pt_coord[1] <= 0.5)
      {
        dir_value = 0.;
      }
      else
      {
        dir_value = -1.;
      }
      vals = std::vector< DataType >(1, dir_value);
    }
    
    // other boundaries are of Neumann type
    // END Exercise D
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
  
  int dir_mat_num_;
  
};

// Functor used for the local assembly of the stiffness matrix and load vector.
class LocalPoissonAssembler : private AssemblyAssistant< DIM, DataType > 
{
public:
  
  void set_parameters (DataType kappa, DataType neumann_bc_val, int dir_mat_num)
  {
    this->kappa_ = kappa;
    this->neumann_bc_val_ = neumann_bc_val;
    this->dir_mat_num_ = dir_mat_num;
  }
  
  void set_newton_solution(VectorType const *newton_sol) 
  {
    prev_newton_sol_ = newton_sol;
  }
  
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

    // compute solution values of previous newton iterate u_k at each quadrature point xq:
    // sol_ns_[q] = u_k (xq)
    sol_ns_.clear();
    grad_sol_ns_.clear();
    this->evaluate_fe_function(*prev_newton_sol_, 0, sol_ns_);
    this->evaluate_fe_function_gradients(*prev_newton_sol_, 0, grad_sol_ns_);
    
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

          // ********************************
          // TODO exercise A)
          
          lm(i, j) += wq * 
                    (2. * kappa_ *  this->Phi(j, q, 0) * sol_ns_[q] * dot(grad_sol_ns_[q], this->grad_Phi(i, q, 0)) 
                     + (1. + kappa_ * sol_ns_[q] * sol_ns_[q]) * dot(this->grad_Phi(j, q, 0), this->grad_Phi(i, q, 0)) )
                   * dJ;
                   
          // END EXERCISE A)
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

    // compute solution values of previous newton iterate u_k at each quadrature point xq:
    // sol_ns_[q] = u_k (xq)
    sol_ns_.clear();
    grad_sol_ns_.clear();
    this->evaluate_fe_function(*prev_newton_sol_, 0, sol_ns_);
    this->evaluate_fe_function_gradients(*prev_newton_sol_, 0, grad_sol_ns_);
    
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
        // TODO exercise B)
        
        lv[i] += wq 
                * ( (1. + kappa_ * sol_ns_[q] * sol_ns_[q]) * dot(grad_sol_ns_[q], this->grad_Phi(i, q, 0)) 
                    - f(this->x(q)) * this->Phi(i, q, 0)) 
                * dJ;
                
        // END EXERCISE B)
      } 
    }
  }
   
  // compute local right hand side vector contributions from boundary facets
  // [in]  element:    contains information about current cell
  // [in]  facet_number: local index of element facet
  // [in]  quadrature: quadrature rule to be used for approximating the integrals
  // [out] lv: contribution of the current cell to the global system right hand side
  void operator()(const Element< DataType, DIM > &element, 
                  int facet_number,
                  const Quadrature< DataType > &quadrature,
                  LocalVector &lv)
  {
    const bool need_basis_hessians = false;
    
    AssemblyAssistant< DIM, DataType >::initialize_for_facet(element,
                                                             quadrature, 
                                                             facet_number,
                                                             need_basis_hessians);
                                                             
    // get material number of current facet
    IncidentEntityIterator facet = element.get_cell().begin_incident(DIM - 1);
    for (int i = 0; i < facet_number; ++i, ++facet) {}
    
    const int material_number = facet->get_material_number();
  
    // number of degrees of freedom on current cell
    const size_t num_dof = this->num_dofs_total();
    
    // number of quadrature points
    const int num_q = this->num_quadrature_points();
    
    // loop over quadrature points
    for (int q = 0; q < num_q; ++q) 
    {
      // quadrature weight
      const DataType wq = w(q);
      
      // surface element of cell transformation
      const DataType dS = std::abs(this->ds(q)); 
      
      // loop over test DOFs <-> test function v
      for (int i = 0; i < num_dof; ++i) 
      { 
        // x(q): physical coordinates of quadrature point q
        
        // TODO exercise C)
        
        if (material_number == dir_mat_num_)
        {
          // skip Dirichlet BC
          continue;
        }
        DataType neumann_bc = this->NeumannBC(this->x(q));
        
        lv[i] -= wq 
                * neumann_bc 
                * this->Phi(i,q,0)
                * dS;
                
        // END EXERCISE C)
      } 
    }
  }
  
  DataType NeumannBC (Vec< DIM, DataType > pt) 
  {
    return neumann_bc_val_;
  }
  
  DataType f(Vec< DIM, DataType > pt) 
  {
    return 1.;
  }
  
  FunctionValues< DataType > sol_ns_; // solution at previous newton step
  FunctionValues< Vec< DIM, DataType > > grad_sol_ns_; // gradient of solution at previous newton step
  
  DataType kappa_, neumann_bc_val_;
  int dir_mat_num_;
  VectorType const * prev_newton_sol_;
  
};
