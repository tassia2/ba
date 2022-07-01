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
struct VelocityDirichletBC
{
  VelocityDirichletBC(DataType time,
                      int top_mat_number, 
                      DataType vel_x, 
                      DataType eps)
  : time_(time), top_mat_num_ (top_mat_number), eps_(eps),
  vel_x_ (vel_x)
  {}
  
  void evaluate(const mesh::Entity &face,
                const Vec< DIM, DataType> &pt_coord, 
                std::vector<DataType> &vals) const 
  {
    const int material_number = face.get_material_number();
    vals.clear();
    
    // **********************************************
    // TODO exercise A
    DataType time_factor = std::sin(10 * time_);
    if (material_number == top_mat_num_)
    {
      vals.resize(DIM, 0.);
      const DataType x = pt_coord[0];
      if (x < eps_ )
      {
        vals[0] = x / eps_ * vel_x_ * time_factor;
      }
      else if (x > 1. - eps_)
      {
        vals[0] = (((1-eps_)-x)/ eps_ * vel_x_ + vel_x_) * time_factor;
      }
      else
      {
        vals[0] = vel_x_ * time_factor;
      }
    }
    else
    {
      vals.resize(DIM, 0.);
    }
    // END Exercise A
    // *********************************************
  }
    
  size_t nb_comp() const {
    return DIM;
  }
  
  size_t nb_func() const {
    return 1;
  }
  
  size_t iv2ind(size_t j, size_t v) const {
    return v;
  }
  
  int top_mat_num_;
  DataType vel_x_;
  DataType eps_;
  DataType time_;
};

// Functor used for the local assembly of the stiffness matrix and load vector.
class LocalStokesAssembler : private AssemblyAssistant< DIM, DataType > 
{
public:
  
  void set_parameters (DataType time, DataType theta, DataType dt, DataType nu, DataType fz)
  {
    this->nu_ = nu;
    this->f_ = fz;
    this->dt_ = dt;
    this->time_ = time;
    this->theta_ = theta;
  }
  
  void set_previous_solution(VectorType const *prev_sol) 
  {
    prev_time_sol_ = prev_sol;
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
        // get test function values for flux variables
        Vec<DIM, DataType> phiV_i;
        Mat<DIM, DIM, DataType> DphiV_i;
         
        for (size_t var = 0; var < DIM; ++var)
        {
          phiV_i.set(var, this->Phi(i, q, var));
          for (int d = 0; d != DIM; ++d)
          {
            DphiV_i.set(var, d, this->grad_Phi(i, q, var)[d]);
          }
        }
        // get test function values for pressure variable
        DataType phiP_i = this->Phi(i, q, DIM);
        Vec<DIM, DataType> DphiP_i = this->grad_Phi(i, q, DIM);
          
        // loop over trrial DOFs <-> trial function u 
        for (int j = 0; j < num_dof; ++j) 
        {
          // get ansatz function values for velocity variables
          Vec<DIM, DataType> phiV_j;
          Mat<DIM, DIM, DataType> DphiV_j;
         
          for (size_t var = 0; var < DIM; ++var)
          {
            phiV_j.set(var, this->Phi(j, q, var));
            for (int d = 0; d != DIM; ++d)
            {
              DphiV_j.set(var, d, this->grad_Phi(j, q, var)[d]);
            }
          }
          // get ansatz function values for pressure variable
          DataType phiP_j = this->Phi(j, q, DIM);
          Vec<DIM, DataType> DphiP_j = this->grad_Phi(j, q, DIM);


          // --------------------------------------------------------------------------
          // ----- start assembly of individual terms in variational formulation ------
          
          // begin with velocity - velocity part
          if (   this->first_dof_for_var(0) <= i && i < this->last_dof_for_var(DIM-1) 
              && this->first_dof_for_var(0) <= j && j < this->last_dof_for_var(DIM-1))
          {
            // *****************************************
            // TODO EXERCISE B
            // 1 / dt * u*v + nu * theta * grad(u) : grad(v)
            DataType l0 = dot(phiV_j, phiV_i);
            DataType l1 = dot(DphiV_j, DphiV_i);
            lm(i, j) += wq * (l0 / dt_ + theta_ * this->nu_ * l1) * dJ; 
            // END EXERCISE B
            // *****************************************
          }
          
          // velocity - pressure part
          if (   this->first_dof_for_var(0) <= i && i < this->last_dof_for_var(DIM-1) 
              && this->first_dof_for_var(DIM) <= j && j < this->last_dof_for_var(DIM))
          {
            // *****************************************
            // TODO EXERCISE B
            // - p * div(v)
            DataType l1 = - phiP_j * trace(DphiV_i);
            lm(i, j) += wq * l1 * dJ;
            // END EXERCISE B
            // *****************************************
          }
        
          // pressure - velocity part
          if (   this->first_dof_for_var(DIM) <= i && i < this->last_dof_for_var(DIM) 
              && this->first_dof_for_var(0) <= j && j < this->last_dof_for_var(DIM-1))
          {
            // *****************************************
            // TODO EXERCISE B
            // div(u) * q
            DataType l2 = phiP_i * trace(DphiV_j);
            lm(i,j) += wq * l2 * dJ;
            // END EXERCISE B
            // *****************************************
          }
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
    
    // compute velocity solution values of previous time step u_(n-1) at each quadrature point xq:
    // sol_ts_[v][q] = u_(n-1)_v (xq)
    // v = velocity component
    for (int v=0; v!= DIM; ++v)
    {
      this->sol_vel_ts_[v].clear();
      this->grad_sol_vel_ts_[v].clear();
      this->evaluate_fe_function(*prev_time_sol_, v, sol_vel_ts_[v]);
      this->evaluate_fe_function_gradients(*prev_time_sol_, v, grad_sol_vel_ts_[v]);
    }
    
    // loop over quadrature points
    for (int q = 0; q < num_q; ++q) 
    {
      Vec<DIM, DataType> v_prev;
      Mat<DIM, DIM, DataType> Dv_prev;
      for (int v = 0; v != DIM; ++v)
      {
        v_prev.set(v, this->sol_vel_ts_[v][q]);
        for (int d = 0; d!= DIM; ++d)
        {
          Dv_prev.set(v,d, this->grad_sol_vel_ts_[v][q][d]);
        }
      }
      
      // quadrature weight
      const DataType wq = w(q);
      
      // volume element of cell transformation
      const DataType dJ = std::abs(this->detJ(q)); 
      
      // loop over test DOFs <-> test function v
      for (int i = 0; i < num_dof; ++i) 
      { 
        // get test function values for flux variables
        Vec<DIM, DataType> phiV_i;
        Mat<DIM, DIM, DataType> DphiV_i;
         
        for (size_t var = 0; var < DIM; ++var)
        {
          phiV_i.set(var, this->Phi(i, q, var));
          for (int d = 0; d != DIM; ++d)
          {
            DphiV_i.set(var, d, this->grad_Phi(i, q, var)[d]);
          }
        }
        // get test function values for pressure variable
        DataType phiP_i = this->Phi(i, q, DIM);
        Vec<DIM, DataType> DphiP_i = this->grad_Phi(i, q, DIM);

        // momentum equation
        if ( this->first_dof_for_var(0) <= i && i < this->last_dof_for_var(DIM-1) )
        {
          // *****************************************
          // TODO EXERCISE B
          DataType l0 = dot(v_prev, phiV_i) / dt_;
          DataType l1 = -(1. - theta_) * nu_ * dot(Dv_prev, DphiV_i);
          DataType l2 = dot (this->force(this->x(q)), phiV_i);
          
          lv[i] += wq * (l0 + l1 + l2) * dJ;
          // END EXERCISE B
          // ****************************************
        }
        
        // mass equation
        if ( this->first_dof_for_var(DIM) <= i && i < this->last_dof_for_var(DIM) ) 
        {
          // *******************************************
          // TODO EXERCISE B
          // do nothing
          lv[i] += wq * this->source(this->x(q)) * phiP_i * dJ; 
          // END EXERCISE B
          // ******************************************* 
        }
      } 
    }
  }
     
  Vec<DIM,DataType> force(Vec< DIM, DataType > pt) 
  {
    Vec<DIM,DataType> f;
    f.set(DIM-1, this->f_);
    return f;
  }
  
  DataType source(Vec< DIM, DataType > pt) 
  {
    return 0;
  }
  
  VectorType const * prev_time_sol_;
  FunctionValues< DataType > sol_vel_ts_[DIM];               // velocity solution at previous time step
  FunctionValues< Vec< DIM, DataType > > grad_sol_vel_ts_[DIM];  // gradient of velocity solution at previous time step
  
  DataType time_;
  DataType theta_;
  DataType dt_;
  DataType nu_;
  DataType f_;
};

template < int DIM, class LAD >
class DivergenceIntegral : private AssemblyAssistant< DIM, typename LAD::DataType > 
{
  typedef typename LAD::MatrixType OperatorType;
  typedef typename LAD::VectorType VectorType;
  typedef typename LAD::DataType DataType;

public:
  DivergenceIntegral(const VectorType &sol)
      : sol_(sol)
  {}

  void operator()(const Element< DataType, DIM > &element,
                  const Quadrature< DataType > &quadrature,
                  DataType &l2_div) 
  {
    // **************************************
    // TODO: EXERCISE D
    AssemblyAssistant< DIM, DataType >::initialize_for_element(element, quadrature, false);
    for (int v=0; v!= DIM; ++v)
    {
      grad_vel_[v].clear();
      this->evaluate_fe_function_gradients(sol_, v, grad_vel_[v]);
    }
    
    const int num_q = this->num_quadrature_points();

    // loop over quadrature points
    for (int q = 0; q < num_q; ++q) 
    {
      const DataType wq = this->w(q);
      DataType dJ = std::abs(this->detJ(q));
      DataType div = 0.;
      for (int v=0; v!= DIM; ++v)
      {
        div += grad_vel_[v][q][v];
      }
      l2_div += wq * div * div * dJ;
    }
    // END EXERCISE D
    // **************************************
  }
  
  const VectorType &sol_;
  FunctionValues< Vec<DIM,DataType> > grad_vel_[DIM];
};

template < int DIM, class LAD >
class PressureIntegral : private AssemblyAssistant< DIM, typename LAD::DataType > 
{
  typedef typename LAD::MatrixType OperatorType;
  typedef typename LAD::VectorType VectorType;
  typedef typename LAD::DataType DataType;

public:
  PressureIntegral(const VectorType &sol)
      : sol_(sol)
  {}

  void operator()(const Element< DataType, DIM > &element,
                  const Quadrature< DataType > &quadrature,
                  DataType &pressure) 
  {
    AssemblyAssistant< DIM, DataType >::initialize_for_element(element, quadrature, false);
    this->evaluate_fe_function(sol_, DIM, p_);

    const int num_q = this->num_quadrature_points();

    // loop over quadrature points
    for (int q = 0; q < num_q; ++q) 
    {
      const DataType wq = this->w(q);
      DataType dJ = std::abs(this->detJ(q));
      pressure += wq * p_[q] * dJ;
    }
  }
  const VectorType &sol_;
  FunctionValues< DataType > p_;
};

template < int DIM, class LAD >
class VolumeIntegral : private AssemblyAssistant< DIM, typename LAD::DataType > 
{
  typedef typename LAD::MatrixType OperatorType;
  typedef typename LAD::VectorType VectorType;
  typedef typename LAD::DataType DataType;

public:
  VolumeIntegral()
  {}

  void operator()(const Element< DataType, DIM > &element,
                  const Quadrature< DataType > &quadrature, 
                  DataType &vol) 
  {
    AssemblyAssistant< DIM, DataType >::initialize_for_element(element, quadrature, false);
    const int num_q = this->num_quadrature_points();

    // loop over quadrature points
    for (int q = 0; q < num_q; ++q) 
    {
      const DataType wq = this->w(q);
      vol += wq * std::abs(this->detJ(q));
    }
  }
};
