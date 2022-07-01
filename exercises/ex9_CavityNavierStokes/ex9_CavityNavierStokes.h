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

const static double pi = 3.14159265;

typedef Vec<DIM, DataType> Coord;

// Rank of the master process.
const int MASTER_RANK = 0;

class ExactSol
{
public:
  
  ExactSol()
  {
    max_v_ = 1.;
  }
  
  void set_fe_ind (size_t fe_ind)
  {
    this->fe_ind_ = fe_ind;
    if (fe_ind == 0)
    {
      this->nb_comp_ = DIM;
    }
    else
    {
      this->nb_comp_ = 1;
    }
  }
  
  size_t nb_func() const 
  {
    return 1;
  }
  
  size_t nb_comp() const 
  {
    return this->nb_comp_;
  }
  
  size_t weight_size() const 
  {
    return nb_func() * nb_comp();
  }
  
  inline size_t iv2ind (size_t i, size_t var ) const 
  {
    assert (i==0);
    assert (var < this->nb_comp());
    return var;
  }
  
  void evaluate(const Entity& cell, const Vec<DIM, DataType> & x, DataType time, std::vector<DataType>& vals) const
  {
    assert (DIM >= 2);
    vals.clear();
    vals.resize (this->nb_comp(), 0.);
    
    std::vector<DataType> tmp_vals = this->eval (x, time);
    if (this->fe_ind_ == 0)
    {
      for (size_t d=0; d<DIM; ++d)
      {
        vals[d] = tmp_vals[d];
      }
    }
    else if (this->fe_ind_ == 1)
    {
      vals[0] = tmp_vals[DIM];
    }
    else
    {
      assert (false); 
    }
  }
  
  DataType eval_time_factor (DataType time) const 
  {
    return std::sin(10 * pi * time);
  }
  
  DataType eval_dt (DataType time) const 
  {
    return 10 * pi * std::cos(10 * pi * time);
  }
  
  std::vector<DataType> eval(const Vec< DIM, DataType > &x, DataType time) const 
  {
    std::vector<DataType> sol (DIM+1, 0.);
    
    // vel_x
    sol[0] = this->max_v_ * std::sin(pi * x[0]) * std::sin(pi * x[0]) * std::sin(2. * pi * x[1]);

    // vel_y
    sol[1] = - this->max_v_ * std::sin(2. * pi * x[0]) * std::sin(pi * x[1]) * std::sin(pi * x[1]);
    
    // pressure
    sol[DIM] = std::cos(2. * pi * x[1]);
    
    for (int d=0; d!=DIM+1; ++d)
    {
      sol[d] *= this->eval_time_factor(time);
    }
    return sol;
  }

  std::vector<DataType> eval_dt(const Vec< DIM, DataType > &x, DataType time) const 
  {
    std::vector<DataType> sol (DIM+1, 0.);
    
    // vel_x
    sol[0] = this->max_v_ * std::sin(pi * x[0]) * std::sin(pi * x[0]) * std::sin(2. * pi * x[1]);

    // vel_y
    sol[1] = - this->max_v_ * std::sin(2. * pi * x[0]) * std::sin(pi * x[1]) * std::sin(pi * x[1]);
    
    // pressure
    sol[DIM] = std::cos(2. * pi * x[1]);
    
    for (int d=0; d!=DIM+1; ++d)
    {
      sol[d] *= this->eval_dt(time);
    }
    
    return sol;
  }
  
  std::vector< Vec<DIM, DataType> > eval_grad(const Vec< DIM, DataType > &x, DataType time) const 
  {
    std::vector< Vec<DIM,DataType> > sol (DIM+1);
    // sol[v][d] = \partial_d [sol_v]

    sol[0].set(0, 2. * pi * this->max_v_ * std::sin(pi * x[0]) * std::cos(pi * x[0]) * std::sin(2. * pi * x[1])   );
    sol[0].set(1, 2. * pi * this->max_v_ * std::sin(pi * x[0]) * std::sin(pi * x[0]) * std::cos(2. * pi * x[1])   );
    sol[1].set(0, - 2. * pi * this->max_v_ * std::cos(2. * pi * x[0]) * std::sin(pi * x[1]) * std::sin(pi * x[1]) );
    sol[1].set(1, - 2. * pi * this->max_v_ * std::sin(2. * pi * x[0])  * std::sin(pi * x[1]) * std::cos(pi * x[1]));

    
    sol[DIM].set(0, 0.);
    sol[DIM].set(1, -2. * pi * std::sin(2. * pi * x[1]));    

    for (int d=0; d!=DIM+1; ++d)
    {
      sol[d] *= this->eval_time_factor(time);
    }

    return sol;
  }
  
  std::vector<DataType> eval_laplace(const Vec< DIM, DataType > &x, DataType time) const 
  {
    std::vector<DataType> sol (DIM+1, 0.);
    // sol[v] = \Delta [sol_v]
    
    sol[0] = 2. * pi 
           * this->max_v_ * ( pi * std::cos(pi * x[0]) * std::cos(pi * x[0]) - pi * std::sin(pi * x[0]) * std::sin(pi * x[0]))
           * std::sin(2. * pi * x[1])
           - 2. * pi 
           * this->max_v_ * std::sin(pi * x[0]) * std::sin(pi * x[0]) 
           * 2. * pi * std::sin(2. * pi * x[1]);
    
    sol[1] = 2. * pi 
           * 2. * pi * this->max_v_ * std::sin(2. * pi * x[0])
           * std::sin(pi * x[1]) * std::sin(pi * x[1])
           - 2. * pi 
           * this->max_v_ * std::sin(2. * pi * x[0]) 
           * ( pi * std::cos(pi * x[1]) * std::cos(pi * x[1]) - pi * std::sin(pi * x[1]) * std::sin(pi * x[1]) );

    for (int d=0; d!=DIM+1; ++d)
    {
      sol[d] *= this->eval_time_factor(time);
    }
    
    return sol;
  }
  
  
  DataType max_v_;
  size_t nb_comp_;
  size_t fe_ind_; 
};
  
// Functor used to impose u(x) = c on the boundary.
struct VelocityDirichletBC
{
  VelocityDirichletBC(DataType time)
  {
    exact_sol_.set_fe_ind(0);
    time_ = time;
  }
  
  void evaluate(const mesh::Entity &face,
                const Vec< DIM, DataType> &pt_coord, 
                std::vector<DataType> &vals) const 
  {
    const int material_number = face.get_material_number();
    vals.clear();
    vals.resize(DIM, 0.);
    exact_sol_.evaluate(face, pt_coord, time_, vals);
  }
  
  Vec<DIM, DataType> dirichlet_val(const Entity& face, const Coord& pt) const 
  {
    std::vector<DataType> tmp_val;
    this->evaluate(face, pt, tmp_val);
        
    Vec<DIM, DataType> vals;
    for (int d=0; d!=DIM; ++d)
    {
      vals.set(d, tmp_val[d]);
    }
    return vals;
  }
  
  int get_if_type (const Entity& face) const
  {
    const int mat_num = face.get_material_number();
    if (mat_num > 0)
    {
      // dirichlet
      return 1;
    }
    // interior
    return 0;
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
  
  ExactSol exact_sol_;
  DataType time_;
};

// Functor used for the local assembly of the stiffness matrix and load vector.
class LocalFlowAssembler : private AssemblyAssistant< DIM, DataType > 
{
public:
  
  void set_parameters (DataType time, DataType theta, DataType dt, DataType nu, int solver_type)
  {
    this->time_ = time;
    this->nu_ = nu;
    this->dt_ = dt;
    this->theta_ = theta;
    this->solver_type_ = solver_type;
  }
  
  void set_newton_solution(VectorType const *newton_sol) 
  {
    prev_newton_sol_ = newton_sol;
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
   
    // evaluate last newton iterate at all quadrature points
    this->evaluate_last_newton_and_time_iterate();
    
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
      
      // previous Newton iterate
      // store velocity, velocity jacobian and pressure at current quad point
      Vec<DIM, DataType> vk;                //-> u_k
      Mat<DIM, DIM, DataType> Dvk;          // -> grad(u_k)
      DataType pk = this->sol_ns_[DIM][q];  // -> p_k
      
      for (int v=0; v!= DIM; ++v)
      {
        vk.set(v, this->sol_ns_[v][q]);
        for (int d=0; d!= DIM; ++d)
        {
          Dvk.set(v, d, this->grad_sol_ns_[v][q][d]);
        }
      }
  
      // previous time step
      Vec<DIM, DataType> vn;
      Mat<DIM, DIM, DataType> Dvn;
      for (int v = 0; v != DIM; ++v)
      {
        vn.set(v, this->sol_vel_ts_[v][q]);
        for (int d = 0; d!= DIM; ++d)
        {
          Dvn.set(v, d, this->grad_sol_vel_ts_[v][q][d]);
        }
      }
      
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
            // ************************************
            if (solver_type_ == 0)
            {
              // TODO EXERCISE A
              // Semi-implicit formulation:


              // END EXERCISE A
            }
            else if (solver_type_ == 1)
            {
              // TODO EXERCISE B
              // implicit formulation with Picard iteration


              // END EXERCISE B
            }
            else if (solver_type_ == 2)
            {
              // END EXERCISE C
              // implicit formulation with Newton method

 
              // END EXERCISE C
            }
          }
          // ******************************************
          
          // velocity - pressure part
          if (   this->first_dof_for_var(0) <= i && i < this->last_dof_for_var(DIM-1) 
              && this->first_dof_for_var(DIM) <= j && j < this->last_dof_for_var(DIM))
          {
            // - p * div(v)
            DataType l4 = - phiP_j * trace(DphiV_i);
            lm(i, j) += wq * dt_ * l4 * dJ;
          }
        
          // pressure - velocity part
          if (   this->first_dof_for_var(DIM) <= i && i < this->last_dof_for_var(DIM) 
              && this->first_dof_for_var(0) <= j && j < this->last_dof_for_var(DIM-1))
          {
            // div(u) * q
            DataType l5 = phiP_i * trace(DphiV_j);
            lm(i,j) += wq * l5 * dJ;
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
   
    // evaluate last newton iterate at all quadrature points
    this->evaluate_last_newton_and_time_iterate();
    
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
      
      // previous Newton iterate
      // store velocity, velocity jacobian and pressure at current quad point
      Vec<DIM, DataType> vk;                //-> u_k
      Mat<DIM, DIM, DataType> Dvk;          // -> grad(u_k)
      DataType pk = this->sol_ns_[DIM][q];  // -> p_k
      
      for (int v=0; v!= DIM; ++v)
      {
        vk.set(v, this->sol_ns_[v][q]);
        for (int d=0; d!= DIM; ++d)
        {
          Dvk.set(v, d, this->grad_sol_ns_[v][q][d]);
        }
      }
  
      // previous time step
      Vec<DIM, DataType> vn;
      Mat<DIM, DIM, DataType> Dvn;
      for (int v = 0; v != DIM; ++v)
      {
        vn.set(v, this->sol_vel_ts_[v][q]);
        for (int d = 0; d!= DIM; ++d)
        {
          Dvn.set(v, d, this->grad_sol_vel_ts_[v][q][d]);
        }
      }
      
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
          // **************************************
          if (solver_type_ == 0)
          {
            // TODO exercise A
            // Semi-implicit formulation:


                      
            // END EXERCISE A
          }
          else if (solver_type_ == 1)
          {
            // TODO exercise B
            // implicit formulation with Picard iteration


                  
            // END EXERCISE B
          }
          else if (solver_type_ == 2)
          {
            // TODO exercise C
            // implicit formulation with Newton method
          


          // END EXERCISE C
          }
          // ***************************
        }
        
        // mass equation
        if ( this->first_dof_for_var(DIM) <= i && i < this->last_dof_for_var(DIM) ) 
        {
          if (solver_type_ == 2)
          {
            lv[i] += wq * phiP_i * trace(Dvk) * dJ;
          }
        }
      } 
    }
  }

  Vec<DIM,DataType> force(Vec< DIM, DataType > pt, DataType time) 
  {
    Vec<DIM, DataType> f;
    std::vector< DataType > exact_val = this->exact_sol_.eval(pt, time);
    std::vector< DataType > exact_dt = this->exact_sol_.eval_dt(pt, time);
    std::vector< DataType > exact_laplace = this->exact_sol_.eval_laplace(pt, time);
    std::vector< Vec<DIM,DataType> > exact_grad = this->exact_sol_.eval_grad(pt, time);
    
    // TODO EXERCISE C
    for (int var=0; var < DIM; ++var)
    {
      f.set(var, exact_dt[var] - this->nu_ * exact_laplace[var] + exact_grad[DIM][var]);
      for (int d=0; d!= DIM; ++d)
      {
        f.add(var, exact_val[d] * exact_grad[var][d]);
      }
    }
    // END EXERCISE C
    return f;
  }
   
  // evaluate FE function corresponding to vector prev_newton_sol
  // at all quadrature points
  void evaluate_last_newton_and_time_iterate()
  {
    // compute velocity solution values of previous time step u_(n-1) at each quadrature point xq:
    // sol_ts_[v][q] = u_(n-1)_v (xq)
    // v = velocity component
    for (int v=0; v!=DIM+1; ++v)
    {
      sol_ns_[v].clear();
      grad_sol_ns_[v].clear();
      
      this->evaluate_fe_function(*prev_newton_sol_, v, sol_ns_[v]);
      this->evaluate_fe_function_gradients(*prev_newton_sol_, v, grad_sol_ns_[v]);
      
    }
    for (int v=0; v!= DIM; ++v)
    {
      this->sol_vel_ts_[v].clear();
      this->grad_sol_vel_ts_[v].clear();
      this->evaluate_fe_function(*prev_time_sol_, v, sol_vel_ts_[v]);
      this->evaluate_fe_function_gradients(*prev_time_sol_, v, grad_sol_vel_ts_[v]);
    }
  }
  
  ExactSol exact_sol_;
  int solver_type_;
  
  FunctionValues< DataType > sol_ns_[DIM+1]; // solution at previous newton step
  FunctionValues< Vec< DIM, DataType > > grad_sol_ns_[DIM+1]; // gradient of solution at previous newton step
  
  VectorType const * prev_newton_sol_;
    
  VectorType const * prev_time_sol_;
  FunctionValues< DataType > sol_vel_ts_[DIM];               // velocity solution at previous time step
  FunctionValues< Vec< DIM, DataType > > grad_sol_vel_ts_[DIM];  // gradient of velocity solution at previous time step
  
  DataType theta_;
  DataType dt_;
  DataType nu_;
  DataType time_;
};

class WkpErrorIntegrator : private AssemblyAssistant< DIM, DataType > 
{
public:
  
  WkpErrorIntegrator(int k, DataType p, VectorType &sol, ExactSol* exact_sol, DataType time) 
  : p_(p), sol_(sol), exact_sol_(exact_sol), k_(k), time_(time)
  {
  }

                  
  void operator()(const Element< DataType, DIM > &element,
                  const Quadrature< DataType > &quadrature, 
                  DataType &value) 
  {
    AssemblyAssistant< DIM, DataType >::initialize_for_element(element, quadrature, false);
    value = 0.;
    int nb_comp = this->exact_sol_->nb_comp();
    
    this->fe_sol_.clear();
    this->fe_sol_.resize(DIM+1);
    this->fe_grad_sol_.clear();
    this->fe_grad_sol_.resize(DIM+1);
    
    // Evaluate the computed solution at all quadrature points.
    for (int l=0; l<DIM+1; ++l)
    {
      this->evaluate_fe_function(this->sol_, l, this->fe_sol_[l]);
      this->evaluate_fe_function_gradients(this->sol_, l, this->fe_grad_sol_[l]);
    }
    
    const int num_q = this->num_quadrature_points();
    for (int q = 0; q < num_q; ++q) 
    {
      Vec<DIM, DataType> pt = this->x(q);

      const DataType wq = this->w(q);
      std::vector<DataType> exact_sol = this->exact_sol_->eval(pt, time_);
      std::vector<Vec<DIM,DataType> > exact_grad_sol = this->exact_sol_->eval_grad(pt, time_);
      
      // l2 error
      DataType diff_lp = 0.;
      
      // h1 error 
      DataType diff_wp = 0.;
      if (nb_comp == DIM)
      {
        // velocity error
        for (int v=0; v<nb_comp; ++v)
        {
          diff_lp += std::pow(std::abs(exact_sol[v] - this->fe_sol_[v][q]), this->p_);
          for (int d=0; d!= DIM; ++d)
          {
            diff_wp += std::pow(std::abs(exact_grad_sol[v][d] - this->fe_grad_sol_[v][q][d]), this->p_);
          }
        }
      }
      else if (nb_comp == 1)
      {
        // pressure error
        diff_lp += std::pow(std::abs(exact_sol[DIM] - this->fe_sol_[DIM][q]), this->p_);
        for (int d=0; d!= DIM; ++d)
        {
          diff_wp += std::pow(std::abs(exact_grad_sol[DIM][d] - this->fe_grad_sol_[DIM][q][d]), this->p_);
        }
      }
      
      value += wq * diff_lp * std::abs(this->detJ(q));
      if (k_==1)
      {
        value += wq * diff_wp * std::abs(this->detJ(q));
      }
    }
  }
  
  // coefficients of the computed solution
  const VectorType &sol_;
  
  // functor to evaluate exact solution
  ExactSol* exact_sol_;
  
  // vector with values of computed solution evaluated at each quadrature point
  std::vector<FunctionValues< DataType > > fe_sol_;
  std::vector<FunctionValues< Vec<DIM,DataType> > > fe_grad_sol_;
  
  DataType time_;
  DataType p_;
  int k_;
};

template < int DIM, class LAD >
struct PressureIntegral : private AssemblyAssistant< DIM, typename LAD::DataType > 
{
  typedef typename LAD::MatrixType OperatorType;
  typedef typename LAD::VectorType VectorType;
  typedef typename LAD::DataType DataType;

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
struct VolumeIntegral : private AssemblyAssistant< DIM, typename LAD::DataType > 
{
  typedef typename LAD::MatrixType OperatorType;
  typedef typename LAD::VectorType VectorType;
  typedef typename LAD::DataType DataType;

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
