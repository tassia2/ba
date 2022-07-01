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
  
  void evaluate(const Entity& cell, const Vec<DIM, DataType> & x, std::vector<DataType>& vals) const
  {
    assert (DIM >= 2);
    vals.clear();
    vals.resize (this->nb_comp(), 0.);
    
    std::vector<DataType> tmp_vals = this->eval (x);
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
  
  std::vector<DataType> eval(const Vec< DIM, DataType > &x) const 
  {
    std::vector<DataType> sol (DIM+1, 0.);
    
    // vel_x
    sol[0] = this->max_v_ * std::sin(pi * x[0]) * std::sin(pi * x[0]) * std::sin(2. * pi * x[1]);

    // vel_y
    sol[1] = - this->max_v_ * std::sin(2. * pi * x[0]) * std::sin(pi * x[1]) * std::sin(pi * x[1]);
    
    // pressure
    sol[DIM] = std::cos(2. * pi * x[1]);
              
    return sol;
  }

  std::vector< Vec<DIM, DataType> > eval_grad(const Vec< DIM, DataType > &x) const 
  {
    std::vector< Vec<DIM,DataType> > sol (DIM+1);
    // sol[v][d] = \partial_d [sol_v]
    
    
    // TODO EXERCISE B
    sol[0].set(0, 2. * pi * this->max_v_ * std::sin(pi * x[0]) * std::cos(pi * x[0])  * std::sin(2. * pi * x[1])  );
    sol[0].set(1, 2. * pi * this->max_v_ * std::sin(pi * x[0]) * std::sin(pi * x[0]) * std::cos(2. * pi * x[1])   );
    sol[1].set(0, - 2. * pi * this->max_v_ * std::cos(2. * pi * x[0]) * std::sin(pi * x[1]) * std::sin(pi * x[1]) );
    sol[1].set(1, - 2. * pi * this->max_v_ * std::sin(2. * pi * x[0])  * std::sin(pi * x[1]) * std::cos(pi * x[1]));
    sol[DIM].set(0, 0.);
    sol[DIM].set(1, -2. * pi * std::sin(2. * pi * x[1]));  
 
    // END EXERCISE B
    return sol;
  }
  
  std::vector<DataType> eval_laplace(const Vec< DIM, DataType > &x) const 
  {
    std::vector<DataType> sol (DIM+1, 0.);
    // sol[v] = \Delta [sol_v]
    
    // TODO EXERCISE B
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
    // END EXERCISE B
    return sol;
  }
  
  
  DataType max_v_;
  size_t nb_comp_;
  size_t fe_ind_; 
};
  
// Functor used to impose u(x) = c on the boundary.
struct VelocityDirichletBC
{
  VelocityDirichletBC()
  {
    exact_sol_.set_fe_ind(0);
  }
  
  void evaluate(const mesh::Entity &face,
                const Vec< DIM, DataType> &pt_coord, 
                std::vector<DataType> &vals) const 
  {
    const int material_number = face.get_material_number();
    vals.clear();
    vals.resize(DIM, 0.);
    exact_sol_.evaluate(face, pt_coord, vals);
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
};

// Functor used for the local assembly of the stiffness matrix and load vector.
class LocalStokesAssembler : private AssemblyAssistant< DIM, DataType > 
{
public:
  
  void set_parameters (DataType nu, int pressure_stab_type, DataType gamma_p)
  {
    this->nu_ = nu;
    this->pressure_stab_type_ = pressure_stab_type;
    this->gamma_p_ = gamma_p;
  }
  
  // compute local matrix 
  // [in]  element:    contains information about current cell
  // [in]  quadrature: quadrature rule to be used for approximating the integrals
  // [out] lm: contribution of the current cell to the global system matrix
  
  void operator()(const Element< DataType, DIM > &element,
                  const Quadrature< DataType > &quadrature, 
                  LocalMatrix &lm) 
  {
    const bool need_basis_hessians = true;
    
    // AssemblyAssistant sets up the local FE basis functions for the current cell
    AssemblyAssistant< DIM, DataType >::initialize_for_element(element, quadrature, need_basis_hessians);
   
    // diameter of cell
    const DataType hc = this->h();
    
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
          Mat<DIM, DIM, DataType> HphiV_j[DIM]; // HphiV_j[DIM][v] = hessian matrix of v-th basis function component
           
          for (size_t var = 0; var < DIM; ++var)
          {
            phiV_j.set(var, this->Phi(j, q, var));
            HphiV_j[var] = this->H_Phi(j, q, var);
            for (int d = 0; d != DIM; ++d)
            {
              DphiV_j.set(var, d, this->grad_Phi(j, q, var)[d]);
            }
          }
          
          
          Vec<DIM, DataType> LphiV_j;
          
          // EXERCISE A
          for (size_t var = 0; var < DIM; ++var)
          {
            for (int d = 0; d != DIM; ++d)
            {
              LphiV_j.add(var, HphiV_j[var](d,d));
            }
          }
          // END EXERCSIE A
          
          // get ansatz function values for pressure variable
          DataType phiP_j = this->Phi(j, q, DIM);
          Vec<DIM, DataType> DphiP_j = this->grad_Phi(j, q, DIM);

          // --------------------------------------------------------------------------
          // ----- start assembly of individual terms in variational formulation ------
          
          // begin with velocity - velocity part
          if (   this->first_dof_for_var(0) <= i && i < this->last_dof_for_var(DIM-1) 
              && this->first_dof_for_var(0) <= j && j < this->last_dof_for_var(DIM-1))
          {
            // nu * grad(u) : grad(v)
            DataType l0 = dot(DphiV_j, DphiV_i);
            lm(i, j) += wq * this->nu_ * l0 * dJ; 
          }
          
          // velocity - pressure part
          if (   this->first_dof_for_var(0) <= i && i < this->last_dof_for_var(DIM-1) 
              && this->first_dof_for_var(DIM) <= j && j < this->last_dof_for_var(DIM))
          {
            // - p * div(v)
            DataType l1 = - phiP_j * trace(DphiV_i);
            lm(i, j) += wq * l1 * dJ;
          }
        
          // pressure - velocity part
          if (   this->first_dof_for_var(DIM) <= i && i < this->last_dof_for_var(DIM) 
              && this->first_dof_for_var(0) <= j && j < this->last_dof_for_var(DIM-1))
          {
            // div(u) * q
            DataType l2 = phiP_i * trace(DphiV_j);
            lm(i,j) += wq * l2 * dJ;

            // TODO EXERCISE A
            if (pressure_stab_type_ == 3)
            {
              lm(i,j) -= wq * gamma_p_ * hc * hc * nu_ * dot(LphiV_j, DphiP_i) * dJ;
            }
            // END EXERCISE A
          }
          
          // pressure - pressure part
          if (   this->first_dof_for_var(DIM) <= i && i < this->last_dof_for_var(DIM) 
              && this->first_dof_for_var(DIM) <= j && j < this->last_dof_for_var(DIM))
          {
            // TODO EXERCISE A
            if (pressure_stab_type_ == 1)
            {
              lm(i,j) += wq * gamma_p_ * hc * hc * phiP_i * phiP_j * dJ;
            }
            else if (pressure_stab_type_ == 2)
            {
              lm(i,j) += wq * gamma_p_ * hc * hc * dot(DphiP_i, DphiP_j) * dJ;
            }
            else if (pressure_stab_type_ == 3)
            {
              lm(i,j) += wq * gamma_p_ * hc * hc * dot(DphiP_i, DphiP_j) * dJ;
            }
            // END EXERCISE A
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
    const bool need_basis_hessians = true;
    
    // AssemblyAssistant sets up the local FE basis functions for the current cell
    AssemblyAssistant< DIM, DataType >::initialize_for_element(element, quadrature, need_basis_hessians);
   
    // diameter of cell
    const DataType hc = this->h();
    
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

        // momentum equation
        if ( this->first_dof_for_var(0) <= i && i < this->last_dof_for_var(DIM-1) )
        {
          Vec<DIM, DataType> f = this->force(this->x(q));
          lv[i] += wq * dot(f, phiV_i) * dJ;
        }
        
        // mass equation
        if ( this->first_dof_for_var(DIM) <= i && i < this->last_dof_for_var(DIM) ) 
        {
          // TODO EXERCISE A        
          if (pressure_stab_type_ == 0)
          {
            lv[i] += wq * this->source(this->x(q)) * phiP_i * dJ; 
          }
          else if (pressure_stab_type_ == 1)
          {
            lv[i] += wq * this->source(this->x(q)) * phiP_i * dJ; 
          }
          else if (pressure_stab_type_ == 2)
          {
            lv[i] += wq * this->source(this->x(q)) * phiP_i * dJ; 
          }
          else if (pressure_stab_type_ == 3)
          {
            lv[i] += wq * (this->source(this->x(q)) * phiP_i + gamma_p_ * hc * hc * dot(DphiP_i, this->force(this->x(q)))) * dJ;
          }
          // END EXERCISE A 
        }
      } 
    }
  }
     
  Vec<DIM,DataType> force(Vec< DIM, DataType > pt) 
  {
    Vec<DIM, DataType> f;
    std::vector<DataType> exact_val = this->exact_sol_.eval(pt);
    std::vector< DataType > exact_laplace = this->exact_sol_.eval_laplace(pt);
    std::vector< Vec<DIM,DataType> > exact_grad = this->exact_sol_.eval_grad(pt);
    
    // TODO EXERCISE B
    for (int var=0; var < DIM; ++var)
    {
      f.set( var, - this->nu_ * exact_laplace[var] + exact_grad[DIM][var]);
    }

    // END EXERCISE B
    return f;
  }
  
  DataType source(Vec< DIM, DataType > pt) 
  {
    std::vector< Vec<DIM,DataType> > exact_grad = this->exact_sol_.eval_grad(pt);
    
    DataType res = 0.;
    
    // TODO EXERCISE B
    for (int d = 0; d < DIM; ++d)
    {
      res += exact_grad[d][d];
    }
    // END EXERCISE B
    return res;
  }
  
  DataType nu_;
  ExactSol exact_sol_;
  DataType gamma_p_;
  int pressure_stab_type_;
};

class WkpErrorIntegrator : private AssemblyAssistant< DIM, DataType > 
{
public:
  
  WkpErrorIntegrator(int k, DataType p, VectorType &sol, ExactSol* exact_sol) 
  : p_(p), sol_(sol), exact_sol_(exact_sol), k_(k)
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
      std::vector<DataType> exact_sol = this->exact_sol_->eval(pt);
      std::vector<Vec<DIM,DataType> > exact_grad_sol = this->exact_sol_->eval_grad(pt);
      
      // l2 error
      DataType diff_lp = 0.;
      
      // h1 error 
      DataType diff_wp = 0.;
      if (nb_comp == DIM)
      {
        // velocity error
        // TODO EXERCISE C
        for (int v=0; v<nb_comp; ++v)
        {
          diff_lp += std::pow(std::abs(exact_sol[v] - this->fe_sol_[v][q]), this->p_);
          for (int d=0; d!= DIM; ++d)
          {
            diff_wp += std::pow(std::abs(exact_grad_sol[v][d] - this->fe_grad_sol_[v][q][d]), this->p_);
          }
        }
        // END EXERCISE C
      }
      else if (nb_comp == 1)
      {
        // pressure error
        // TODO EXERCISE C
        diff_lp += std::pow(std::abs(exact_sol[DIM] - this->fe_sol_[DIM][q]), this->p_);
        for (int d=0; d!= DIM; ++d)
        {
          diff_wp += std::pow(std::abs(exact_grad_sol[DIM][d] - this->fe_grad_sol_[DIM][q][d]), this->p_);
        }
        // END EXERCISE C
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
