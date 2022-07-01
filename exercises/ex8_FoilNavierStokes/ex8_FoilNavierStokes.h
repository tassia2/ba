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
                      int inflow_mat_number,
                      int outflow_mat_number,
                      int slipX_mat_number,
                      int noslip_mat_number, 
                      int foil1_mat_number,
                      int foil2_mat_number,
                      DataType inflow_vel_x)
  : time_(time), 
  inflow_mat_num_ (inflow_mat_number),
  outflow_mat_num_ (outflow_mat_number),
  noslip_mat_num_ (noslip_mat_number),
  slipX_mat_num_ (slipX_mat_number),
  foil1_mat_num_ (foil1_mat_number),
  foil2_mat_num_ (foil2_mat_number), 
  inflow_vel_x_ (inflow_vel_x)
  {}
  
  void evaluate(const mesh::Entity &face,
                const Vec< DIM, DataType> &pt_coord, 
                std::vector<DataType> &vals) const 
  {
    const int material_number = face.get_material_number();
    vals.clear();
    

    DataType time_factor = 1.;
    if (  material_number == noslip_mat_num_
       || material_number == foil1_mat_num_
       || material_number == foil2_mat_num_)
    {
      vals.resize(DIM, 0.);
    }
    else if ( material_number == slipX_mat_num_ )
    {
      vals.resize(DIM, 0.);
      vals[0] = inflow_vel_x_;
    }
    else if ( material_number == inflow_mat_num_ )
    {
      vals.resize(DIM, 0.);
      vals[0] = inflow_vel_x_;
    }
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
  
  int noslip_mat_num_;
  int slipX_mat_num_;
  int inflow_mat_num_;
  int outflow_mat_num_;
  int foil1_mat_num_;
  int foil2_mat_num_;
  DataType inflow_vel_x_;
  DataType time_;
};

// Functor used for the local assembly of the stiffness matrix and load vector.
class LocalFlowAssembler : private AssemblyAssistant< DIM, DataType > 
{
public:
  
  void set_parameters (DataType theta, DataType dt, DataType nu, DataType fz)
  {
    this->nu_ = nu;
    this->f_ = fz;
    this->dt_ = dt;
    this->theta_ = theta;
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
            // TODO EXERCISE A            


            lm(i, j) += wq * 1. * dJ;      
            
            // END EXERCSIE A      
          }
          
          // velocity - pressure part
          if (   this->first_dof_for_var(0) <= i && i < this->last_dof_for_var(DIM-1) 
              && this->first_dof_for_var(DIM) <= j && j < this->last_dof_for_var(DIM))
          {
            // TODO EXERCISE A


            lm(i, j) += wq * 1. * dJ;  
            // END EXERCISE A
          }
        
          // pressure - velocity part
          if (   this->first_dof_for_var(DIM) <= i && i < this->last_dof_for_var(DIM) 
              && this->first_dof_for_var(0) <= j && j < this->last_dof_for_var(DIM-1))
          {
            // TODO EXERCISE A

            lm(i, j) += wq * 1. * dJ;  
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
          // TODO EXERCISE B


          lv[i] += wq * (1. ) * dJ;
          // END EXERCISE B
        }
        
        // mass equation
        if ( this->first_dof_for_var(DIM) <= i && i < this->last_dof_for_var(DIM) ) 
        {
          // TODO EXERCISE B


          lv[i] += wq * (1. ) * dJ;
          // END EXERCISE B 
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
  
  FunctionValues< DataType > sol_ns_[DIM+1]; // solution at previous newton step
  FunctionValues< Vec< DIM, DataType > > grad_sol_ns_[DIM+1]; // gradient of solution at previous newton step
  
  VectorType const * prev_newton_sol_;
    
  VectorType const * prev_time_sol_;
  FunctionValues< DataType > sol_vel_ts_[DIM];               // velocity solution at previous time step
  FunctionValues< Vec< DIM, DataType > > grad_sol_vel_ts_[DIM];  // gradient of velocity solution at previous time step
  
  DataType theta_;
  DataType dt_;
  DataType nu_;
  DataType f_;
};

template < int DIM, class LAD >
class ForceIntegral : private AssemblyAssistant< DIM, typename LAD::DataType > 
{
  typedef typename LAD::MatrixType OperatorType;
  typedef typename LAD::VectorType VectorType;
  typedef typename LAD::DataType DataType;

public:
  ForceIntegral(DataType mu, 
                const VectorType *sol,
                int force_direction,
                int obstacle_mat_num)
      : mu_(mu), force_dir_(force_direction), sol_(sol), obstacle_mat_num_ (obstacle_mat_num)
  {
  } 

  // compute local scalar contributions from boundary facets
  // [in]  element:    contains information about current cell
  // [in]  facet_number: local index of element facet
  // [in]  quadrature: quadrature rule to be used for approximating the integrals
  // [out] ls[facet_number]: contribution of the current facet to the global scalar
  void operator()(const Element< DataType, DIM > &element, 
                  int facet_number,
                  const Quadrature< DataType > &quadrature,
                  std::vector<DataType> &ls)
  {
    const bool need_basis_hessians = false;
    
    AssemblyAssistant< DIM, DataType >::initialize_for_facet(element,
                                                             quadrature, 
                                                             facet_number,
                                                             need_basis_hessians);
                                                             
    this->recompute_function_values();
    
    // get material number of current facet
    IncidentEntityIterator facet = element.get_cell().begin_incident(DIM - 1);
    for (int i = 0; i < facet_number; ++i, ++facet) {}
    
    const int material_number = facet->get_material_number();
    
    // number of quadrature points
    const int num_q = this->num_quadrature_points();
    
    // this variable stores the integral over the current facet
    DataType facet_integral = 0.;
    
    // loop over quadrature points
    for (int q = 0; q < num_q; ++q) 
    {
      // quadrature weight
      const DataType wq = this->w(q);
      
      // surface element of cell transformation
      const DataType dS = std::abs(this->ds(q)); 
      
      // surface normal
      // multiply by -1, 
      // because we want to have the outward normal w.r.t. the obstacle and not w.r.t. the compuational domain
      const Vec<DIM, DataType> nq = (-1.) * this->n(q);
      
      if (material_number != obstacle_mat_num_)
      {
        // consider only obstacle surface
        continue;
      }
       
      // force vector 
      // t = sigma * n
      // sigma = mu * (gradv + grad v ^T) - p I
      
      Vec<DIM, DataType> t;
      Mat<DIM, DIM, DataType> sigma;
      for (int v=0; v!= DIM; ++v)
      {
        for (int d=0; d!=DIM; ++d)
        {
          sigma.add(v,d, mu_ * (grad_vel_[v][q][d] + grad_vel_[d][q][v]));
        }
        sigma.add(v,v, -press_[q]);
      }
      sigma.VectorMult(nq, t);
      
      facet_integral += wq * t[force_dir_] * dS;
    }
    
    ls[facet_number] = facet_integral; 
  }

private:
  void recompute_function_values() 
  {
    for (int d = 0; d < DIM; ++d) 
    {
      vel_[d].clear();
      grad_vel_[d].clear();

      this->evaluate_fe_function(*sol_, d, vel_[d]);
      this->evaluate_fe_function_gradients(*sol_, d, grad_vel_[d]);
    }
    press_.clear();
    this->evaluate_fe_function(*sol_, DIM, press_);
  }

  DataType mu_;
  int obstacle_mat_num_, force_dir_;
  FunctionValues< DataType > vel_[DIM], press_;
  FunctionValues< Vec< DIM, DataType > > grad_vel_[DIM];
  const VectorType *sol_;
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
