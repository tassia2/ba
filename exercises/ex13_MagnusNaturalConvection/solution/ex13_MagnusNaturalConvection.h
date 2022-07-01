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
#include <math.h>
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

// Functor used to impose Dirichlet BC for velocity
// time: current time
// foil_mat_number:   material number of foil
// inflow_mat_number: material number of inflow boundary
// outflow_mat_number: material number of outflow boundary
// slipX_mat_number: material number of artificial boundary (with inflow conditions)
// inflow_vel_x: x-component of inflow velocity (y-comp = 0)
// inflow_ramp: inflow_vel(t=0) = 0, inflow_vel(t=t_inflow_ramp) = inflow_vel_x, for t \in (0,inflow_ramp): linear interpolation
//              same behavviour for revolutions_per_second
// rev_per_sec: revolutions per second of foil
// center_x: center of foil (x-coordinate)
// center_y: center of foil (y-coordinate) 
struct VelocityDirichletBC
{
  VelocityDirichletBC(DataType time,
                      int foil_mat_number,
                      int inflow_mat_number,
                      int outflow_mat_number,
                      int slipX_mat_number,
                      DataType inflow_vel_x,
                      DataType inflow_ramp,
                      DataType rev_per_sec,
                      DataType center_x,
                      DataType center_y)
  : time_(time), 
  inflow_mat_num_ (inflow_mat_number),
  outflow_mat_num_ (outflow_mat_number),
  slipX_mat_num_ (slipX_mat_number),
  foil_mat_num_ (foil_mat_number),
  inflow_vel_x_ (inflow_vel_x),
  inflow_ramp_ (inflow_ramp),
  center_x_ (center_x),
  center_y_ (center_y),
  rev_per_sec_ (rev_per_sec)
  {}
  
  // evaluate Dirichlet BC at given boundary facet and point lying on this facet 
  // HINT: vector vals must be empty for Neumann BC 
  void evaluate(const mesh::Entity &face,
                const Vec< DIM, DataType> &pt_coord, 
                std::vector<DataType> &vals) const 
  {
    // get material number of boundary facet
    const int material_number = face.get_material_number();
    vals.clear();

    // **********************************************
    // TODO 
    
    DataType time_factor = 1.;
    if (time_ < inflow_ramp_)
    {
      time_factor = time_ / inflow_ramp_;
    }
    
    if (  material_number == foil_mat_num_ )
    {
      DataType omega = rev_per_sec_ * 2. * M_PI;

      vals.resize(DIM, 0.);
      vals[0] = time_factor * omega /** norm*/ * (pt_coord[1] - center_y_);
      vals[1] = -time_factor * omega /** norm*/ * (pt_coord[0] - center_x_);
    }
    else if ( material_number == slipX_mat_num_ )
    {
      vals.resize(DIM, 0.);
      vals[0] = time_factor * inflow_vel_x_;
    }
    else if ( material_number == inflow_mat_num_ )
    {
      vals.resize(DIM, 0.);
      vals[0] = time_factor * inflow_vel_x_;
    }
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
  
  int slipX_mat_num_;
  int inflow_mat_num_;
  int outflow_mat_num_;
  int foil_mat_num_;
  DataType inflow_vel_x_;
  DataType time_;
  DataType inflow_ramp_;
  DataType center_x_;
  DataType center_y_;
  DataType rev_per_sec_;
};

// Functor used to impose Dirichlet BC for velocity
// time: current time
// foil_mat_number:   material number of foil
// inflow_mat_number: material number of inflow boundary
// outflow_mat_number: material number of outflow boundary
// slipX_mat_number: material number of artificial boundary (with inflow conditions)
// inflow_vel_x: x-component of inflow velocity (y-comp = 0)
// inflow_ramp: inflow_vel(t=0) = 0, inflow_vel(t=t_inflow_ramp) = inflow_vel_x, for t \in (0,inflow_ramp): linear interpolation
//              same behavviour for revolutions_per_second
// rev_per_sec: revolutions per second of foil
// center_x: center of foil (x-coordinate)
// center_y: center of foil (y-coordinate) 
struct TemperatureDirichletBC
{
  TemperatureDirichletBC(DataType time,
                         int foil_mat_number,
                         int inflow_mat_number,
                         int outflow_mat_number,
                         int slipX_mat_number,
                         DataType inflow_temp,
                         DataType foil_temp)
  : time_(time), 
  inflow_mat_num_ (inflow_mat_number),
  outflow_mat_num_ (outflow_mat_number),
  slipX_mat_num_ (slipX_mat_number),
  foil_mat_num_ (foil_mat_number),
  inflow_temp_ (inflow_temp),
  foil_temp_ (foil_temp)
  {}
  
  // evaluate Dirichlet BC at given boundary facet and point lying on this facet 
  // HINT: vector vals must be empty for Neumann BC 
  void evaluate(const mesh::Entity &face,
                const Vec< DIM, DataType> &pt_coord, 
                std::vector<DataType> &vals) const 
  {
    // get material number of boundary facet
    const int material_number = face.get_material_number();
    vals.clear();

    // **********************************************
    // TODO 
        
    if (  material_number == foil_mat_num_ )
    {
      vals.resize(1, 0.);
      vals[0] = foil_temp_;
    }
    else if ( material_number == inflow_mat_num_ )
    {
      vals.resize(1, 0.);
      vals[0] = inflow_temp_;
    }
    // *********************************************
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
  
  int slipX_mat_num_;
  int inflow_mat_num_;
  int outflow_mat_num_;
  int foil_mat_num_;
  DataType inflow_temp_;
  DataType time_;
  DataType foil_temp_;
};

// Functor used for the local assembly of the stiffness matrix and load vector.
class LocalFlowAssembler : private AssemblyAssistant< DIM, DataType > 
{
public:
  // dt: time step size
  // nu: viscosity 
  // Cv: SUPG parameter 
  // Cp: PSPG parameter
  // bdf_type: 1 (first order) or 2 (second order)
  // with_convection: false: Stokes, true: Navier-Stokes
  // use_stab: if true, SUPG and PSPG are used
  void set_parameters (DataType dt, 
                       DataType nu, 
                       DataType Cv, 
                       DataType Cp,
                       DataType kappa,
                       DataType gamma,
                       int bdf_type, 
                       bool with_convection, 
                       bool use_stab)
  {
    this->nu_ = nu;
    this->kappa_ = kappa;
    this->dt_ = dt;
    this->Cv_ = Cv;
    this->Cp_ = Cp;
    this->bdf_type_ = bdf_type;
    this->gamma_ = gamma;
    this->with_convection_ = with_convection;
    this->use_stab_ = use_stab;
    this->grav_.set(0, 0.);
    this->grav_.set(1, 0.);
    this->grav_.set(DIM-1, -1);
    // ***************************************************
    // TODO

    // set time stepping coefficients according to bdf_type
    // alpha_dt_i: coefficients for dt terms, 
    // alpha_conv_i: coefficients for convection terms,

    // i = 0 ~ t_n
    // i = 1 ~ t_(n-1)
    // i = 2 ~ t_(n-2)

    if (bdf_type == 1)
    {
      // ...
      this->alpha_dt_0_ = 1.;
      this->alpha_dt_1_ = -1.;
      this->alpha_dt_2_ = 0.;

      this->alpha_conv_1_ = 1.;
      this->alpha_conv_2_ = 0.;
    }
    else if (bdf_type == 2)
    {
      // ...
      this->alpha_dt_0_ = 1.5;
      this->alpha_dt_1_ = -2.;
      this->alpha_dt_2_ = 0.5;

      this->alpha_conv_1_ = 2.;
      this->alpha_conv_2_ = -1.;
    }
    if (!with_convection)
    {
      // ...
      this->alpha_conv_1_ = 0.;
      this->alpha_conv_2_ = 0.;
    }
    // ***************************************************
  }

  // set solution of previous time step t_(n-1)
  void set_previous_solution(VectorType const *prev_sol) 
  {
    prev_time_sol_ = prev_sol;
  }

  // set solution of previous previous time step t_(n-2)
  void set_previous_previous_solution(VectorType const *pprev_sol) 
  {
    prev_prev_time_sol_ = pprev_sol;
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
   
    // evaluate last time iterate at all quadrature points
    this->evaluate_last_time_iterate();
    
    // number of degrees of freedom on current cell
    const size_t num_dof = this->num_dofs_total();
    
    // number of quadrature points
    const int num_q = this->num_quadrature_points();
    
    // diameter of current cell 
    const DataType hK = this->h();

    // loop over quadrature points
    for (int q = 0; q < num_q; ++q) 
    {
      // quadrature weight
      const DataType wq = this->w(q);
      
      // volume element of cell transformation
      const DataType dJ = std::abs(this->detJ(q)); 

      // ********************************************************************
      // TODO
      
      DataType deltaK_v = 0.5 * Cv_ * hK;
      DataType deltaK_p = 0.5 * Cp_ * hK;
      if (nu_ >= hK) 
      {
        deltaK_v *= hK;
        deltaK_p *= hK;
      }

      // previous time step
      // store velocity, velocity jacobian and pressure at current quad point
      Vec<DIM, DataType> v_p;                //-> u_(n-1)
      Vec<DIM, DataType> v_pp;               //-> u_(n-2)

      for (int v=0; v!= DIM; ++v)
      {
        v_p.set(v, this->sol_vel_ts_[v][q]);
        v_pp.set(v, this->sol_vel_tts_[v][q]);
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

        // SUPG velocity test function
        Vec<DIM, DataType> supgV_i;
        for (int v=0; v!= DIM; ++v)
        {
          for (int d=0; d!= DIM; ++d)
          {
             supgV_i.add(v, alpha_conv_1_ * v_p[d] * DphiV_i(v,d) 
                         + alpha_conv_2_ * v_pp[d] * DphiV_i(v,d));
          }
        }

        // get test function values for pressure variable
        DataType phiP_i = this->Phi(i, q, DIM);
        Vec<DIM, DataType> DphiP_i = this->grad_Phi(i, q, DIM);

        // get test function values for temperature variable
        DataType phiT_i = this->Phi(i, q, DIM+1);
        Vec<DIM, DataType> DphiT_i = this->grad_Phi(i, q, DIM+1);
        
        // SUPG temperature test function
        DataType supgT_i = alpha_conv_1_ * dot(v_p, DphiT_i) + alpha_conv_2_ * dot(v_pp, DphiT_i);

        // loop over trrial DOFs <-> trial function u 
        for (int j = 0; j < num_dof; ++j) 
        {
          // get ansatz function values for velocity variables
          Vec<DIM, DataType> phiV_j;
          Vec<DIM, DataType> LphiV_j;
          Mat<DIM, DIM, DataType> DphiV_j;
         
          for (size_t var = 0; var < DIM; ++var)
          {
            phiV_j.set(var, this->Phi(j, q, var));
            LphiV_j.set(var, this->L_Phi(j, q, var));
            for (int d = 0; d != DIM; ++d)
            {
              DphiV_j.set(var, d, this->grad_Phi(j, q, var)[d]);
            }
          }
          // get ansatz function values for pressure variable
          DataType phiP_j = this->Phi(j, q, DIM);
          Vec<DIM, DataType> DphiP_j = this->grad_Phi(j, q, DIM);

          // get ansatz function values for temperature variable
          DataType phiT_j = this->Phi(j, q, DIM+1);
          Vec<DIM, DataType> DphiT_j = this->grad_Phi(j, q, DIM+1);


          // SUPG velocity trial function
          Vec<DIM, DataType> supgV_j;

          for (int v=0; v!= DIM; ++v)
          {
            for (int d=0; d!= DIM; ++d)
            {
               supgV_j.add(v, alpha_conv_1_ * v_p[d] * DphiV_j(v,d) 
                            + alpha_conv_2_ * v_pp[d] * DphiV_j(v,d));
            }
          }

          // SUPG temperature trial function
          DataType supgT_j = alpha_conv_1_ * dot(v_p, DphiT_j) + alpha_conv_2_ * dot(v_pp, DphiT_j);

          // --------------------------------------------------------------------------
          // ----- start assembly of individual terms in variational formulation ------
          
          // begin with velocity - velocity part
          if (   this->first_dof_for_var(0) <= i && i < this->last_dof_for_var(DIM-1) 
              && this->first_dof_for_var(0) <= j && j < this->last_dof_for_var(DIM-1))
          {
            // ........
            DataType l_dt = 0.;
            DataType l_nu = 0.;
            DataType l_conv_1 = 0.;
            DataType l_conv_2 = 0.;

            Vec<DIM, DataType> residual;

            // u * v
            l_dt = dot(phiV_j, phiV_i);
            
            // grad(u) : grad(v)
            l_nu = this->nu_ * dot(DphiV_j, DphiV_i);

            for (int v=0; v!= DIM; ++v)
            {
              for (int d=0; d!= DIM; ++d)
              {
                // u_p * grad(u)^T * v
                l_conv_1 += v_p[d] * DphiV_j(v,d) * phiV_i[v];
                
                // u_pp * grad(u)^T * v
                l_conv_2 += v_pp[d] * DphiV_j(v,d) * phiV_i[v];
              }
            }

            lm(i, j) += wq * (alpha_dt_0_ * l_dt / dt_
                              + l_nu 
                              + alpha_conv_1_ * l_conv_1 
                              + alpha_conv_2_ * l_conv_2 
                             ) * dJ;     

            if (this->use_stab_)
            {
              Vec<DIM, DataType> res = alpha_dt_0_ / dt_ * phiV_j 
                                      - this->nu_ * LphiV_j
                                      + supgV_j;
              
              //lm(i, j) += wq * deltaK_v * dot(res, supg_i) * dJ;
              lm(i, j) += wq * deltaK_v * dot(supgV_j, supgV_i) * dJ;
            } 
          }
          
          // velocity - pressure part
          if (   this->first_dof_for_var(0) <= i && i < this->last_dof_for_var(DIM-1) 
              && this->first_dof_for_var(DIM) <= j && j < this->last_dof_for_var(DIM))
          {
            // ........
            // - p * div(v)
            DataType l4 = - phiP_j * trace(DphiV_i);
            lm(i, j) += wq * l4 * dJ;

            if (this->use_stab_)
            {              
              //lm(i, j) += wq * deltaK_v * dot(DphiP_j, supg_i) * dJ;
            } 
          }
        
          // velocity - temperature part
          if (   this->first_dof_for_var(0) <= i && i < this->last_dof_for_var(DIM-1) 
              && this->first_dof_for_var(DIM+1) <= j && j < this->last_dof_for_var(DIM+1))
          {
            lm(i, j) += wq * gamma_ * phiT_j * dot(grav_, phiV_i) * dJ;
          }

          // pressure - velocity part
          if (   this->first_dof_for_var(DIM) <= i && i < this->last_dof_for_var(DIM) 
              && this->first_dof_for_var(0) <= j && j < this->last_dof_for_var(DIM-1))
          {
            // ........
            // div(u) * q
            DataType l5 = phiP_i * trace(DphiV_j);
            lm(i,j) += wq * l5 * dJ;

            if (this->use_stab_)
            {
              Vec<DIM, DataType> res = alpha_dt_0_ / dt_ * phiV_j 
                                      - this->nu_ * LphiV_j
                                      + supgV_j;
              
              //lm(i, j) += wq * deltaK_p * dot(res, DphiP_i) * dJ;
            } 
          }
          // pressure - pressure part
          if (   this->first_dof_for_var(DIM) <= i && i < this->last_dof_for_var(DIM) 
              && this->first_dof_for_var(DIM) <= j && j < this->last_dof_for_var(DIM))
          {
            // ........
            if (this->use_stab_)
            {
              lm(i, j) += wq * deltaK_p * dot(DphiP_j, DphiP_i) * dJ;
            } 
          }

          // temperature - temperature part
          if (   this->first_dof_for_var(DIM+1) <= i && i < this->last_dof_for_var(DIM+1) 
              && this->first_dof_for_var(DIM+1) <= j && j < this->last_dof_for_var(DIM+1))
          {
            DataType l_dt = 0.;
            DataType l_kappa = 0.;
            DataType l_conv_1 = 0.;
            DataType l_conv_2 = 0.;

            // u * v
            l_dt = phiT_j * phiT_i;
            
            // grad(u) : grad(v)
            l_kappa = this->kappa_ * dot(DphiT_j, DphiT_i);

            // u_p * grad(u)^T * v
            l_conv_1 += dot(v_p, DphiT_j) * phiT_i;
            l_conv_2 += dot(v_pp, DphiT_j) * phiT_i;

            lm(i, j) += wq * (alpha_dt_0_ * l_dt / dt_
                              + l_kappa 
                              + alpha_conv_1_ * l_conv_1 
                              + alpha_conv_2_ * l_conv_2 
                             ) * dJ;     

            if (this->use_stab_)
            {              
              lm(i, j) += wq * deltaK_v * supgT_j * supgT_i * dJ;
            } 
          }
        }
      }
      // ***********************************************************************************
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
    this->evaluate_last_time_iterate();
    
    // number of degrees of freedom on current cell
    const size_t num_dof = this->num_dofs_total();
    
    // number of quadrature points
    const int num_q = this->num_quadrature_points();

    // diameter of current cell 
    const DataType hK = this->h();

    // loop over quadrature points
    for (int q = 0; q < num_q; ++q) 
    {
      // quadrature weight
      const DataType wq = w(q);
      
      // volume element of cell transformation
      const DataType dJ = std::abs(this->detJ(q)); 

      // ************************************************************
      // TODO  
      
      DataType deltaK_v = 0.5 * Cv_ * hK;
      DataType deltaK_p = 0.5 * Cp_ * hK;
      if (nu_ >= hK) 
      {
        deltaK_v *= hK;
        deltaK_p *= hK;
      }

      // previous time step
      // store velocity, velocity jacobian and pressure at current quad point
      Vec<DIM, DataType> v_p;                //-> u_(n-1)
      Vec<DIM, DataType> v_pp;               //-> u_(n-2)

      for (int v=0; v!= DIM; ++v)
      {
        v_p.set(v, this->sol_vel_ts_[v][q]);
        v_pp.set(v, this->sol_vel_tts_[v][q]);
      }
      
      DataType t_p = this->sol_temp_ts_[q];
      DataType t_pp = this->sol_temp_tts_[q];
      
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

        // SUPG test function
        Vec<DIM, DataType> supg_i;
        for (int v=0; v!= DIM; ++v)
        {
          for (int d=0; d!= DIM; ++d)
          {
             supg_i.add(v, alpha_conv_1_ * v_p[d] * DphiV_i(v,d) 
                         + alpha_conv_2_ * v_pp[d] * DphiV_i(v,d));
          }
        }

        // get test function values for pressure variable
        DataType phiP_i = this->Phi(i, q, DIM);
        Vec<DIM, DataType> DphiP_i = this->grad_Phi(i, q, DIM);

        DataType phiT_i = this->Phi(i, q, DIM+1);
        Vec<DIM, DataType> DphiT_i = this->grad_Phi(i, q, DIM+1);

        // momentum equation
        if ( this->first_dof_for_var(0) <= i && i < this->last_dof_for_var(DIM-1) )
        {
          // ........
          DataType l_dt_1 = dot(v_p, phiV_i);
          DataType l_dt_2 = dot(v_pp, phiV_i);

          lv[i] += wq * (- alpha_dt_1_ * l_dt_1 / dt_ 
                         - alpha_dt_2_ * l_dt_2 / dt_    
                        ) * dJ;

          if (this->use_stab_)
          {
              Vec<DIM, DataType> res = - alpha_dt_1_ / dt_ * v_p 
                                       - alpha_dt_2_ / dt_ * v_pp;
              
              //lv[i] += wq * deltaK_v * dot(res, supg_i) * dJ;
          } 
        }
        
        // mass equation
        if ( this->first_dof_for_var(DIM) <= i && i < this->last_dof_for_var(DIM) ) 
        {
          // ........
          if (this->use_stab_)
          {
              Vec<DIM, DataType> res = - alpha_dt_1_ / dt_ * v_p 
                                       - alpha_dt_2_ / dt_ * v_pp;
              
              //lv[i] += wq * deltaK_p * dot(res, DphiP_i) * dJ;
          } 
        }

        // heat equation
        if ( this->first_dof_for_var(DIM+1) <= i && i < this->last_dof_for_var(DIM+1) ) 
        {
          // ........
          DataType l_dt_1 = t_p * phiT_i;
          DataType l_dt_2 = t_pp * phiT_i;

          lv[i] += wq * (- alpha_dt_1_ * l_dt_1 / dt_ 
                         - alpha_dt_2_ * l_dt_2 / dt_    
                        ) * dJ;
        }
      }
      // ********************************************************************** 
    }
  }
   
  // evaluate FE function corresponding to vector prev_newton_sol and prev_prev_time_sol at all quadrature points xq
  void evaluate_last_time_iterate()
  {    
    for (int v=0; v!= DIM; ++v)
    {
      this->sol_vel_ts_[v].clear();
      this->sol_vel_tts_[v].clear();
      
      this->evaluate_fe_function(*prev_time_sol_, v, sol_vel_ts_[v]);
      this->evaluate_fe_function(*prev_prev_time_sol_, v, sol_vel_tts_[v]);
    }
    this->sol_temp_ts_.clear();
    this->sol_temp_tts_.clear();
      
    this->evaluate_fe_function(*prev_time_sol_, DIM+1, sol_temp_ts_);
    this->evaluate_fe_function(*prev_prev_time_sol_, DIM+1, sol_temp_tts_);
  }
      
  // FE coefficient vector for previous time steps
  VectorType const * prev_time_sol_;
  VectorType const * prev_prev_time_sol_;

  // velocity solutions at previous time steps, evaluated at quadratur points
  // sol_ts_[v][q] = u_(n-1)_v (xq)
  // sol_tts_[v][q] = u_(n-2)_v (xq)
  // v = velocity component
  FunctionValues< DataType > sol_vel_ts_[DIM];                   
  FunctionValues< DataType > sol_vel_tts_[DIM];                   
  FunctionValues< DataType > sol_temp_ts_;
  FunctionValues< DataType > sol_temp_tts_;

  // physical and control parameters
  bool with_convection_;
  bool use_stab_;
  int bdf_type_;
  DataType dt_;
  DataType nu_;
  DataType kappa_;
  DataType gamma_;
  DataType Cv_;
  DataType Cp_;
  Vec<DIM, DataType> grav_;

  // coefficients for time stepping method
  DataType alpha_dt_0_;
  DataType alpha_dt_1_;
  DataType alpha_dt_2_;
  DataType alpha_conv_1_;
  DataType alpha_conv_2_;

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
                int foil_mat_num)
      : mu_(mu), force_dir_(force_direction), sol_(sol), foil_mat_num_ (foil_mat_num)
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
    // evaluate FE function on all quadrature points                       
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
      
      // *******************************************************
      // TODO

      // HINT: ignore all facets, that are not located on the foil
      // HINT: compute contribution to facet_integral of current facet 
      // -> facet_integral += .....
 
      if (material_number != foil_mat_num_)
      {
        // consider only foil surface
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

      // ***************************************************************
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
  int foil_mat_num_, force_dir_;
  FunctionValues< DataType > vel_[DIM], press_;
  FunctionValues< Vec< DIM, DataType > > grad_vel_[DIM];
  const VectorType *sol_;
};

template < int DIM, class LAD >
class PostProcessingIntegral : private AssemblyAssistant< DIM, typename LAD::DataType > 
{
  typedef typename LAD::MatrixType OperatorType;
  typedef typename LAD::VectorType VectorType;
  typedef typename LAD::DataType DataType;

public:
  PostProcessingIntegral(const VectorType &sol, int type, DataType nu)
      : sol_(sol),
      type_(type),
      nu_(nu)
  {}

  void operator()(const Element< DataType, DIM > &element,
                  const Quadrature< DataType > &quadrature,
                  DataType &result) 
  {
    AssemblyAssistant< DIM, DataType >::initialize_for_element(element, quadrature, false);
    for (int v=0; v!= DIM; ++v)
    {
      vel_[v].clear();
      grad_vel_[v].clear();
      this->evaluate_fe_function(sol_, v, vel_[v]);
      this->evaluate_fe_function_gradients(sol_, v, grad_vel_[v]);
    }
    
    const int num_q = this->num_quadrature_points();

    // loop over quadrature points
    for (int q = 0; q < num_q; ++q) 
    {
      const DataType wq = this->w(q);
      DataType dJ = std::abs(this->detJ(q));

      // *********************************************************************+
      // TODO

      DataType div = 0.;
      DataType diss = 0.;
      DataType kin = 0.;
      for (int v=0; v!= DIM; ++v)
      {
        kin += vel_[v][q] * vel_[v][q];
        div += grad_vel_[v][q][v];
        for (int d=0; d!= DIM; ++d)
        {
          diss += grad_vel_[d][q][v] * grad_vel_[d][q][v];
        }
      }

      if (type_ == 0)
      {
        // kinetic energy, result += ....
        result += wq * kin * dJ;
      }
      else if (type_ == 1)
      {
        // dissipation energy, result += ....
        result += wq * nu_ * diss * dJ;
      }
      else if (type_ == 2)
      {
        // squared divergence, result += ....
        result += wq * div * div * dJ;
      }
      
      // *********************************************************************+
    }
  }
  
  const VectorType &sol_;
  int type_;
  DataType nu_;
  FunctionValues< DataType > vel_[DIM];
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
