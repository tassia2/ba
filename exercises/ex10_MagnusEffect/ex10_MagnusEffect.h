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
                       int bdf_type, 
                       bool with_convection, 
                       bool use_stab)
  {
    this->nu_ = nu;
    this->dt_ = dt;
    this->Cv_ = Cv;
    this->Cp_ = Cp;
    this->bdf_type_ = bdf_type;
    this->with_convection_ = with_convection;
    this->use_stab_ = use_stab;
    
    // ***************************************************
    // TODO

    // set time stepping coefficients according to bdf_type
    // alpha_dt_i: coefficients for dt terms, 
    // alpha_conv_i: coefficients for convection terms,

    // i = 0 ~ t_n
    // i = 1 ~ t_(n-1)
    // i = 2 ~ t_(n-2)


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
      

          // --------------------------------------------------------------------------
          // ----- start assembly of individual terms in variational formulation ------
          
          // begin with velocity - velocity part
          // if (   this->first_dof_for_var(0) <= i && i < this->last_dof_for_var(DIM-1) 
          //     && this->first_dof_for_var(0) <= j && j < this->last_dof_for_var(DIM-1))
          {
            // ........

          }
          
          // velocity - pressure part
          // if (   this->first_dof_for_var(0) <= i && i < this->last_dof_for_var(DIM-1) 
          //     && this->first_dof_for_var(DIM) <= j && j < this->last_dof_for_var(DIM))
          {
            // ........

          }
        
          // pressure - velocity part
          //if (   this->first_dof_for_var(DIM) <= i && i < this->last_dof_for_var(DIM) 
          //    && this->first_dof_for_var(0) <= j && j < this->last_dof_for_var(DIM-1))
          {
            // ........

          }
          // pressure - pressure part
          //if (   this->first_dof_for_var(DIM) <= i && i < this->last_dof_for_var(DIM) 
          //    && this->first_dof_for_var(DIM) <= j && j < this->last_dof_for_var(DIM))
          {
            // ........
 
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
      

        // momentum equation
        //if ( this->first_dof_for_var(0) <= i && i < this->last_dof_for_var(DIM-1) )
        {
          // ........

        }
        
        // mass equation
        //if ( this->first_dof_for_var(DIM) <= i && i < this->last_dof_for_var(DIM) ) 
        {
          // ........

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
  
  // physical and control parameters
  bool with_convection_;
  bool use_stab_;
  int bdf_type_;
  DataType dt_;
  DataType nu_;
  DataType Cv_;
  DataType Cp_;

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
      
      // facet_integral += .....
 


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

      if (type_ == 0)
      {
        // kinetic energy
        // result += 
      }
      else if (type_ == 1)
      {
        // dissipation energy
        // result += 
      }
      else if (type_ == 2)
      {
        // squared divergence
        // result += 
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
