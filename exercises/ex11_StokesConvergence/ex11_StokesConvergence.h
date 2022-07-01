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
    
    /*
    if (DIM == 3)
    {
      sol[0] *= std::sin(2. * pi * x[2]);
      sol[1] *= std::sin(2. * pi * x[2]);
    
      sol[2] = this->max_v_ * std::sin(pi * x[2]) * std::sin(pi * x[2]) * std::sin(2. * pi * x[1]);
    }
    */
    
    // pressure
    sol[DIM] = std::cos(2. * pi * x[1]);
              
    return sol;
  }

  std::vector< Vec<DIM, DataType> > eval_grad(const Vec< DIM, DataType > &x) const 
  {
    std::vector< Vec<DIM,DataType> > sol (DIM+1);
    // sol[v][d] = \partial_d [sol_v]
    
    
    // TODO EXERCISE B
    sol[0].set(0, 2. * pi * this->max_v_ * std::sin(pi * x[0]) * std::cos(pi * x[0])  * std::sin(2. * pi * x[1]));
    sol[0].set(1, 2. * pi * this->max_v_ * std::sin(pi * x[0]) * std::sin(pi * x[0]) * std::cos(2. * pi * x[1]) );
    sol[1].set(0, - 2. * pi * this->max_v_ * std::cos(2. * pi * x[0]) * std::sin(pi * x[1]) * std::sin(pi * x[1]));
    sol[1].set(1, - 2. * pi * this->max_v_ * std::sin(2. * pi * x[0]) * std::sin(pi * x[1]) * std::cos(pi * x[1]));
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
    if (mat_num >= 11)
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
};

// Functor used for the local assembly of the stiffness matrix and load vector.
class LocalStokesAssembler : private DGAssemblyAssistant< DIM, DataType > 
{
public:
  
  void set_parameters (DataType nu, VelocityDirichletBC* bc ,
                       int pressure_stab_type, DataType gamma_p,
                       DataType sigma_u, DataType sigma_p, DataType gamma,
                       bool do_sip_u, bool do_press_stab, bool do_press_flux,
                       DataType* hmax)
  {
    this->nu_ = nu;
    this->bc_u_ = bc;
    this->pressure_stab_type_ = pressure_stab_type;
    this->gamma_p_ = gamma_p;
    sigma_u_ = sigma_u;
    sigma_p_ = sigma_p;
    gamma_ = gamma;
    do_sip_u_ = do_sip_u;
    do_press_stab_ = do_press_stab;
    do_press_flux_ = do_press_flux;
    h_max_ = hmax;
  }
  
  // compute local matrix 
  // [in]  element:    contains information about current cell
  // [in]  quadrature: quadrature rule to be used for approximating the integrals
  // [out] lm: contribution of the current cell to the global system matrix
  
  void operator()(const Element< DataType, DIM > &element,
                  const Quadrature< DataType > &quadrature, 
                  LocalMatrix &lm) 
  {
    const bool need_basis_hessians = (this->pressure_stab_type_ == 3);
    
    // AssemblyAssistant sets up the local FE basis functions for the current cell
    this->initialize_for_element(element, quadrature, need_basis_hessians);
   
    // diameter of cell
    const DataType hc = this->master().h();
    if (hc > *h_max_)
    {
      *h_max_ = hc;
      //std::cout << h_max_ << std::endl;
    }
    // number of degrees of freedom on current cell
    const size_t num_dof = this->num_dofs_total();
    
    // number of quadrature points
    const int num_q = this->num_quadrature_points();
    
    // loop over quadrature points
    for (int q = 0; q < num_q; ++q) 
    {
      // quadrature weight
      const DataType wq = this->w(q);
      
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
            if (need_basis_hessians)
            {
              HphiV_j[var] = this->H_Phi(j, q, var);
            }
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
    const bool need_basis_hessians = (this->pressure_stab_type_ == 3);
    
    // AssemblyAssistant sets up the local FE basis functions for the current cell
    this->initialize_for_element(element, quadrature, need_basis_hessians);
   
    // diameter of cell
    const DataType hc = this->master().h();
    
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
     
  void operator()(const Element< DataType, DIM > &master_elem,
                    const Element< DataType, DIM > &slave_elem,
                    const Quadrature< DataType > &master_quad,
                    const Quadrature< DataType > &slave_quad, 
                    int master_facet_number,
                    int slave_facet_number, 
                    InterfaceSide left_if_side,
                    InterfaceSide right_if_side,
                    int slave_index, int num_slaves,
                    LocalMatrix &lm) 
  {
    this->initialize_for_interface(master_elem, slave_elem, 
                                   master_quad, slave_quad,
                                   master_facet_number, slave_facet_number,
                                   left_if_side, right_if_side,
                                   false);
    const size_t pvar = DIM;
    const size_t nvar = DIM+1;
    
    // get type of interface:
    // 0: interior
    // 1: dirichlet
    // 2: neumann
    // 3: robin
    int if_type_u = 0;
    bool is_boundary = (right_if_side == InterfaceSide::BOUNDARY);
    this->set_facet(master_elem, master_facet_number);
      
    if (is_boundary)
    {
      if (this->bc_u_ == nullptr)
      {
        // Assume homogenous Neumann BC
        // in parallel: need to distinguish between "inner boundary" (between subdomains) and "real boundary"
        assert (false);
        if_type_u = 2;
      }
      else
      {
        if_type_u = this->bc_u_->get_if_type (this->facet_);
      }
    }
    // TODO: korrekt?
    const DataType h = std::pow(std::abs(this->master().detJ(0)), 1. / DataType(DIM));

    const int num_q = this->num_quadrature_points();
    for (int q = 0; q < num_q; ++q) 
    {
      // integration weight
      const DataType w_dS = this->w(q) * std::abs(this->ds(q));
      
      // facet normal
      const Vec<DIM, DataType>& n_f = this->n_F(q);
      
      // jump and average signums
      DataType s_jump_i = 1.;
      DataType s_jump_j = 1.;
      DataType s_average = 1.;
      
      if (if_type_u == 0)
      {
        // interior
        s_jump_i = dot(this->test().n(q), n_f);
        s_jump_j = dot(this->trial().n(q), n_f);
        s_average = 0.5;
      }
      
      DataType s_int = 0.;
      DataType s_dir = 0.;
      DataType s_neu = 0.;
      
      if (if_type_u == 0)
        s_int = 1.;
      if (if_type_u == 1)
        s_dir = 1.;
      if (if_type_u == 2)
        s_neu = 1.;
      
      // penalty parameters in SIP formulation
      DataType c_penalty_u = this->sigma_u_ / h;
          
      // loop over test functions
      for (int i = 0; i < this->test().num_dofs_total(); ++i) 
      {    
        Vec< DIM, DataType> phiV_i;
        DataType phiP_i = 0.;
        Vec< DIM, DataType> DphiP_i;
        Mat< DIM, DIM, DataType > DphiV_i;
            
        // get ansatz function values 
        for (size_t var = 0; var < DIM; ++var)
        {
          phiV_i.set(var, this->test().Phi(i, q, var));
          for (int d=0; d!= DIM; ++d)
          {
            DphiV_i.set(var, d, this->test().grad_Phi(i, q, var)[d]);
          }
        }
        phiP_i = this->test().Phi(i, q, pvar);
        DphiP_i = this->test().grad_Phi(i, q, pvar);
               
        // loop over trial functions
        for (int j = 0; j < this->trial().num_dofs_total(); ++j) 
        {
          // get ansatz function values 
          Vec< DIM, DataType> phiV_j;
          DataType phiP_j = 0.;
          Vec< DIM, DataType> DphiP_j;
          Mat< DIM, DIM, DataType > DphiV_j;
      
          for (size_t var = 0; var < DIM; ++var)
          {
            phiV_j.set(var, this->trial().Phi(j, q, var));
            for (int d=0; d!= DIM; ++d)
            {
              DphiV_j.set(var, d, this->trial().grad_Phi(j, q, var)[d]);
            }
          }
          phiP_j = this->trial().Phi(j, q, pvar);
          DphiP_j = this->trial().grad_Phi(j, q, pvar);

          // --------------------------------
          // ----- momentum - velocity ------
          if (    this->test().first_dof_for_var(0) <= i && i < this->test().last_dof_for_var(pvar-1) 
              && this->trial().first_dof_for_var(0) <= j && j < this->trial().last_dof_for_var(pvar-1))
          {
            if (this->do_sip_u_)
            {
              // SIP : - ave{grad_u} * n_f * jump{v}
              //       - ave{grad_v} * n_f * jump{u)
              //       + sigma / h * jump{u} * jump{v}          
                lm(i,j) += w_dS  
                         * nu_
                         * (c_penalty_u 
                              * (s_int + s_dir)
                              * s_jump_i
                              * s_jump_j
                              * dot (phiV_i, phiV_j)
                            - (s_int + s_dir)
                              * s_jump_i
                              * s_average
                              * inner(phiV_i, DphiV_j, n_f)  
                            - (s_int + s_dir)
                              * s_jump_j
                              * s_average
                              * inner(phiV_j, DphiV_i, n_f) );                        
            }
          }
          
          // --------------------------------
          // ----- momentum - pressure ------
          if (    this->test().first_dof_for_var(0) <= i && i < this->test().last_dof_for_var(pvar-1) 
              && this->trial().first_dof_for_var(pvar) <= j && j < this->trial().last_dof_for_var(pvar))
          {
            if (this->do_press_flux_)
            {
              lm(i,j) += w_dS * (s_int + s_dir) * s_jump_i * dot(phiV_i, n_f) * s_average * phiP_j; 
            }
          }
          
          // ----------------------------------
          // ----- continuity - velocity ------
          if (    this->test().first_dof_for_var(pvar) <= i && i < this->test().last_dof_for_var(pvar) 
              && this->trial().first_dof_for_var(0) <= j && j < this->trial().last_dof_for_var(pvar-1))
          {
            if (this->do_press_flux_)
            {
              lm(i,j) -= w_dS * (s_int + s_dir) * s_jump_j * dot(phiV_j, n_f) * s_average * phiP_i; 
            }
          }
          
          // ----------------------------------
          // ----- continuity - pressure ------
          if (    this->test().first_dof_for_var(pvar) <= i && i < this->test().last_dof_for_var(pvar) 
              && this->trial().first_dof_for_var(pvar) <= j && j < this->trial().last_dof_for_var(pvar))
          {
            if (this->do_press_stab_)
            {
              lm(i,j) += w_dS * h * s_int * s_jump_i * s_jump_j * phiP_i * phiP_j; 
            }
          }
        }
      }
    }
  }

  void operator()(const Element< DataType, DIM > &master_elem,
                  const Element< DataType, DIM > &slave_elem,
                  const Quadrature< DataType > &master_quad,
                  const Quadrature< DataType > &slave_quad, 
                  int master_facet_number,
                  int slave_facet_number, 
                  InterfaceSide right_if_side, 
                  int slave_index, int num_slaves,
                  LocalVector &lv) 
  {
    this->initialize_for_interface(master_elem, slave_elem, 
                                   master_quad, slave_quad,
                                   master_facet_number, slave_facet_number,
                                   InterfaceSide::NONE, right_if_side,
                                   false);
                   
    const size_t pvar = DIM;
    const size_t nvar = DIM+1;
    
    // get type of interface:
    // 0: interior
    // 1: dirichlet
    // 2: neumann
    // 3: robin
    int if_type_u = 0;
    bool is_boundary = (right_if_side == InterfaceSide::BOUNDARY);
    this->set_facet(master_elem, master_facet_number);

    if (is_boundary)
    {
      if (this->bc_u_ == nullptr)
      {
        // Assume homogenous Neumann BC
        assert (false);
        if_type_u = 2;
      }
      else
      {
        if_type_u = this->bc_u_->get_if_type (this->facet_);
      }
    }

    DataType s_int = 0.;
    DataType s_dir = 0.;
    DataType s_neu = 0.;
    
    if (if_type_u == 0)
      s_int = 1.;
    if (if_type_u == 1)
      s_dir = 1.;
    if (if_type_u == 2)
      s_neu = 1.;
        
    const DataType h = std::pow(std::abs(this->master().detJ(0)), 1. / DataType(DIM));

    const int num_q = this->num_quadrature_points();
    for (int q = 0; q < num_q; ++q) 
    {
      // integration weight
      const DataType w_dS = this->w(q) * std::abs(this->ds(q));
      
      // facet normal
      const Vec<DIM, DataType>& n_f = this->n_F(q);
            
      // jump and average signums
      DataType s_jump_i = 1.;
      DataType s_average = 1.;
      
      if (if_type_u == 0)
      {
        // interior
        s_jump_i = dot(this->test().n(q), n_f);
        s_average = 0.5;
      }
      
      // penalty parameters in SIP formulation
      DataType c_penalty_u = this->sigma_u_ / h;
      
      // dirichlet bc values
      Vec<DIM, DataType> dirichlet_u = this->bc_u_->dirichlet_val(this->facet_, this->x(q));
          
      // loop over test functions
      for (int i = 0; i < this->test().num_dofs_total(); ++i) 
      {    
        Vec< DIM, DataType> phiV_i;
        DataType phiP_i = 0.;
        Vec< DIM, DataType> DphiP_i;
        Mat< DIM, DIM, DataType > DphiV_i;
            
        // get ansatz function values 
        for (size_t var = 0; var < DIM; ++var)
        {
          phiV_i.set(var, this->test().Phi(i, q, var));
          for (int d=0; d!= DIM; ++d)
          {
            DphiV_i.set(var, d, this->test().grad_Phi(i, q, var)[d]);
          }
        }
        phiP_i = this->test().Phi(i, q, pvar);
        DphiP_i = this->test().grad_Phi(i, q, pvar);
        
        // ---------------------------------
        // ----- momentum - residual  ------
        if ( this->test().first_dof_for_var(0) <= i && i < this->test().last_dof_for_var(pvar-1))
        {
          if (this->do_sip_u_)
          {
            // SIP : - ave{grad_u} * n_f * jump{v}
            //       - ave{grad_v} * n_f * jump{u)
            //       + sigma / h * jump{u} * jump{v}
            //       + robin * u * v
            lv[i] += w_dS * nu_ 
                   * (s_dir
                      * c_penalty_u 
                      * s_jump_i
                      * dot (phiV_i, dirichlet_u)
                    - s_dir 
                      * s_average
                      * inner(dirichlet_u, DphiV_i, n_f) );
          }
          if (this->do_press_flux_)
          {
          
          }
        }
        
        // ---------------------------
        // continuity residual -------
        if ( this->test().first_dof_for_var(pvar) <= i && i < this->test().last_dof_for_var(pvar))
        {
          if (this->do_press_flux_)
          {
          
          }
          if (this->do_press_stab_)
          {
          }
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
      f.set(var, - this->nu_ * exact_laplace[var] + exact_grad[DIM][var]);
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
   
  void set_facet (const Element<DataType, DIM> & elem, 
                  const int facet_number)
  {
    const Entity& cell = elem.get_cell();
    assert (facet_number < cell.num_incident_entities(DIM-1) );
    
    IncidentEntityIterator facet_it = cell.begin_incident(DIM-1);
    for (size_t l=0; l<facet_number; ++l)
    {
      facet_it++;
    }
    this->facet_ = *(facet_it);
  }
  
  Vec<DIM, DataType> n_F (int q)
  {
    return this->master().n(q);
  }
  
  Entity facet_;
  
  DataType nu_;
  ExactSol exact_sol_;
  DataType gamma_p_;
  int pressure_stab_type_;
  
  DataType sigma_u_, sigma_p_, gamma_;
  bool do_sip_u_;
  bool do_press_stab_, do_press_flux_;
  
  VelocityDirichletBC* bc_u_;
  DataType* h_max_;
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
