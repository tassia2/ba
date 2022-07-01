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

#ifndef HIFLOW_EX12_H
#define HIFLOW_EX12_H

#include <mpi.h>

#include <exception>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "hiflow.h"

/// \author Staffan Ronnas, Philipp Gerstner

using namespace hiflow;
using namespace hiflow::doffem;
using namespace hiflow::la;
using namespace hiflow::mesh;

#define DIM 2

// debugging example with constructed right hand side and exact solution,
// see explication of ChannelBenchmark::eval_exact_sol() in source code
// EXACTSOL == 1 else == 0
#define EXACTSOL 0

// use vector valued elements for velocity, e.g. BDM

// Linear Algebra type renaming.
typedef LADescriptorCoupledD LAD;
typedef LAD::VectorType CVector;
typedef LAD::MatrixType CMatrix;
typedef LAD::DataType Scalar;

typedef Vec<DIM, Scalar> Coord;

// Exception types

struct UnexpectedParameterValue : public std::runtime_error 
{

  UnexpectedParameterValue(const std::string &name, const std::string &value)
      : std::runtime_error("Unexpected value '" + value + "' for parameter " +
                           name) {}
};

/// The default quadrature selection chooses a quadrature rule that is accurate
/// to 2 * max(fe_degree).

struct QuadratureSelection 
{
  QuadratureSelection(int order) : order_(order) {}

  void operator()(const Element< Scalar, DIM > &elem,
                  Quadrature< Scalar > &quadrature) {
    const RefCellType cell_id = elem.ref_cell()->type();

    switch (cell_id) 
    {
    case RefCellType::TRI_STD:
      quadrature.set_cell_tag(CellType::TRIANGLE);
      quadrature.set_quadrature_by_order("GaussTriangle", order_);
      break;
    case RefCellType::QUAD_STD:
      quadrature.set_cell_tag(CellType::QUADRILATERAL);
      quadrature.set_quadrature_by_order("GaussQuadrilateral", order_);
      break;
    case RefCellType::TET_STD:
      quadrature.set_cell_tag(CellType::TETRAHEDRON);
      quadrature.set_quadrature_by_order("GaussTetrahedron", order_);
      break;
    case RefCellType::HEX_STD:
      quadrature.set_cell_tag(CellType::HEXAHEDRON);
      quadrature.set_quadrature_by_order("GaussHexahedron", order_);
      break;
    default:
      assert(false);
    };
  }

  int order_;
};

class CylinderDescriptor : public BoundaryDomainDescriptor<Scalar, DIM> 
{
public:
  CylinderDescriptor(Scalar radius, Coord center)
      : radius_(radius), center_(center) {
    assert(center_.size() > 1);
  }

  Coordinate eval_func(const Coord &p,
                       MaterialNumber mat_num) const {
    if (mat_num == 13) {
      const Scalar x = p[0] - center_[0];
      const Scalar y = p[1] - center_[1];
      return radius_ * radius_ - x * x - y * y;
    } else {
      return 0.;
    }
  }

  Vec<DIM, Scalar> eval_grad(const Coord &p,
                                      MaterialNumber mat_num) const {
    Vec<DIM, Scalar> grad;
    if (mat_num == 13) {
      const Scalar x = p[0] - center_[0];
      const Scalar y = p[1] - center_[1];
      grad.set(0, -2 * x);
      grad.set(1, -2 * y);
    }
    return grad; 
  }

private:
  const Scalar radius_;
  const Coord center_;
};

struct ChannelFlowBC2d 
{
  // Parameters:
  // var - variable
  // H - channel height, W - channel width
  // Um - maximum inflow
  // inflow_bdy - material number of inflow boundary
  // outflow_bdy - material number of outflow boundary

  ChannelFlowBC2d(int var, Scalar H, Scalar Um, int inflow_bdy, int outflow_bdy)
      : var_(var), H_(H), Um_(Um), inflow_bdy_(inflow_bdy),
        outflow_bdy_(outflow_bdy) 
  {
    assert(var_ == 0 || var_ == 1);
    assert(DIM == 2);
  }

  void evaluate(const Entity &face, const Coord& pt, std::vector< Scalar >& values) const 
  {
    values.clear();
    
    const int material_num = face.get_material_number();

    const bool outflow = (material_num == outflow_bdy_);
    const bool inflow = (material_num == inflow_bdy_);
        
    if (!outflow) 
    {
      // All boundaries except outflow have Dirichlet BC.
      
      values.resize(1,0. );

      if (inflow) 
      {
        if (var_ == 0) 
        { // x-component
          values[0] = 4. * Um_ * pt[1] * (H_ - pt[1]) / (H_ * H_);
        } 
        else if (var_ == 1) 
        { // y-components
          values[0] = 0.;
        } 
      } 
      else 
      {
        // not inflow: u = 0
        values[0] = 0.;
      }
    }
  }
  
  const int var_;
  const Scalar H_;  // size in y- direction, respectively.
  const Scalar Um_; // max inflow velocity
  const int inflow_bdy_, outflow_bdy_;
};

struct ChannelFlowBC3d {
  // Parameters:
  // var - variable
  // H - channel height, W - channel width
  // Um - maximum inflow
  // inflow_bdy - material number of inflow boundary
  // outflow_bdy - material number of outflow boundary

  ChannelFlowBC3d(int var, Scalar W, Scalar H, Scalar Um, int inflow_bdy,
                  int outflow_bdy)
      : var_(var), W_(W), H_(H), Um_(Um), inflow_bdy_(inflow_bdy),
        outflow_bdy_(outflow_bdy) {

    assert(var_ == 0 || var_ == 1 || var_ == 2);
    assert(DIM == 3);
  }

  void evaluate(const Entity &face, const Coord& pt, std::vector< Scalar >& values) const 
  {
    const int material_num = face.get_material_number();

    const bool outflow = (material_num == outflow_bdy_);
    const bool inflow = (material_num == inflow_bdy_);
        
    if (!outflow) 
    {
      // All boundaries except outflow have Dirichlet BC.
      values.clear();
      values.resize(1,0. );

      if (inflow) 
      {
        if (var_ == 0) 
        { // x-component
            values[0] = 16. * Um_ * pt[1] * pt[2] * (W_ - pt[1]) *
                        (H_ - pt[2]) / (W_ * W_ * H_ * H_);
        } 
        else if (var_ == 1 || var_ == 2) 
        { // y- and z-components
            values[0] = 0.;
        }
      } 
      else 
      {
        // not inflow: u = 0
        values[0] = 0.;
      }
    }
  }
  
  const int var_;
  const Scalar W_, H_; // size in y- and z- direction, respectively.
  const Scalar Um_;    // max inflow velocity
  const int inflow_bdy_, outflow_bdy_;
};

//////////////// InstationaryAssembler ////////////////////////////////

class InstationaryFlowAssembler
    : private AssemblyAssistant< DIM, Scalar > 
    {
public:
  InstationaryFlowAssembler(Scalar nu, Scalar rho)
      : nu_(nu), inv_rho_(1. / rho) {}

  void set_newton_solution(const CVector *newton_sol) 
  {
    prev_newton_sol_ = newton_sol;
  }

  void set_time_solution(const CVector *time_sol) 
  { 
    prev_time_sol_ = time_sol; 
  }

  void set_time_stepping_weights(Scalar alpha0, Scalar alpha1, Scalar alpha2, Scalar alpha3) 
  {
    alpha0_ = alpha0;
    alpha1_ = alpha1;
    alpha2_ = alpha2;
    alpha3_ = alpha3;
  }

  void operator()(const Element< Scalar, DIM > &element,
                  const Quadrature< Scalar > &quadrature, LocalMatrix &lm) 
  {
    AssemblyAssistant< DIM, Scalar >::initialize_for_element(element, quadrature, false);

    // recompute previous solution values
    for (int d = 0; d < DIM; ++d) 
    {
      vel_ns_[d].clear();
      grad_vel_ns_[d].clear();

      this->evaluate_fe_function(*prev_newton_sol_, d, vel_ns_[d]);
      this->evaluate_fe_function_gradients(*prev_newton_sol_, d, grad_vel_ns_[d]);
    }

    const int num_q = num_quadrature_points();

    // loop q
    std::vector<Scalar> phi_i (DIM+1, 0.);
    std::vector< Vec<DIM, Scalar> > grad_phi_i (DIM+1);
    std::vector<Scalar> phi_j (DIM+1, 0.);
    std::vector< Vec<DIM, Scalar> > grad_phi_j (DIM+1);

    for (int q = 0; q < num_q; ++q) 
    {
      const Scalar wq = w(q);
      const Scalar dJ = std::abs(detJ(q));

      // Vector with velocity of previous newton point
      Vec< DIM, Scalar > vel_ns;
      for (int var = 0; var < DIM; ++var) 
      {
        vel_ns.set(var, vel_ns_[var][q]);
      }

      // loop over test functions
      for (int i = 0; i < num_dofs_total(); ++i) 
      {    
        const bool active_test_v = (this->first_dof_for_var(0) <= i 
                                    && i < this->last_dof_for_var(DIM-1));
      
        const bool active_test_p = (this->first_dof_for_var(DIM) <= i 
                                    && i < this->last_dof_for_var(DIM));
                                  
        // get ansatz function values 
        for (size_t var = 0; var < DIM+1; ++var)
        {
          phi_i[var] = this->Phi(i, q, var);
          grad_phi_i[var] = this->grad_Phi(i, q, var);
        }
  
        // precompute divergence of test function
        Scalar div_i = 0.;
        for (size_t var = 0; var < DIM; ++var)
        {
          div_i += grad_phi_i[var][var];
        }
            
        // loop over trial functions
        for (int j = 0; j < num_dofs_total(); ++j) 
        {
          const bool active_trial_v = (this->first_dof_for_var(0) <= j 
                                        && j < this->last_dof_for_var(DIM-1));
      
          const bool active_trial_p = (this->first_dof_for_var(DIM) <= j 
                                        && j < this->last_dof_for_var(DIM));
                                    
          // get ansatz function values 
          for (size_t var = 0; var < DIM+1; ++var)
          {
            phi_j[var] = this->Phi(j, q, var);
            grad_phi_j[var] = this->grad_Phi(j, q, var);
          }

          // ----- start assembly of individual terms in variational formulation ------
          
          // begin with momentum - velocity part
          // the following if condition is inserted only for perrformance reasons:
          // avoid unncessary processing of zero values
          if (   active_test_v && active_trial_v )
          {
            // mass tern: a0(u,v) = \int { u * v }
            // laplace term : a1(u,v) = \int {\grad(u) : \grad(v)}
            // convective term: a2(u,v) = \int { (vel_ns*\grad{u})*v }
            
            Scalar laplace = 0.;
            Scalar mass = 0.;
            Scalar convection = 0.;
            for (size_t var = 0; var < DIM; ++var)
            {
              mass += phi_j[var] * phi_i[var];
              laplace += dot(grad_phi_j[var], grad_phi_i[var]);
              convection += dot(vel_ns, grad_phi_j[var]) * phi_i[var];
            }
            mass *= alpha0_;
            laplace *= nu_ * alpha1_;
            convection *= alpha1_;

            // reaction term: a3(u,v) = \int { (u\grad{u_ns}*v }
            Scalar reaction = 0.;
            for (size_t var_i = 0; var_i < DIM; ++var_i)
            {
              for (size_t var_j = 0; var_j < DIM; ++var_j)
              {
                reaction += phi_j[var_j] * grad_vel_ns_[var_i][q][var_j] * phi_i[var_i]; 
              }
            }
            reaction *= alpha1_;
            lm(i, j) += wq * (mass + laplace + convection + reaction) * dJ; 
          }
          
          // momentum - pressure part
          if (   active_test_v && active_trial_p ) 
          {
            const int p_var = DIM;
          
            // b(p, v) = - \int{p div{v}}
            lm(i, j) -= wq * alpha2_ * inv_rho_ * phi_j[p_var] * div_i * dJ;
          }
        
          // continuity - velocity part
          if (   active_test_p && active_trial_v ) 
          {
            const int p_var = DIM;
            
            // bT(u, q) = \int{q div(u)}
            Scalar div_j = 0.;
            for (size_t var = 0; var < DIM; ++var)
            {
              div_j += grad_phi_j[var][var];
            }
            lm(i,j) += wq * inv_rho_ * phi_i[p_var] * div_j * dJ; 
          }
        }
      }
    }
  }

  void operator()(const Element< Scalar, DIM > &element,
                  const Quadrature< Scalar > &quadrature, LocalVector &lv) 
  {
    AssemblyAssistant< DIM, Scalar >::initialize_for_element(element, quadrature, false);
    // recompute previous solution values
    for (int d = 0; d < DIM; ++d) 
    {
      vel_ns_[d].clear();
      vel_ts_[d].clear();
      grad_vel_ns_[d].clear();
      grad_vel_ts_[d].clear();
      evaluate_fe_function(*prev_newton_sol_, d, vel_ns_[d]);
      evaluate_fe_function_gradients(*prev_newton_sol_, d, grad_vel_ns_[d]);

      evaluate_fe_function(*prev_time_sol_, d, vel_ts_[d]);
      evaluate_fe_function_gradients(*prev_time_sol_, d, grad_vel_ts_[d]);
    }

    // recompute pressure
    p_ns_.clear();
    evaluate_fe_function(*prev_newton_sol_, DIM, p_ns_);

    // indices j -> trial variable, i -> test variable
    // basis functions \phi -> velocity components, \eta -> pressure

    const int num_q = num_quadrature_points();
    std::vector<Scalar> phi_i (DIM+1, 0.);
    std::vector< Vec<DIM, Scalar> > grad_phi_i (DIM+1);
    std::vector<Scalar> phi_j (DIM+1, 0.);
    std::vector< Vec<DIM, Scalar> > grad_phi_j (DIM+1);

    // loop quadrature points
    for (int q = 0; q < num_q; ++q) 
    {
      const Scalar wq = w(q);
      const Scalar dJ = std::abs(detJ(q));

      // get previous newton and time step solutions in vector form
      Vec< DIM, Scalar > vel_ts, vel_ns;
      for (int var = 0; var < DIM; ++var) 
      {
        vel_ns.set(var, vel_ns_[var][q]);
        vel_ts.set(var, vel_ts_[var][q]);
      }

      // loop over test functions
      for (int i = 0; i < num_dofs_total(); ++i) 
      {    
        const bool active_test_v = (this->first_dof_for_var(0) <= i 
                                    && i < this->last_dof_for_var(DIM-1));
      
        const bool active_test_p = (this->first_dof_for_var(DIM) <= i 
                                    && i < this->last_dof_for_var(DIM));
                                    
        // get ansatz function values 
        for (size_t var = 0; var < DIM+1; ++var)
        {
          phi_i[var] = this->Phi(i, q, var);
          grad_phi_i[var] = this->grad_Phi(i, q, var);
        }
        
        // Residual without incompressibility
        if (active_test_v)
        {
          // l0(v) = \int {dot(vel_ns - vel_ts, v)}
          // l1(v) = \nu * \int{(\alpha1\grad{vel_ns} + \alpha3\grad{vel_ts}) * \grad{v}}
          // l2(v) = \int{ (\alpha1*dot(vel_ns, \grad{\vel_ns} + \alpha3*dot(vel_ts, \grad{\vel_ts}) * v}
          // l3(v) = -\alpha2/rho*\int{p_ns * div(v)}
          
          Scalar l0 = 0.;
          Scalar l1 = 0.;
          Scalar l2 = 0.;
          Scalar l3 = 0.;
          Scalar l4 = 0.;
          for (size_t var = 0; var < DIM; ++var)
          {
            l0 += alpha0_ * (vel_ns[var] - vel_ts[var]) * phi_i[var];
            l1 += nu_ * dot((alpha1_ * grad_vel_ns_[var][q]) 
                          + (alpha3_ * grad_vel_ts_[var][q]), grad_phi_i[var]);
            l2 += (alpha1_ * dot(grad_vel_ns_[var][q], vel_ns) 
                 + alpha3_ * dot(grad_vel_ts_[var][q], vel_ts)) * phi_i[var];
            l3 -= alpha2_ * inv_rho_ * p_ns_[q] * grad_phi_i[var][var];
          }
          lv[i] += wq * (l0 + l1 + l2 + l3 + l4) * dJ;
        }
      
        // Incompressibility term
        if (active_test_p)
        {
          const int p_var = DIM;
          Scalar div_u_k = 0.;
          for (int d = 0; d < DIM; ++d) 
          {
            div_u_k += grad_vel_ns_[d][q][d];
          }
          
          // l4(\eta_i) = 1/rho*\int{\eta_i * div(vel_ns)}
          lv[i] += wq * inv_rho_ * div_u_k * phi_i[p_var] * dJ; 
        }
      }
    }
  }

private:
  const CVector *prev_time_sol_;
  const CVector *prev_newton_sol_;

  Scalar alpha0_, alpha1_, alpha2_, alpha3_;

  Scalar nu_, inv_rho_;
  FunctionValues< Scalar > vel_ns_[DIM]; // velocity at previous newton step
  FunctionValues< Scalar > vel_ts_[DIM]; // velocity at previous timestep
  FunctionValues< Scalar > p_ns_; // pressure at previous newton step
  FunctionValues< Vec< DIM, Scalar > > grad_vel_ns_[DIM]; // gradient of velocity at previous newton step
  FunctionValues< Vec< DIM, Scalar > > grad_vel_ts_[DIM]; // gradient of velocity at previous timestep
};

// Functor used for the local evaluation of the square of the L2-norm
// of a set of variables on each element.

class L2NormIntegratorPp : private AssemblyAssistant< DIM, Scalar > 
{
public:
  L2NormIntegratorPp(CVector &pp_sol,
                     const std::vector< int > &vars)
      : pp_sol_(pp_sol), vars_(vars) {}

  void operator()(const Element< Scalar, DIM > &element,
                  const Quadrature< Scalar > &quadrature, Scalar &value) 
  {
    AssemblyAssistant< DIM, Scalar >::initialize_for_element(element, quadrature, false);

    const int num_q = num_quadrature_points();
    for (int v = 0, end_v = vars_.size(); v != end_v; ++v) 
    {
      const int var = vars_[v];
      evaluate_fe_function(pp_sol_, var, approx_sol_);

      for (int q = 0; q < num_q; ++q) 
      {
        const Scalar wq = w(q);
        const Scalar dJ = std::abs(detJ(q));

        value += wq * approx_sol_[q] * approx_sol_[q] * dJ;
      }
    }
  }

private:
  // coefficients of the computed solution
  CVector &pp_sol_;
  // vector with values of computed solution evaluated at each quadrature point
  FunctionValues< Scalar > approx_sol_;
  // variables for which to compute the norm
  std::vector< int > vars_;
};

// Functor used for the local evaluation of the square of the H1-seminorm
// of a set of variables on each element.

class H1semiNormIntegratorPp : private AssemblyAssistant< DIM, Scalar > 
{
public:
  H1semiNormIntegratorPp(CVector &pp_sol,
                         const std::vector< int > &vars)
      : pp_sol_(pp_sol), vars_(vars) {}

  void operator()(const Element< Scalar, DIM > &element,
                  const Quadrature< Scalar > &quadrature, Scalar &value) 
  {
    AssemblyAssistant< DIM, Scalar >::initialize_for_element(element, quadrature, false);

    const int num_q = num_quadrature_points();
    for (int v = 0, end_v = vars_.size(); v != end_v; ++v) 
    {
      const int var = vars_[v];
      evaluate_fe_function_gradients(pp_sol_, var, approx_sol_grad_);

      for (int q = 0; q < num_q; ++q) 
      {
        const Scalar wq = w(q);
        const Scalar dJ = std::abs(detJ(q));

        value += wq * dot(approx_sol_grad_[q], approx_sol_grad_[q]) * dJ;
      }
    }
  }

private:
  // coefficients of the computed solution
  CVector &pp_sol_;
  // vector with values of computed solution evaluated at each quadrature point
  FunctionValues< Vec< DIM, Scalar > > approx_sol_grad_;
  // variables for which to compute the norm
  std::vector< int > vars_;
};

// Functor used for the local evaluation of the square of the L2-norm of the
// difference between the solution of last end penultimate timestep.

class InstationaryL2ErrorIntegrator : private AssemblyAssistant< DIM, Scalar > 
{
public:
  InstationaryL2ErrorIntegrator(const LAD::VectorType &coeff,
                                const LAD::VectorType &coeff_penult)
      : coeff_(coeff), coeff_penult_(coeff_penult) {}

  void operator()(const Element< Scalar, DIM > &element,
                  const Quadrature< Scalar > &quadrature, Scalar &value)
  {
    AssemblyAssistant< DIM, Scalar >::initialize_for_element(element, quadrature, false);

    const int num_q = num_quadrature_points();
    for (int var = 0; var < DIM; ++var) 
    {
      evaluate_fe_function(coeff_, var, approx_sol_);
      evaluate_fe_function(coeff_penult_, var, approx_sol_penult_);
      for (int q = 0; q < num_q; ++q) 
      {
        const Scalar wq = w(q);
        const Scalar dJ = std::abs(detJ(q));
        const Scalar delta = approx_sol_penult_[q] - approx_sol_[q];
        value += wq * delta * delta * dJ;
      }
    }
  }

private:
  // coefficients of soluition of last timestep
  const LAD::VectorType &coeff_;
  // coefficients of solution of penultimate timestep
  const LAD::VectorType &coeff_penult_;
  // vector with values of computed solution evaluated at each quadrature point
  // for last and penultimate timestep
  FunctionValues< Scalar > approx_sol_, approx_sol_penult_;
};

// Functor used for the local evaluation of the square of the L2-norm of the
// solution in one timestep on each element.

class InstationaryL2Integrator : private AssemblyAssistant< DIM, Scalar > 
{
public:
  InstationaryL2Integrator(const LAD::VectorType &coeff) : coeff_(coeff) {}

  void operator()(const Element< Scalar, DIM > &element,
                  const Quadrature< Scalar > &quadrature, Scalar &value) 
  {
    AssemblyAssistant< DIM, Scalar >::initialize_for_element(element, quadrature, false);

    const int num_q = num_quadrature_points();
    for (int var = 0; var < DIM; ++var) 
    {
      evaluate_fe_function(coeff_, var, approx_sol_);
      for (int q = 0; q < num_q; ++q) {
        const Scalar wq = w(q);
        const Scalar dJ = std::abs(detJ(q));
        value += wq * approx_sol_[q] * approx_sol_[q] * dJ;
      }
    }
  }

private:
  // coefficients of soluition of a timestep
  const LAD::VectorType &coeff_;
  // vector with values of computed solution evaluated at each quadrature point
  // for a timestep
  FunctionValues< Scalar > approx_sol_;
};

#endif
