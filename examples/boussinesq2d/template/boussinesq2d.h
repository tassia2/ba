#ifndef HIFLOW_BOUSSINESQ2D_H
#define HIFLOW_BOUSSINESQ2D_H

#include <mpi.h>

#include <exception>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "hiflow.h"


using namespace hiflow;
using namespace hiflow::doffem;
using namespace hiflow::la;
using namespace hiflow::mesh;

#define DIM 2
// debugging example with constructed right hand side and exact solution,
// see explication of ChannelBenchmark::eval_exact_sol() in source code
// EXACTSOL == 1 else == 0
#define EXACTSOL 0

// Linear Algebra type renaming.
typedef LADescriptorCoupledD LAD;
typedef LAD::VectorType CVector;
typedef LAD::MatrixType CMatrix;
typedef LAD::DataType Scalar;

// typedef Vec<DIM, double> Coord;
typedef Vec<DIM, Scalar> Coord;

// Exception types

struct UnexpectedParameterValue : public std::runtime_error {

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

struct BoussinesqBC2d {
    // Parameters:
    // var - variable
    // hT_ - hot Temperature, cT_ - cold Temperature
    // left_-, right_-, top_-, bottom_bdy - material number of x boundary
      BoussinesqBC2d(int var, double hT, double cT, int left_bdy, int right_bdy, int top_bdy, int bottom_bdy)
      : var_(var), hT_(hT), cT_(cT), left_bdy_(left_bdy), right_bdy_(right_bdy), top_bdy_(top_bdy), 
      bottom_bdy_(bottom_bdy) {
    assert(var_ == 0 || var_ == 1 || var_ == DIM-1 ||var_ == DIM+1);
    assert(DIM == 2);
    }

  void evaluate(const Entity &face, const Coord& pt, std::vector< Scalar >& values) const  
  {
    values.clear();

    const int material_num = face.get_material_number();

    const bool right = (material_num == right_bdy_);
    const bool left = (material_num == left_bdy_);
    const bool top = (material_num == top_bdy_);
    const bool bottom = (material_num == bottom_bdy_);

    if (left)
    {
        if (var_ == 0 || var_ == 1) 
        {
            values.resize(1, 0.);
        }
        else if (var_ == DIM+1) 
        {
          // Temperatur fällt an x-Koordinate linear ab
          values.resize(1, hT_);
          // initial conditions  values[i] = hT_ - pt[0] * (hT_ - cT_) / coords_on_face.size();
        }
    }
    else if (right)
    {
        if (var_ == 0 || var_ == 1) 
        {
            values.resize(1, 0.);
        }
        else if (var_ == DIM+1) 
        {
            values.resize(1, cT_);
        }
    }

    // if right values[i] = cT_

    else if (top || bottom) // !(!right) sondern if (top_bdy)
    {
        if (var_ == 0 || var_ == 1) 
        {
            values.resize(1, 0.);
        }
    }
  }

  const int var_, left_bdy_, right_bdy_, top_bdy_, bottom_bdy_;
  const double hT_, cT_;
};

struct InitialBoussinesq {

  InitialBoussinesq(int var, double hT, double cT)
      : var_(var), hT_(hT), cT_(cT)
  {
    assert(var_ == 0 || var_ == 1 || var_ == DIM ||var_ == DIM+1);
    assert(DIM == 2);
  }

  size_t nb_comp() const 
  {
    return 1;
  }

  size_t nb_func() const 
  {
    return 1;  
  }
  
  size_t iv2ind(size_t j, size_t v) const 
  {
    return v;
  }
  
  size_t weight_size() const 
  {
    return this->nb_comp() * this->nb_func();
  }

  void evaluate(const Entity &entity, const Coord& pt, std::vector< Scalar >& values) const  
  {
    values.clear();
    values.resize(1, 0.);

    if (var_ == DIM +1)
    {
      values[0] = hT_ - (pt[0] * (hT_ - cT_));
    }
  }

  // const double hT_, cT_;
  const int var_;
  const double hT_, cT_;
};

class BoussinesqAssembler
    : private AssemblyAssistant< DIM, double > {
public:
  BoussinesqAssembler(double rho, double Pr, double Ra)
      : rho_(rho), Pr_(Pr), Ra_(Ra), inv_sq_Ra_( 1 / sqrt(Ra)) {}

  void set_newton_solution(const CVector *newton_sol) {
    prev_newton_sol_ = newton_sol;
  }

  void set_time_solution(const CVector *time_sol) { prev_time_sol_ = time_sol; }

  void set_time_stepping_weights(double alpha1, double alpha2, double alpha3) {
    alpha1_ = alpha1;
    alpha2_ = alpha2;
    alpha3_ = alpha3;
  }

  void operator()(const Element< double, DIM > &element,
                  const Quadrature< double > &quadrature, LocalMatrix &lm) {
    AssemblyAssistant< DIM, double >::initialize_for_element(element, quadrature, false);
    // recompute previous solution values
    for (int d = 0; d < DIM; ++d) {
      vel_ns_[d].clear();
      grad_vel_ns_[d].clear();

      evaluate_fe_function(*prev_newton_sol_, d, vel_ns_[d]);
      evaluate_fe_function_gradients(*prev_newton_sol_, d, grad_vel_ns_[d]);
    }

    temp_ns_.clear();
    grad_temp_ns_.clear();
    evaluate_fe_function(*prev_newton_sol_, DIM + 1, temp_ns_);
    evaluate_fe_function_gradients(*prev_newton_sol_, DIM + 1, grad_temp_ns_);

    // indices j -> trial variable, i -> test variable

    const int num_q = num_quadrature_points();

    // loop quadrature points
    for (int q = 0; q < num_q; ++q) {
      const double wq = w(q);
      const double dJ = std::abs(detJ(q));

      // Vector with velocity of previous newton point
      Vec< DIM, double > vel_ns;
      for (int var = 0; var < DIM; ++var) {
        vel_ns.set(var, vel_ns_[var][q]);
      }

      // Vector with temperature of previous newton point
      double temp_ns; 
      Vec< DIM, double > grad_temp_ns;
      temp_ns = temp_ns_[q];
      grad_temp_ns = grad_temp_ns_[q];

      // Velocity terms symmetric in test and trial variables
      for (int var = 0; var < DIM; ++var) {
        const int n_dofs = num_dofs(var);
        for (int i = 0; i < n_dofs; ++i) {
          for (int j = 0; j < n_dofs; ++j) {
            lm(dof_index(i, var), dof_index(j, var)) +=
                wq *
                // a0(\phi_i,\phi_j) = \int{ dot(\phi_j,\phi_i) }
                (phi(j, q, var) * phi(i, q, var) +

                 // a1(\phi_i, \phi_j) = alpha1 * \int{ \nu \grad{phi_j} :
                 // \grad{\phi_i} }
                 alpha1_ * Pr_ * inv_sq_Ra_ * dot(grad_phi(j, q, var), grad_phi(i, q, var)) +

                 // c1(\phi_i,\phi_j) = \alpha1 * \int { (vel_ns*\grad{\phi_j})
                 // * \phi_i }
                 alpha1_ * dot(vel_ns, grad_phi(j, q, var)) * phi(i, q, var)) *
                dJ;
                
          }
        }
      }

      // Temperaturteil aus 1. Gleichung
      // lm(dof_index(i, g_var), dof_index(j, t_var))
      // Testfunktion für erste Gleichung, g_var-Komponente: phi(i, q, g_var)
      // Ansatzfunktionen für Temperatur: phi(j, q, t_var)
      const int t_var = DIM + 1; 
      const int g_var = DIM - 1;
      for (int i = 0; i < num_dofs(g_var); ++i) {
        for (int j = 0; j < num_dofs(t_var); ++j){
          lm(dof_index(i , g_var), dof_index(j, t_var)) -= 
          wq * alpha1_ *  Pr_ * phi(j, q, t_var) * phi(i, q, g_var) * dJ;
        }
      }

      // Druck aus 1. Gleichung
      const int p_var = DIM;
      for (int v_var = 0; v_var < DIM; ++v_var) {
          for (int i = 0; i < num_dofs(v_var); ++i) {
            for (int j = 0; j < num_dofs (p_var); ++j) {
              lm(dof_index(i, v_var), dof_index(j, p_var)) +=
                wq * alpha2_ * (
                  -phi(j, q, p_var) * grad_phi(i, q, v_var)[v_var]
                ) * dJ;
                // ? oder auch einfach wie bT wie in channel_benchmark: 
              lm(dof_index(j, p_var), dof_index(i, v_var)) +=
                wq * (
                  phi(j, q, p_var) * grad_phi(i, q, v_var)[v_var]
                ) * dJ;
            }
          }
      }

      // c2(\phi_i,\phi_j) = \int \alpha1 * { (\phi_j*\grad{vel_ns}*\phi_i }
      for (int test_var = 0; test_var < DIM; ++test_var) {
        for (int trial_var = 0; trial_var < DIM; ++trial_var) {
          for (int i = 0; i < num_dofs(test_var); ++i) {
            for (int j = 0; j < num_dofs(trial_var); ++j) {
              lm(dof_index(i, test_var), dof_index(j, trial_var)) +=
                  wq * alpha1_ *
                  (grad_vel_ns_[test_var][q][trial_var] * phi(j, q, trial_var) *
                   phi(i, q, test_var)) *
                  dJ;
            }
          }
        }
      }

      // temperature
      for (int i = 0; i < num_dofs(t_var); ++i){
          for (int j = 0; j < num_dofs(t_var); ++j){
            lm(dof_index(i, t_var), dof_index(j, t_var)) += 
              wq * (
                (phi(j, q, t_var) * phi(i, q, t_var)) +
                alpha1_ * dot(vel_ns, grad_phi(j, q, t_var)) * phi(i, q, t_var) +
                alpha1_ * inv_sq_Ra_ * dot(grad_phi(j, q, t_var), grad_phi(i, q, t_var))
              ) * dJ;
          }
      }
      for (int v_var = 0; v_var < DIM; ++v_var) {
        for (int i = 0; i < num_dofs(t_var); ++i) {
          for (int j = 0; j < num_dofs(v_var); ++j) {
              lm(dof_index(i, t_var), dof_index(j, v_var)) +=
              wq * alpha1_ * (
                phi(j, q, v_var) * grad_temp_ns[v_var] * phi(i, q, t_var) 
              ) * dJ;
          }
        }
      }
    } // end loop q
  }

  void operator()(const Element< double, DIM > &element,
                  const Quadrature< double > &quadrature, LocalVector &lv) 
  {
    AssemblyAssistant< DIM, double >::initialize_for_element(element,quadrature, false);
    // recompute previous solution values
    for (int d = 0; d < DIM; ++d) {
      vel_ns_[d].clear();
      vel_ts_[d].clear();
      grad_vel_ns_[d].clear();
      grad_vel_ts_[d].clear();

      evaluate_fe_function(*prev_newton_sol_, d, vel_ns_[d]);
      evaluate_fe_function(*prev_time_sol_, d, vel_ts_[d]);
      evaluate_fe_function_gradients(*prev_newton_sol_, d, grad_vel_ns_[d]);
      evaluate_fe_function_gradients(*prev_time_sol_, d, grad_vel_ts_[d]);
    }

    temp_ns_.clear();
    temp_ts_.clear();
    evaluate_fe_function(*prev_newton_sol_, DIM + 1, temp_ns_);
    evaluate_fe_function(*prev_time_sol_, DIM + 1, temp_ts_);
    evaluate_fe_function_gradients(*prev_newton_sol_, DIM + 1, grad_temp_ns_);
    evaluate_fe_function_gradients(*prev_time_sol_, DIM + 1, grad_temp_ts_);
    
    // recompute pressure
    p_ns_.clear();
    evaluate_fe_function(*prev_newton_sol_, DIM, p_ns_);
    // p_ts_.clear();
    // evaluate_fe_function(*prev_time_sol_, DIM, p_ts_);

    // indices j -> trial variable, i -> test variable
    // basis functions \phi -> velocity components, \eta -> pressure

    const int num_q = num_quadrature_points();

    // loop quadrature points
    for (int q = 0; q < num_q; ++q) {
      const double wq = w(q);
      const double dJ = std::abs(detJ(q));

      // get previous newton and time step solutions in vector form
      // warum nicht auch grad_x?
      Vec< DIM, double > vel_ts, vel_ns;
      for (int var = 0; var < DIM; ++var) {
        vel_ns.set(var, vel_ns_[var][q]);
        vel_ts.set(var, vel_ts_[var][q]);
      }
      double temp_ns, temp_ts;
      temp_ns = temp_ns_[q];
      temp_ts = temp_ts_[q];


      // Residual without incompressibility
      for (int v_var = 0; v_var < DIM; ++v_var) {
        for (int i = 0; i < num_dofs(v_var); ++i) {
          lv[dof_index(i, v_var)] +=
              wq * (
              // l0(\phi_i) = \int {dot(vel_ns - vel_ts, \phi_i)}
              (vel_ns[v_var] - vel_ts[v_var]) * phi(i, q, v_var) +

               // l1(\phi_i) =
               // Pr * 1/sqrt(Ra) * \int{(\alpha1\grad{vel_ns} + \alpha3\grad{vel_ts}) *
               // \grad{\phi_i}}
               Pr_ * inv_sq_Ra_ * (dot((alpha1_ * grad_vel_ns_[v_var][q]) +
                              (alpha3_ * grad_vel_ts_[v_var][q]),
                          grad_phi(i, q, v_var))) +

               // l2(\phi_i) = \int{ (\alpha1*dot(vel_ns, \grad{\vel_ns}
               //                   + \alpha3*dot(vel_ts, \grad{\vel_ts}) *
               //                   \phi_i}
               (alpha1_ * dot(grad_vel_ns_[v_var][q], vel_ns) +
                alpha3_ * dot(grad_vel_ts_[v_var][q], vel_ts)) *
                   phi(i, q, v_var) +

               // l3(\phi_i) = -\alpha2/rho*\int{p_ns * div(\phi_i)}
               -(alpha2_ * p_ns_[q] *
                 grad_phi(i, q, v_var)[v_var])

              ) * dJ;       
        }
      }

      const int g_var = DIM - 1;
      for (int i = 0; i < num_dofs(g_var); ++i){
        lv[dof_index(i, g_var)] -=
        wq * (
          alpha1_ * Pr_ * temp_ns * phi(i, q, g_var) +
          alpha3_ * Pr_ * temp_ts * phi(i, q, g_var)
        ) * dJ;
      }

      // lv_3 bzw. lv_(DIM)
    
      const int q_var = DIM;
      double div_u_k = 0.;
      for (int d = 0; d < DIM; ++d) {
        div_u_k += grad_vel_ns_[d][q][d];
      }

      for (int i = 0; i < num_dofs(q_var); ++i) {
        lv[dof_index(i, q_var)] +=
            wq * (div_u_k * phi(i, q, q_var)) * dJ;
      }

// Was ist mit dem Rand? 
    const int t_var = DIM + 1;
    for(int i = 0; i < num_dofs(t_var); ++i){
      lv[dof_index(i, t_var)] +=
      wq * (
        ( temp_ns - temp_ts ) * phi(i, q, t_var) +
        alpha1_ * ( dot(vel_ns, grad_temp_ns_[q]) * phi(i, q, t_var)
        + inv_sq_Ra_ * dot(grad_temp_ns_[q], grad_phi(i, q, t_var)))
        + alpha3_ * ( dot( vel_ts, grad_temp_ts_[q]) *  phi(i, q, t_var) 
        + inv_sq_Ra_ * dot( grad_temp_ts_[q], grad_phi(i, q, t_var))
        )
      ) * dJ;
    }

    } // end quadrature loop
  }

private:
  const CVector *prev_time_sol_;
  const CVector *prev_newton_sol_;

  double alpha1_, alpha2_, alpha3_;

  double rho_, Pr_, Ra_, inv_sq_Ra_;
  FunctionValues< double > vel_ns_[DIM]; // velocity at previous newton step
  FunctionValues< double > vel_ts_[DIM]; // velocity at previous timestep
  FunctionValues< double > p_ns_; // pressure at previous newton step
  FunctionValues< double > temp_ns_; // temperature at previous newton step
  FunctionValues< double > temp_ts_; // temperature at previous timestep
  FunctionValues< Vec< DIM, double > >
      grad_vel_ns_[DIM]; // gradient of velocity at previous newton step
  FunctionValues< Vec< DIM, double > >
      grad_vel_ts_[DIM]; // gradient of velocity at previous timestep
  FunctionValues< Vec< DIM, double > >
      grad_temp_ns_; // gradient of temperature at previous newton step
  FunctionValues< Vec< DIM, double > >
      grad_temp_ts_; // gradient of temperature at previous timestep
};

// Functor used for the local evaluation of the square of the L2-norm
// of a set of variables on each element.

class L2NormIntegratorPp : private AssemblyAssistant< DIM, double > {
public:
  L2NormIntegratorPp(CVector &pp_sol,
                     const std::vector< int > &vars)
      : pp_sol_(pp_sol), vars_(vars) {}

  void operator()(const Element< double, DIM > &element,
                  const Quadrature< double > &quadrature, double &value) {
    AssemblyAssistant< DIM, double >::initialize_for_element(element, quadrature, false);

    const int num_q = num_quadrature_points();
    for (int v = 0, end_v = vars_.size(); v != end_v; ++v) {
      const int var = vars_[v];
      evaluate_fe_function(pp_sol_, var, approx_sol_);

      for (int q = 0; q < num_q; ++q) {
        const double wq = w(q);
        const double dJ = std::abs(detJ(q));

        value += wq * approx_sol_[q] * approx_sol_[q] * dJ;
      }
    }
  }

private:
  // coefficients of the computed solution
  CVector &pp_sol_;
  // vector with values of computed solution evaluated at each quadrature point
  FunctionValues< double > approx_sol_;
  // variables for which to compute the norm
  std::vector< int > vars_;
};

// Functor used for the local evaluation of the square of the H1-seminorm
// of a set of variables on each element.

class H1semiNormIntegratorPp : private AssemblyAssistant< DIM, double > {
public:
  H1semiNormIntegratorPp(CVector &pp_sol,
                         const std::vector< int > &vars)
      : pp_sol_(pp_sol), vars_(vars) {}

  void operator()(const Element< double, DIM > &element,
                  const Quadrature< double > &quadrature, double &value) {
    AssemblyAssistant< DIM, double >::initialize_for_element(element, quadrature, false);

    const int num_q = num_quadrature_points();
    for (int v = 0, end_v = vars_.size(); v != end_v; ++v) {
      const int var = vars_[v];
      evaluate_fe_function_gradients(pp_sol_, var, approx_sol_grad_);

      for (int q = 0; q < num_q; ++q) {
        const double wq = w(q);
        const double dJ = std::abs(detJ(q));

        value += wq * dot(approx_sol_grad_[q], approx_sol_grad_[q]) * dJ;
      }
    }
  }

private:
  // coefficients of the computed solution
  CVector &pp_sol_;
  // vector with values of computed solution evaluated at each quadrature point
  FunctionValues< Vec< DIM, double > > approx_sol_grad_;
  // variables for which to compute the norm
  std::vector< int > vars_;
};

// Functor used for the local evaluation of the square of the L2-norm of the
// difference between the solution of last end penultimate timestep.

class InstationaryL2ErrorIntegrator
    : private AssemblyAssistant< DIM, double > {
public:
  InstationaryL2ErrorIntegrator(const LAD::VectorType &coeff,
                                const LAD::VectorType &coeff_penult)
      : coeff_(coeff), coeff_penult_(coeff_penult) {}

  void operator()(const Element< double, DIM > &element,
                  const Quadrature< double > &quadrature, double &value) {
    AssemblyAssistant< DIM, double >::initialize_for_element(element, quadrature, false);

    const int num_q = num_quadrature_points();
    for (int var = 0; var < DIM; ++var) {
      evaluate_fe_function(coeff_, var, approx_sol_);
      evaluate_fe_function(coeff_penult_, var, approx_sol_penult_);
      for (int q = 0; q < num_q; ++q) {
        const double wq = w(q);
        const double dJ = std::abs(detJ(q));
        const double delta = approx_sol_penult_[q] - approx_sol_[q];
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
  FunctionValues< double > approx_sol_, approx_sol_penult_;
};

// Functor used for the local evaluation of the square of the L2-norm of the
// solution in one timestep on each element.

class InstationaryL2Integrator
    : private AssemblyAssistant< DIM, double > {
public:
  InstationaryL2Integrator(const LAD::VectorType &coeff) : coeff_(coeff) {}

  void operator()(const Element< double, DIM > &element,
                  const Quadrature< double > &quadrature, double &value) {
    AssemblyAssistant< DIM, double >::initialize_for_element(element, quadrature, false);

    const int num_q = num_quadrature_points();
    for (int var = 0; var < DIM; ++var) {
      evaluate_fe_function(coeff_, var, approx_sol_);
      for (int q = 0; q < num_q; ++q) {
        const double wq = w(q);
        const double dJ = std::abs(detJ(q));
        value += wq * approx_sol_[q] * approx_sol_[q] * dJ;
      }
    }
  }

private:
  // coefficients of soluition of a timestep
  const LAD::VectorType &coeff_;
  // vector with values of computed solution evaluated at each quadrature point
  // for a timestep
  FunctionValues< double > approx_sol_;
};


#endif
