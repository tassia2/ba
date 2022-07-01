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
  DirichletBC()
  {}
  
  DataType evaluate (const int material_number,
                     const Vec< DIM, DataType> &pt_coord) const 
  {
    // TODO: EXERCISE B)

    return 0.;
    // END EXERCISE B)
  }
  
  void evaluate(const mesh::Entity &face,
                const Vec< DIM, DataType> &coords_on_face, 
                std::vector<DataType> &vals) const 
  {
    const int material_number = face.get_material_number();
    vals.clear();
    
    DataType dir_value = this->evaluate(material_number, coords_on_face);
    
    // return array with Dirichlet values for dof:s on boundary face
    vals = std::vector< DataType >(1, dir_value);
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
  
};

// Functor used for the local assembly of the stiffness matrix and load vector.
class LocalPoissonAssembler : private AssemblyAssistant< DIM, DataType > 
{
public:
  
  void set_parameters (DataType kappa, DataType beta, DataType gamma)
  {
    this->kappa_ = kappa;
    this->beta_ = beta;
    this->gamma_ = gamma;
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
      
      Vec<DIM, DataType> conv = this->convection(this->x(q));
      
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
          // TODO exercise C)
          
          lm(i, j) += wq * 
                    (1.)
                   * dJ;
                   
          // END EXERCISE C)
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
        // TODO exercise C)
        
        lv[i] += wq 
                * 1.
                * dJ;
                
        // END EXERCISE C)
      } 
    }
  }

  Vec<DIM, DataType> convection (const Vec< DIM, DataType >& pt) const 
  {
    Vec<DIM, DataType> conv;
   
    // TODO Exercise C)
    conv.set(0, 1. / std::sqrt(2.));
    conv.set(1, 1. / std::sqrt(2.));
    
    // END EXERCISE C)
    return conv;
  }
   
  DataType f(Vec< DIM, DataType > pt) 
  {
    return 0.;
  }
  
  DataType kappa_, beta_, gamma_;
};
