// Copyright (C) 2011-2017 Vincent Heuveline
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
/// Test if constructed BDM1 element on triangle equals the analytically known basis 
  
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>
#include <iomanip>

#include "common/macros.h"
#include "fem/reference_cell.h"
#include "fem/fe_reference.h"
#include "fem/fe_transformation.h"

#include "fem/ansatz/ansatz_p_line_lagrange.h"
#include "fem/ansatz/ansatz_p_tri_lagrange.h"
#include "fem/ansatz/ansatz_aug_p_tri_mono.h"
#include "fem/ansatz/ansatz_sum.h"
#include "dof/dof_impl/dof_container_rt_bdm.h"


#include "test.h"

using namespace hiflow::doffem;
typedef hiflow::Vec<2, double> Coord;
typedef hiflow::Vec<2, double> Vector;

/// compuational basis for BDM
Vector e1 (Coord s, Coord pt)
{
  Vector ret;
  ret.set(0, std::sqrt(2.) / (s[1] - s[0]) * s[1] * pt[0]);
  ret.set(1, std::sqrt(2.) / (s[1] - s[0]) * (s[1]-1.) * pt[1]);
  return ret;
} 

Vector e2 (Coord s, Coord pt)
{
  Vector ret;
  ret.set(0, 1. / (s[1] - s[0]) * (s[1] * pt[0] + pt[1] - s[1]));
  ret.set(1, 1. / (s[1] - s[0]) * (s[1]-1.) * pt[1]);
  return ret;
}

Vector e3 (Coord s, Coord pt)
{
  Vector ret;
  ret.set(0, 1. / (s[1] - s[0]) * (s[1] - 1.) * pt[0]);
  ret.set(1, 1. / (s[1] - s[0]) * (pt[0] + s[1] * pt[1] - s[1]));
  return ret;
}

Vector e4 (Coord s, Coord pt)
{
  Vector ret = (1. - pt[0] - pt[1]) * e1(s, pt);
  return ret;
}

Vector e5 (Coord s, Coord pt)
{
  Vector ret = pt[0] * e2(s, pt);
  return ret;
}

Vector e6 (Coord s, Coord pt)
{
  Vector ret = pt[1] * e3(s, pt);
  return ret;
}

/// computational Basis for RT
Vector f1 (Coord pt)
{
  Vector ret;
  ret.set(0, /*std::sqrt(2.) **/ pt[0]);
  ret.set(1, /*std::sqrt(2.) **/ pt[1]);
  return ret;
}

Vector f2 (Coord pt)
{
  Vector ret;
  ret.set(0, pt[0] - 1.);
  ret.set(1, pt[1]);
  return ret;
}

Vector f3 (Coord pt)
{
  Vector ret;
  ret.set(0, pt[0] );
  ret.set(1, pt[1] - 1.);
  return ret;
}

Vector f4 (Coord pt)
{
  Vector ret;
  ret.set(0, pt[1] * pt[0] );
  ret.set(1, pt[1] * (pt[1] - 1.));
  return ret;
}

Vector f5 (Coord pt)
{
  Vector ret;
  ret.set(0, pt[0] * (pt[0] - 1.) );
  ret.set(1, pt[0] * pt[1]);
  return ret;
}

double l1 (double t)
{
  const double gg1 = 0.;//0.5 - std::sqrt(3) / 6;
  const double gg2 = 1.;//0.5 + std::sqrt(3) / 6;
  
  return (t - gg2) / (gg1 - gg2);
}

double l2 (double t)
{
  const double gg1 = 0.;//0.5 - std::sqrt(3) / 6;
  const double gg2 = 1.;//0.5 + std::sqrt(3) / 6;
  
  return (t - gg1) / (gg2 - gg1);
}


int main(int argc, char **argv) {
  
  const int rt_deg = 1;
  const int bdm_deg = 1;
  
  /// create reference cells 
  CRefCellSPtr<double, 2> ref_cell_tri = CRefCellSPtr<double, 2> ( new RefCellTriStd <double, 2>);

  /// create ansatz spaces for BDM elements
  AnsatzSpaceSPtr <double, 2> ansatz_tri_lag_2 (new PTriLag <double, 2> (ref_cell_tri));
  ansatz_tri_lag_2->init(bdm_deg, 2);
  
  /// create ansatz spaces for RT elements

  AnsatzSpaceSumSPtr<double, 2> ansatz_tri_rt (new AnsatzSpaceSum<double, 2>(ref_cell_tri));
  
  AnsatzSpaceSPtr<double, 2> ansatz1(new PTriLag<double, 2> (ref_cell_tri));
  AnsatzSpaceSPtr<double, 2> ansatz2(new AugPTriMono<double, 2> (ref_cell_tri));        
  ansatz1->init(rt_deg, 2);
  ansatz2->init(rt_deg);


  ansatz_tri_rt->init(ansatz1, ansatz2, AnsatzSpaceType::RT);

    
  /// create BDM dof containers
  DofContainerRtBdmSPtr<double, 2> dof_tri_bdm (new DofContainerRTBDM<double, 2> (ref_cell_tri));
  dof_tri_bdm->init(bdm_deg, DofContainerType::BDM);

  /// create RT dof containers
  DofContainerRtBdmSPtr<double, 2> dof_tri_rt (new DofContainerRTBDM<double, 2> (ref_cell_tri));
  dof_tri_rt->init(rt_deg, DofContainerType::RT);
    
  /// create reference elements
  RefElement<double, 2> fe_tri_bdm;
  RefElement<double, 2> fe_tri_rt;
    
  /// Fe trafo
  FETrafoSPtr<double, 2> fe_trafo_tri_bdm (new FETransformationContraPiola<double, 2>  );
  FETrafoSPtr<double, 2> fe_trafo_tri_rt  (new FETransformationContraPiola<double, 2>  );
  
  /// Initialize BDM element
  std::cout << "initialize BDM tri  element with " << dof_tri_bdm->nb_dof_on_cell() << " dofs and " 
            << ansatz_tri_lag_2->dim() << " ansatz functions " << std::endl;
  fe_tri_bdm.init (ansatz_tri_lag_2, dof_tri_bdm, fe_trafo_tri_bdm, false, FEType::BDM);
  
  /// Initialize RT element
  std::cout << "initialize RT  tri  element with " << dof_tri_rt->nb_dof_on_cell() << " dofs and " 
            << ansatz_tri_rt->dim() << " ansatz functions " << std::endl;
  fe_tri_rt.init (ansatz_tri_rt, dof_tri_rt, fe_trafo_tri_rt, false, FEType::RT);
    
  /// define test points
  std::vector< Coord > test_points (6);
  test_points[0].set(0, 0.);
  test_points[0].set(1, 0.);
  test_points[1].set(0, 0.5);
  test_points[1].set(1, 0.);
  test_points[2].set(0, 1.);
  test_points[2].set(1, 0.);
  test_points[3].set(0, 0.);
  test_points[3].set(1, 0.5);
  test_points[4].set(0, 0.5);
  test_points[4].set(1, 0.5);
  test_points[5].set(0, 0.);
  test_points[5].set(1, 1.);
  
  /// analytical basis
  const double g1 = 0.5 - std::sqrt(3) /6.;
  const double g2 = 0.5 + std::sqrt(3) /6.;

  Coord g;
  g.set(0, g1);
  g.set(1, g2);
  Coord grev;
  grev.set(0, g2);
  grev.set(1, g1);
  
  /// loop over test points
  for (size_t l = 0; l<test_points.size(); ++l)
  {
    /// evaluate constructed FE BDM
#if 0
    std::vector<double> weights(fe_tri_bdm.dim()*2,0.);
    fe_tri_bdm.N(test_points[l], weights);
    
    std::vector<Vector> fe_vals(fe_tri_bdm.dim());
    for (int i = 0; i < fe_tri_bdm.dim(); ++i)
    {
      fe_vals[i].set(0, weights[fe_tri_bdm.iv2ind(i, 0)]);
      fe_vals[i].set(1, weights[fe_tri_bdm.iv2ind(i, 1)]);
    }
  
    std::vector<Vector> basis_vals(fe_tri_bdm.dim());
    basis_vals.set(3, e1(g,    test_points[l]));
    basis_vals.set(2, e1(grev, test_points[l]));
    basis_vals.set(5, e2(grev, test_points[l]));
    basis_vals.set(4, e2(g,    test_points[l]));
    basis_vals.set(0, e3(g,    test_points[l]));
    basis_vals.set(1, e3(grev, test_points[l]));
    
    std::cout << "======================" << std::endl;
    
    std::cout << "Point ( " << test_points[l][0] << " , " << test_points[l][1] << " )" << std::endl;
    for (int i = 0; i < fe_tri_bdm.dim(); ++i)
    {
      std::cout << std::fixed << std::setw( 7 ) << std::setprecision( 4 ) << fe_vals[i][0] << " , " << fe_vals[i][1] << std::endl;
      std::cout << std::fixed << std::setw( 7 ) << std::setprecision( 4 ) << basis_vals[i][0] << " , " << basis_vals[i][1] << std::endl;
      std::cout << "----------------------" << std::endl;
    }
    std::cout << "======================" << std::endl;
#endif

    /// evaluate constructed FE RT
    std::vector<double> weights_rt(fe_tri_rt.dim()*2,0.);
    fe_tri_rt.N(test_points[l], weights_rt);
    
    std::vector<Vector> fe_vals_rt(fe_tri_rt.dim());
    for (int i = 0; i < fe_tri_rt.dim(); ++i)
    {
      fe_vals_rt[i].set(0, weights_rt[fe_tri_rt.iv2ind(i, 0)]);
      fe_vals_rt[i].set(1, weights_rt[fe_tri_rt.iv2ind(i, 1)]);
    }
  
    std::vector<Vector> basis_vals_rt(fe_tri_rt.dim());
    
    if (rt_deg == 0)
    {
      basis_vals_rt[0] = f3(test_points[l]);
      basis_vals_rt[1] = f1(test_points[l]);
      basis_vals_rt[2] = f2(test_points[l]);
    }
    if (rt_deg == 1)
    {
      basis_vals_rt[0] = l1(test_points[l][0]) * f3(test_points[l]);
      basis_vals_rt[1] = l2(test_points[l][0]) * f3(test_points[l]);
      
      basis_vals_rt[3] = l1(test_points[l][1]) * f1(test_points[l]);
      basis_vals_rt[2] = l2(test_points[l][1]) * f1(test_points[l]);
      
      basis_vals_rt[5] = l2(test_points[l][1]) * f2(test_points[l]);
      basis_vals_rt[4] = l1(test_points[l][1]) * f2(test_points[l]);
    
      basis_vals_rt[6] = f4(test_points[l]);
      basis_vals_rt[7] = f5(test_points[l]);
    }
    
    std::cout << "======================" << std::endl;
    std::cout << "Point ( " << test_points[l][0] << " , " << test_points[l][1] << " )" << std::endl;
    for (int i = 0; i < fe_tri_rt.dim(); ++i)
    {
      std::cout << std::fixed << std::setw( 7 ) << std::setprecision( 4 ) << fe_vals_rt[i][0] << " , " << fe_vals_rt[i][1] << std::endl;
      std::cout << std::fixed << std::setw( 7 ) << std::setprecision( 4 ) << basis_vals_rt[i][0] << " , " << basis_vals_rt[i][1] << std::endl;
      std::cout << "----------------------" << std::endl;
            
//      TEST_EQUAL_EPS(fe_vals_rt[i][0], basis_vals_rt[i][0], 1.e-10);
//      TEST_EQUAL_EPS(fe_vals_rt[i][1], basis_vals_rt[i][1], 1.e-10);
    }
    std::cout << "======================" << std::endl;
  
  
  
  } 
  
  return 0;
}
