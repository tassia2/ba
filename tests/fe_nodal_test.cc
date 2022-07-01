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
/// TODO: uncomment missing elements

#define BOOST_TEST_MODULE fe_transformation

#include <boost/test/unit_test.hpp>
#include <boost/test/included/unit_test.hpp>

#include <mpi.h>
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

#include "common/macros.h"
#include "common/vector_algebra.h"
#include "fem/reference_cell.h"
#include "fem/fe_reference.h"
#include "fem/fe_transformation.h"

#include "fem/ansatz/ansatz_p_line_lagrange.h"
#include "fem/ansatz/ansatz_p_tri_lagrange.h"
#include "fem/ansatz/ansatz_aug_p_tri_mono.h"
#include "fem/ansatz/ansatz_aug_p_tet_mono.h"
#include "fem/ansatz/ansatz_sum.h"
#include "fem/ansatz/ansatz_transformed.h"
#include "fem/ansatz/ansatz_p_tet_lagrange.h"
#include "fem/ansatz/ansatz_q_quad_lagrange.h"
#include "fem/ansatz/ansatz_q_hex_lagrange.h"
#include "fem/ansatz/ansatz_pyr_lagrange.h"

#include "dof/dof_impl/dof_container_lagrange.h"
#include "dof/dof_impl/dof_container_rt_bdm.h"


#include "test.h"

using namespace hiflow::doffem;

BOOST_AUTO_TEST_CASE(fe_nodal) {

  int argc = boost::unit_test::framework::master_test_suite().argc;
  char** argv = boost::unit_test::framework::master_test_suite().argv;
  
  int rank;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD,  &rank);
  
  if (rank >= 1)
  {
    return;
  }
  
  /// Test if shape functions satisfy nodal property, 

  /// create reference cells 
  CRefCellSPtr<double, 1> ref_cell_line = CRefCellSPtr<double, 1> ( new RefCellLineStd<double, 1>);
  CRefCellSPtr<double, 2> ref_cell_tri = CRefCellSPtr<double, 2> ( new RefCellTriStd <double, 2>);
  CRefCellSPtr<double, 2> ref_cell_quad = CRefCellSPtr<double, 2> ( new RefCellQuadStd<double, 2>);
  CRefCellSPtr<double, 3> ref_cell_tet = CRefCellSPtr<double, 3> ( new RefCellTetStd <double, 3>);
  CRefCellSPtr<double, 3> ref_cell_hex = CRefCellSPtr<double, 3> ( new RefCellHexStd <double, 3>);
  CRefCellSPtr<double, 3> ref_cell_pyr = CRefCellSPtr<double, 3> ( new RefCellPyrStd <double, 3>);

  /// create Lagrange ansatz spaces
  AnsatzSpaceSPtr<double, 1> ansatz_line_lag (new PLineLag<double, 1> (ref_cell_line));
  AnsatzSpaceSPtr<double, 2> ansatz_tri_lag (new PTriLag <double, 2> (ref_cell_tri));
  AnsatzSpaceSPtr<double, 3> ansatz_tet_lag (new PTetLag <double, 3> (ref_cell_tet));
  AnsatzSpaceSPtr<double, 2> ansatz_quad_lag (new QQuadLag<double, 2> (ref_cell_quad));
  AnsatzSpaceSPtr<double, 3> ansatz_hex_lag (new QHexLag <double, 3> (ref_cell_hex));
  AnsatzSpaceSPtr<double, 3> ansatz_pyr_lag (new PyrLag  <double, 3> (ref_cell_pyr));

  /// create Lagrange dof containers
  DofContainerLagSPtr<double, 1> dof_line_lag (new DofContainerLagrange<double, 1> (ref_cell_line));
  DofContainerLagSPtr<double, 2> dof_tri_lag  (new DofContainerLagrange<double, 2> (ref_cell_tri) );
  DofContainerLagSPtr<double, 3> dof_tet_lag  (new DofContainerLagrange<double, 3> (ref_cell_tet) );
  DofContainerLagSPtr<double, 2> dof_quad_lag (new DofContainerLagrange<double, 2> (ref_cell_quad));
  DofContainerLagSPtr<double, 3> dof_hex_lag  (new DofContainerLagrange<double, 3> (ref_cell_hex) );
  DofContainerLagSPtr<double, 3> dof_pyr_lag  (new DofContainerLagrange<double, 3> (ref_cell_pyr) );
  
  /// create ansatz spaces for RT and BDM elements
  AnsatzSpaceSPtr<double, 2> ansatz_tri_lag_2 (new PTriLag <double, 2>(ref_cell_tri));
  
  AnsatzSpaceSumSPtr<double, 2> ansatz_tri_rt (new AnsatzSpaceSum<double, 2> (ref_cell_tri));
  AnsatzSpaceSumSPtr<double, 3> ansatz_tet_rt (new AnsatzSpaceSum<double, 3> (ref_cell_tet));

  AnsatzSpaceSPtr<double, 2> ansatz_tri_rt_1_ptr(new PTriLag<double, 2>(ref_cell_tri));
  AnsatzSpaceSPtr<double, 2> ansatz_tri_rt_2_ptr(new AugPTriMono<double, 2>(ref_cell_tri));
  AnsatzSpaceSPtr<double, 3> ansatz_tet_rt_1_ptr(new PTetLag<double, 3>(ref_cell_tet));
  AnsatzSpaceSPtr<double, 3> ansatz_tet_rt_2_ptr(new AugPTetMono<double, 3>(ref_cell_tet));

  // RT quad
  AnsatzSpaceSPtr<double, 2> ansatz_quad_rt (new QQuadLag<double, 2> (ref_cell_quad));
  AnsatzSpaceSPtr<double, 3> ansatz_hex_rt  (new QHexLag<double, 3>  (ref_cell_hex) );

  /// create BDM dof containers 
  DofContainerRtBdmSPtr<double, 2> dof_tri_bdm (new DofContainerRTBDM<double, 2> (ref_cell_tri) );
  DofContainerRtBdmSPtr<double, 2> dof_tri_rt  (new DofContainerRTBDM<double, 2> (ref_cell_tri) );
  DofContainerRtBdmSPtr<double, 3> dof_tet_rt  (new DofContainerRTBDM<double, 3> (ref_cell_tet) );
  DofContainerRtBdmSPtr<double, 2> dof_quad_rt (new DofContainerRTBDM<double, 2> (ref_cell_quad));
  DofContainerRtBdmSPtr<double, 3> dof_hex_rt  (new DofContainerRTBDM<double, 3> (ref_cell_hex) );

  const int min_bdm_deg = 1;
  const int max_bdm_deg = 3;  // TODO: this should be 3
  
  const int min_rt_deg = 0;
  const int max_rt_deg = 3;  // TODO: this should be 3
  
  const size_t lag_nb_comp = 2;
   
  for (int deg = 0; deg < 4; ++deg) 
  {
    CONSOLE_OUTPUT(rank, " ===================== ");

#if 1
    bool modal_basis = false;

    /// Lagrange line   
    ansatz_line_lag->init(deg, lag_nb_comp);
    dof_line_lag->init(deg, lag_nb_comp);
    CONSOLE_OUTPUT(rank, "initialize Lagrange line element of degree " << deg << " with " << dof_line_lag->nb_dof_on_cell() << " dofs and " 
              << ansatz_line_lag->dim() << " ansatz functions ");
    RefElement<double, 1> fe_line_lag;
    FETrafoSPtr<double, 1> fe_trafo_line_lag (new FETransformationStandard<double, 1> );
    fe_line_lag.init(ansatz_line_lag,dof_line_lag,fe_trafo_line_lag,modal_basis, FEType::LAGRANGE);

    /// Lagrange triangle
    ansatz_tri_lag->init(deg, lag_nb_comp);
    dof_tri_lag->init(deg, lag_nb_comp);
    CONSOLE_OUTPUT(rank, "initialize Lagrange tri  element of degree " << deg << " with " << dof_tri_lag->nb_dof_on_cell() << " dofs and " 
              << ansatz_tri_lag->dim() << " ansatz functions ");
    RefElement<double, 2> fe_tri_lag;
    FETrafoSPtr<double, 2> fe_trafo_tri_lag (new FETransformationStandard<double, 2> );
    fe_tri_lag.init (ansatz_tri_lag, dof_tri_lag, fe_trafo_tri_lag, modal_basis, FEType::LAGRANGE);
    
    /// Lagrange tet
    ansatz_tet_lag->init(deg, lag_nb_comp);
    dof_tet_lag->init(deg, lag_nb_comp);
    std::cout << "initialize Lagrange tetrahedron element with " << dof_tet_lag->nb_dof_on_cell() << " dofs and " 
              << ansatz_tet_lag->dim() << " ansatz functions " << std::endl;
    RefElement<double, 3> fe_tet_lag;
    FETrafoSPtr<double, 3> fe_trafo_tet_lag (new FETransformationStandard<double, 3> );
    fe_tet_lag.init (ansatz_tet_lag, dof_tet_lag, fe_trafo_tet_lag, modal_basis, FEType::LAGRANGE);

    /// Lagrange quad
    ansatz_quad_lag->init(deg, lag_nb_comp);
    dof_quad_lag->init(deg, lag_nb_comp);
    std::cout << "initialize Lagrange quad element with " << dof_quad_lag->nb_dof_on_cell() << " dofs and " 
              << ansatz_quad_lag->dim() << " ansatz functions " << std::endl;
    RefElement<double, 2> fe_quad_lag;
    FETrafoSPtr<double, 2> fe_trafo_quad_lag (new FETransformationStandard<double, 2> );
    fe_quad_lag.init(ansatz_quad_lag,dof_quad_lag,fe_trafo_quad_lag,modal_basis, FEType::LAGRANGE);

    ///  Lagrange hex
    ansatz_hex_lag->init(deg, lag_nb_comp);
    dof_hex_lag->init(deg, lag_nb_comp);
    CONSOLE_OUTPUT(rank, "initialize Lagrange hex  element of degree " << deg << " with " << dof_hex_lag->nb_dof_on_cell() << " dofs and " 
              << ansatz_hex_lag->dim() << " ansatz functions ");
    RefElement<double, 3> fe_hex_lag;
    FETrafoSPtr<double, 3> fe_trafo_hex_lag (new FETransformationStandard<double, 3> );
    fe_hex_lag.init (ansatz_hex_lag, dof_hex_lag, fe_trafo_hex_lag, modal_basis, FEType::LAGRANGE);

    /// Lagrange pyr
    size_t pyr_deg = std::min(2, deg);
    ansatz_pyr_lag->init(pyr_deg);
    dof_pyr_lag->init(pyr_deg, 1);
    
    CONSOLE_OUTPUT(rank, "initialize Lagrange pyramid element with " << dof_pyr_lag->nb_dof_on_cell() << " dofs and " 
              << ansatz_pyr_lag->dim() << " ansatz functions ");
    RefElement<double, 3> fe_pyr_lag;
    FETrafoSPtr<double, 3> fe_trafo_pyr_lag (new FETransformationStandard<double, 3> );
    fe_pyr_lag.init (ansatz_pyr_lag,dof_pyr_lag, fe_trafo_pyr_lag, modal_basis, FEType::LAGRANGE);
#endif

    /// BDM
    RefElement<double, 2> fe_tri_bdm;
    FETrafoSPtr<double, 2> fe_trafo_tri_bdm (new FETransformationContraPiola<double, 2> );
    if (deg >= min_bdm_deg && deg <= max_bdm_deg)
    {
      ansatz_tri_lag_2->init(deg, 2);
      dof_tri_bdm->init(deg, DofContainerType::BDM);
      CONSOLE_OUTPUT(rank, "initialize BDM(" << deg << ") tri  element with " << dof_tri_bdm->nb_dof_on_cell() << " dofs and " 
                << ansatz_tri_lag_2->dim() << " ansatz functions ");
      fe_tri_bdm.init (ansatz_tri_lag_2, dof_tri_bdm, fe_trafo_tri_bdm, false, FEType::BDM);
    }

    /// RT
    RefElement<double, 2> fe_tri_rt;
    FETrafoSPtr<double, 2> fe_trafo_tri_rt (new FETransformationContraPiola<double, 2> );
    if (deg >= min_rt_deg && deg <= max_rt_deg)
    {
      ansatz_tri_rt_1_ptr->init(deg,2);
      ansatz_tri_rt_2_ptr->init(deg);
          
      ansatz_tri_rt->init(ansatz_tri_rt_1_ptr, ansatz_tri_rt_2_ptr, AnsatzSpaceType::RT);

      dof_tri_rt->init(deg, DofContainerType::RT);  
      CONSOLE_OUTPUT(rank, "initialize RT(" << deg << ")  tri  element with " << dof_tri_rt->nb_dof_on_cell() << " dofs and " 
              << ansatz_tri_rt->dim() << " ansatz functions ");

      fe_tri_rt.init (ansatz_tri_rt, dof_tri_rt, fe_trafo_tri_rt, false, FEType::RT);
      BOOST_TEST(fe_tri_rt.dim() == (deg+1)*(deg+3));
      BOOST_TEST(dof_tri_rt->nb_dof_on_cell() == ansatz_tri_rt->dim());
    }

    /// RT tet
    RefElement<double, 3> fe_tet_rt;
    FETrafoSPtr<double, 3> fe_trafo_tet_rt (new FETransformationContraPiola<double, 3> );
    if (deg >= min_rt_deg && deg <= max_rt_deg)
    {
      ansatz_tet_rt_1_ptr->init(deg,3);
      ansatz_tet_rt_2_ptr->init(deg);
        
      ansatz_tet_rt->init(ansatz_tet_rt_1_ptr, ansatz_tet_rt_2_ptr, AnsatzSpaceType::RT);

      dof_tet_rt->init(deg, DofContainerType::RT);  
      CONSOLE_OUTPUT(rank, "initialize RT(" << deg << ")  tet  element with " << dof_tet_rt->nb_dof_on_cell() << " dofs and " 
              << ansatz_tet_rt->dim() << " ansatz functions ");

      fe_tet_rt.init (ansatz_tet_rt, dof_tet_rt, fe_trafo_tet_rt, false, FEType::RT);
      BOOST_TEST(fe_tet_rt.dim() == (deg+1)*(deg+2)*(deg+4)/2);
      BOOST_TEST(dof_tet_rt->nb_dof_on_cell() == ansatz_tet_rt->dim());
    }
    
    /// RT quad
    RefElement<double, 2> fe_quad_rt;
    FETrafoSPtr<double, 2> fe_trafo_quad_rt (new FETransformationContraPiola<double, 2> );
    if (deg >= min_rt_deg && deg <= max_rt_deg)
    {
      std::vector< std::vector <size_t> > degrees;
      degrees.resize(2);
      degrees[0].resize(2);
      degrees[1].resize(2);
      degrees[0][0] = deg + 1;
      degrees[0][1] = deg;
      degrees[1][0] = deg;
      degrees[1][1] = deg + 1;
      ansatz_quad_rt->init(degrees);

      dof_quad_rt->init(deg, DofContainerType::RT);  

      CONSOLE_OUTPUT(rank, "initialize RT(" << deg << ")  quad  element with " << dof_quad_rt->nb_dof_on_cell() << " dofs and " 
              << ansatz_quad_rt->dim() << " ansatz functions ");

      fe_quad_rt.init (ansatz_quad_rt, dof_quad_rt, fe_trafo_quad_rt, false, FEType::RT);
      BOOST_TEST(fe_quad_rt.dim() == 2 * (deg + 1) * (deg + 2));
      BOOST_TEST(dof_quad_rt->nb_dof_on_cell() == ansatz_quad_rt->dim());
    }

    /// RT hex
    RefElement<double, 3> fe_hex_rt;
    FETrafoSPtr<double, 3> fe_trafo_hex_rt (new FETransformationContraPiola<double, 3> );
    if (deg >= min_rt_deg && deg <= max_rt_deg)
    {
      std::vector< std::vector <size_t> > degrees;
      degrees.resize(3);
      degrees[0].resize(3);
      degrees[1].resize(3);
      degrees[2].resize(3);
      degrees[0][0] = deg + 1;
      degrees[0][1] = deg;
      degrees[0][2] = deg;
      degrees[1][0] = deg;
      degrees[1][1] = deg + 1;
      degrees[1][2] = deg;
      degrees[2][0] = deg;
      degrees[2][1] = deg;
      degrees[2][2] = deg + 1;
      ansatz_hex_rt->init(degrees);

      dof_hex_rt->init(deg, DofContainerType::RT);  

      CONSOLE_OUTPUT(rank, "initialize RT(" << deg << ")  hex  element with " << dof_hex_rt->nb_dof_on_cell() << " dofs and " 
              << ansatz_hex_rt->dim() << " ansatz functions ");

      fe_hex_rt.init (ansatz_hex_rt, dof_hex_rt, fe_trafo_hex_rt, false, FEType::RT);
      BOOST_TEST(fe_hex_rt.dim() == 3 * (deg + 1) * (deg + 1) * (deg + 2));
      BOOST_TEST(dof_hex_rt->nb_dof_on_cell() == ansatz_hex_rt->dim());
    }
    
#if 1
    /// Test Lagrange HEXAHEDRON
    std::vector< cDofId > dof_ids_hex_lag  (dof_hex_lag->nb_dof_on_cell(),0);
    for (size_t i = 0; i < dof_hex_lag->nb_dof_on_cell(); ++i)
    {
      dof_ids_hex_lag[i] = i;
    }
    std::vector< std::vector<double> > dof_val_hex_lag;
    dof_hex_lag->evaluate (&fe_hex_lag, dof_ids_hex_lag, dof_val_hex_lag);
    
    for (size_t i = 0; i < dof_val_hex_lag.size(); ++i)
    {
      for (size_t j = 0; j < dof_val_hex_lag[i].size(); ++j)
      {
        double val = dof_val_hex_lag[i][j];
        if (i != j)
        {
          BOOST_TEST(std::abs(val) < 1.e-10);
        }
        else
        {
          BOOST_TEST(std::abs(val - 1.0) < 1.e-10);
        }
      }
    }
    CONSOLE_OUTPUT(rank, "hex  lag(" << deg << ") test: passed ");

    /// Test Lagrange TETRAHEDRON
    std::vector< cDofId > dof_ids_tet_lag  (dof_tet_lag->nb_dof_on_cell(),0);
    for (size_t i = 0; i < dof_tet_lag->nb_dof_on_cell(); ++i)
    {
      dof_ids_tet_lag[i] = i;
    }
    std::vector< std::vector<double> > dof_val_tet_lag;
    dof_tet_lag->evaluate (&fe_tet_lag, dof_ids_tet_lag, dof_val_tet_lag);
    
    for (size_t i = 0; i < dof_val_tet_lag.size(); ++i)
    {
      for (size_t j = 0; j < dof_val_tet_lag[i].size(); ++j)
      {
        double val = dof_val_tet_lag[i][j];
        if (i != j)
        {
          BOOST_TEST(std::abs(val) < 1.e-10);
        }
        else
        {
          BOOST_TEST(std::abs(val - 1.0)< 1.e-10);
        }
      }
    }
    std::cout << "tet  lag(" << deg << ") test: passed " << std::endl;

    /// Test Lagrange TRIANGLE
    std::vector< cDofId > dof_ids_tri_lag  (dof_tri_lag->nb_dof_on_cell(),0);
    for (size_t i = 0; i < dof_tri_lag->nb_dof_on_cell(); ++i)
    {
      dof_ids_tri_lag[i] = i;
    }
    std::vector< std::vector<double> > dof_val_tri_lag;
    dof_tri_lag->evaluate (&fe_tri_lag, dof_ids_tri_lag, dof_val_tri_lag);
    
    for (size_t i = 0; i < dof_val_tri_lag.size(); ++i)
    {
      for (size_t j = 0; j < dof_val_tri_lag[i].size(); ++j)
      {
        double val = dof_val_tri_lag[i][j];
        if (i != j)
        {
          BOOST_TEST(std::abs(val) < 1.e-10);
        }
        else
        {
          BOOST_TEST(std::abs(val - 1.0)< 1.e-10);
        }
      }
    }
    CONSOLE_OUTPUT(rank, "tri  lag(" << deg << ") test: passed ");

    /// Test Lagrange QUADRILATERAL

    std::vector< cDofId > dof_ids_quad_lag (dof_quad_lag->nb_dof_on_cell(),0);
    for (size_t i = 0; i < dof_quad_lag->nb_dof_on_cell(); ++i)
    {
      dof_ids_quad_lag[i] = i;
    }
    std::vector< std::vector<double> > dof_val_quad_lag;
    dof_quad_lag->evaluate (&fe_quad_lag, dof_ids_quad_lag, dof_val_quad_lag);
    
    for (size_t i = 0; i < dof_val_quad_lag.size(); ++i)
    {
      for (size_t j = 0; j < dof_val_quad_lag[i].size(); ++j)
      {
        double val = dof_val_quad_lag[i][j];
        if (i != j)
        {
          BOOST_TEST(std::abs(val) < 1.e-10);
        }
        else
        {
          BOOST_TEST(std::abs(val - 1.0)< 1.e-10);
        }
      }
    }
    std::cout << "quad lag(" << deg << ") test: passed " << std::endl;
 
    /// Test Lagrange LINE
    std::vector< std::vector<double> > dof_val_line_lag;
    std::vector< cDofId > dof_ids_line_lag (dof_line_lag->nb_dof_on_cell(),0);
    for (size_t i = 0; i < dof_line_lag->nb_dof_on_cell(); ++i)
    {
      dof_ids_line_lag[i] = i;
    }
    dof_line_lag->evaluate (&fe_line_lag, dof_ids_line_lag, dof_val_line_lag);
    
    for (size_t i = 0; i < dof_val_line_lag.size(); ++i)
    {
      for (size_t j = 0; j < dof_val_line_lag[i].size(); ++j)
      {
        double val = dof_val_line_lag[i][j];
        if (i != j)
        {
          BOOST_TEST(std::abs(val) < 1.e-10);
        }
        else
        {
          BOOST_TEST(std::abs(val - 1.0) < 1.e-10);
        }
      }
    }
    CONSOLE_OUTPUT(rank, "line lag(" << deg << ") test: passed ");


    /// Test Lagrange PYRAMID

    std::vector< cDofId > dof_ids_pyr_lag  (dof_pyr_lag->nb_dof_on_cell(),0);
    for (size_t i = 0; i < dof_pyr_lag->nb_dof_on_cell(); ++i)
    {
      dof_ids_pyr_lag[i] = i;
    }
    std::vector< std::vector<double> > dof_val_pyr_lag;
    dof_pyr_lag->evaluate (&fe_pyr_lag, dof_ids_pyr_lag, dof_val_pyr_lag);
    
    for (size_t i = 0; i < dof_val_pyr_lag.size(); ++i)
    {
      for (size_t j = 0; j < dof_val_pyr_lag[i].size(); ++j)
      {
        double val = dof_val_pyr_lag[i][j];
        if (i != j)
        {
          BOOST_TEST(std::abs(val) < 1.e-10);
        }
        else
        {
          BOOST_TEST(std::abs(val - 1.0) < 1.e-10);
        }
      }
    }
    CONSOLE_OUTPUT(rank, "pyr lag(" << deg << ") test: passed ");

#endif
    
  
    /// Test BDM TRIANGLE
    if (deg >= min_bdm_deg && deg <= max_bdm_deg)
    {
      std::vector< cDofId > dof_ids_tri_bdm;
      dof_ids_tri_bdm.resize(dof_tri_bdm->nb_dof_on_cell(), 0);
      for (size_t i = 0; i < dof_tri_bdm->nb_dof_on_cell(); ++i)
      {
        dof_ids_tri_bdm[i] = i;
      }
      std::vector< std::vector<double> > dof_val_tri_bdm;
      dof_tri_bdm->evaluate (&fe_tri_bdm, dof_ids_tri_bdm, dof_val_tri_bdm);
      
      for (size_t i = 0; i < dof_val_tri_bdm.size(); ++i)
      {
        for (size_t j = 0; j < dof_val_tri_bdm[i].size(); ++j)
        {
          double val = dof_val_tri_bdm[i][j];
          if (i != j)
          {
            BOOST_TEST(std::abs(val) < 1.e-10);
          }
          else
          {
            BOOST_TEST(std::abs(val - 1.0) < 1.e-10);
          }
        }
      }
      CONSOLE_OUTPUT(rank, "tri  BDM(" << deg << ") test: passed ");
    }
    
    /// Test RT TRIANGLE
    if (deg >= min_rt_deg && deg <= max_rt_deg)
    {
      std::vector< cDofId > dof_ids_tri_rt;
      dof_ids_tri_rt.resize(dof_tri_rt->nb_dof_on_cell(), 0);
      for (size_t i = 0; i < dof_tri_rt->nb_dof_on_cell(); ++i)
      {
        dof_ids_tri_rt[i] = i;
      }
      std::vector< std::vector<double> > dof_val_tri_rt;
      dof_tri_rt->evaluate (&fe_tri_rt, dof_ids_tri_rt, dof_val_tri_rt);
        
      for (size_t i = 0; i < dof_val_tri_rt.size(); ++i)
      {
        for (size_t j = 0; j < dof_val_tri_rt[i].size(); ++j)
        {
          double val = dof_val_tri_rt[i][j];
          if (i != j)
          {
            BOOST_TEST(std::abs(val) < 1.e-10);
          }
          else
          {
            BOOST_TEST(std::abs(val - 1.0) < 1.e-10);
          }
        }
      }
      CONSOLE_OUTPUT(rank, "tri  RT(" << deg << ")  test: passed ");
    }

    /// Test RT TETRAHEDRON
    if (deg >= min_rt_deg && deg <= max_rt_deg)
    {
      std::vector< cDofId > dof_ids_tet_rt;
      dof_ids_tet_rt.resize(dof_tet_rt->nb_dof_on_cell(), 0);
      for (size_t i = 0; i < dof_tet_rt->nb_dof_on_cell(); ++i)
      {
        dof_ids_tet_rt[i] = i;
      }
      std::vector< std::vector<double> > dof_val_tet_rt;
      dof_tet_rt->evaluate (&fe_tet_rt, dof_ids_tet_rt, dof_val_tet_rt);
        
      for (size_t i = 0; i < dof_val_tet_rt.size(); ++i)
      {
        for (size_t j = 0; j < dof_val_tet_rt[i].size(); ++j)
        {
          double val = dof_val_tet_rt[i][j];
          if (i != j)
          {
            BOOST_TEST(std::abs(val) < 1.e-10);
          }
          else
          {
            BOOST_TEST(std::abs(val - 1.0) < 1.e-10);
          }
        }
      }
      CONSOLE_OUTPUT(rank, "tet  RT(" << deg << ")  test: passed ");
    }

  /// Test RT QUADRILATERAL
  if (deg >= min_rt_deg && deg <= max_rt_deg)
  {
    std::vector< cDofId > dof_ids_quad_rt (dof_quad_rt->nb_dof_on_cell(),0);
    for (size_t i = 0; i < dof_quad_rt->nb_dof_on_cell(); ++i)
    {
      dof_ids_quad_rt[i] = i;
    }
    std::vector< std::vector<double> > dof_val_quad_rt;
    dof_quad_rt->evaluate (&fe_quad_rt, dof_ids_quad_rt, dof_val_quad_rt);
    
    for (size_t i = 0; i < dof_val_quad_rt.size(); ++i)
    {
      for (size_t j = 0; j < dof_val_quad_rt[i].size(); ++j)
      {
        double val = dof_val_quad_rt[i][j];
        if (i != j)
        {
          BOOST_TEST(std::abs(val) < 1.e-10);
        }
        else
        {
          BOOST_TEST(std::abs(val - 1.0)< 1.e-10);
        }
      }
    }
    CONSOLE_OUTPUT(rank, "quad  RT(" << deg << ")  test: passed ");
    }

  /// Test RT HEXAHEDRON
  if (deg >= min_rt_deg && deg <= max_rt_deg)
  {
    std::vector< cDofId > dof_ids_hex_rt  (dof_hex_rt->nb_dof_on_cell(),0);
    for (size_t i = 0; i < dof_hex_rt->nb_dof_on_cell(); ++i)
    {
      dof_ids_hex_rt[i] = i;
    }
    std::vector< std::vector<double> > dof_val_hex_rt;
    dof_hex_rt->evaluate (&fe_hex_rt, dof_ids_hex_rt, dof_val_hex_rt);

    for (size_t i = 0; i < dof_val_hex_rt.size(); ++i)
    {
      for (size_t j = 0; j < dof_val_hex_rt[i].size(); ++j)
      {
        double val = dof_val_hex_rt[i][j];
        if (i != j)
        {
          BOOST_TEST(std::abs(val) < 1.e-10);
        }
        else
        {
          BOOST_TEST(std::abs(val - 1.0) < 1.e-10);
        }
      }
    }
    CONSOLE_OUTPUT(rank, "hex  rt(" << deg << ") test: passed ");
    }
  }

  MPI_Finalize();
  return;
}
