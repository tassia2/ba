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

/// \author Michael Schick, Philipp Gerstner
/// TODO: uncomment missing elements

#define BOOST_TEST_MODULE fe_weight

#include <boost/test/unit_test.hpp>
#include <boost/test/included/unit_test.hpp>
namespace utf = boost::unit_test;

#include <mpi.h>

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

#include "common/macros.h"
#include "common/parcom.h"
#include "fem/reference_cell.h"
#include "fem/fe_reference.h"
#include "fem/fe_transformation.h"

#include "fem/ansatz/ansatz_p_line_lagrange.h"
#include "fem/ansatz/ansatz_p_tri_lagrange.h"
#include "fem/ansatz/ansatz_p_tet_lagrange.h"
#include "fem/ansatz/ansatz_q_quad_lagrange.h"
#include "fem/ansatz/ansatz_q_hex_lagrange.h"
#include "fem/ansatz/ansatz_pyr_lagrange.h"

#include "dof/dof_impl/dof_container_lagrange.h"

#include "test.h"

using namespace hiflow;
using namespace hiflow::doffem;

BOOST_AUTO_TEST_CASE(fe_weight, *utf::tolerance(1.0e-9)) {
  // Test if sum of weights is equal to one
  // and if sum of derivatives is equal to zero

  int argc = boost::unit_test::framework::master_test_suite().argc;
  char** argv = boost::unit_test::framework::master_test_suite().argv;
  
  int rank;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD,  &rank);
  
  typedef hiflow::Vec<3, double> Coord3;
  typedef hiflow::Vec<2, double> Coord2;
  typedef hiflow::Vec<1, double> Coord1;

  int nb_3d_pts_line = 11;
  int nb_2d_pts_line = 11;
  int nb_1d_pts_line = 11;

  double h_3d = 1.0 / (nb_3d_pts_line - 1.);
  double h_2d = 1.0 / (nb_2d_pts_line - 1.);
  // double h_1d = 1.0/(nb_1d_pts_line - 1.);

  // create reference cells 
  CRefCellSPtr<double, 1> ref_cell_line = CRefCellSPtr<double, 1> ( new RefCellLineStd<double, 1>);
  CRefCellSPtr<double, 2> ref_cell_tri = CRefCellSPtr<double, 2> ( new RefCellTriStd <double, 2>);
  CRefCellSPtr<double, 2> ref_cell_quad = CRefCellSPtr<double, 2> ( new RefCellQuadStd<double, 2>);
  CRefCellSPtr<double, 3> ref_cell_tet = CRefCellSPtr<double, 3> ( new RefCellTetStd <double, 3>);
  CRefCellSPtr<double, 3> ref_cell_hex = CRefCellSPtr<double, 3> ( new RefCellHexStd <double, 3>);
  CRefCellSPtr<double, 3> ref_cell_pyr = CRefCellSPtr<double, 3> ( new RefCellPyrStd <double, 3>);

  // create ansatz spaces
  AnsatzSpaceSPtr<double, 1> ansatz_line_lag (new PLineLag<double, 1> (ref_cell_line));
  AnsatzSpaceSPtr<double, 2> ansatz_tri_lag (new PTriLag<double, 2> (ref_cell_tri));
  AnsatzSpaceSPtr<double, 3> ansatz_tet_lag (new PTetLag<double, 3> (ref_cell_tet));
  AnsatzSpaceSPtr<double, 2> ansatz_quad_lag (new QQuadLag<double, 2> (ref_cell_quad));
  AnsatzSpaceSPtr<double, 3> ansatz_hex_lag (new QHexLag<double, 3> (ref_cell_hex));
  AnsatzSpaceSPtr<double, 3> ansatz_pyr_lag (new PyrLag<double, 3> (ref_cell_pyr));

  // create dof containers
  DofContainerLagSPtr<double, 1> dof_line_lag (new DofContainerLagrange<double, 1> (ref_cell_line));
  DofContainerLagSPtr<double, 2> dof_tri_lag  (new DofContainerLagrange<double, 2> (ref_cell_tri));
  DofContainerLagSPtr<double, 3> dof_tet_lag  (new DofContainerLagrange<double, 3> (ref_cell_tet));
  DofContainerLagSPtr<double, 2> dof_quad_lag (new DofContainerLagrange<double, 2> (ref_cell_quad));
  DofContainerLagSPtr<double, 3> dof_hex_lag  (new DofContainerLagrange<double, 3> (ref_cell_hex));
  DofContainerLagSPtr<double, 3> dof_pyr_lag  (new DofContainerLagrange<double, 3> (ref_cell_pyr));
  
  for (int deg = 0; deg < 6; ++deg) 
  {
    
    // initialize reference elements
    bool modal_basis = false;
    ansatz_line_lag->init(deg, 1);
    dof_line_lag->init(deg, 1);
    CONSOLE_OUTPUT(rank, "initialize Lagrange line element of degree " << deg << " with " << dof_line_lag->nb_dof_on_cell() << " dofs and " 
              << ansatz_line_lag->dim() << " ansatz functions ");
    RefElement<double, 1> fe_line_lag;
    FETrafoSPtr<double, 1> fe_trafo_line_lag (new FETransformationStandard<double, 1> );
    fe_line_lag.init(ansatz_line_lag,dof_line_lag,fe_trafo_line_lag,modal_basis, FEType::LAGRANGE);
    
    ansatz_tri_lag->init(deg, 1);
    dof_tri_lag->init(deg, 1);
    CONSOLE_OUTPUT(rank, "initialize Lagrange tri  element of degree " << deg << " with " << dof_tri_lag->nb_dof_on_cell() << " dofs and " 
              << ansatz_tri_lag->dim() << " ansatz functions ");
    RefElement<double, 2> fe_tri_lag;
    FETrafoSPtr<double, 2> fe_trafo_tri_lag (new FETransformationStandard<double, 2> );
    fe_tri_lag.init (ansatz_tri_lag, dof_tri_lag, fe_trafo_tri_lag, modal_basis, FEType::LAGRANGE);
    
    ansatz_tet_lag->init(deg, 1);
    dof_tet_lag->init(deg, 1);
    std::cout << "initialize Lagrange tetrahedron element with " << dof_tet_lag->nb_dof_on_cell() << " dofs and " 
              << ansatz_tet_lag->dim() << " ansatz functions " << std::endl;
    RefElement<double, 3> fe_tet_lag;
    FETrafoSPtr<double, 3> fe_trafo_tet_lag (new FETransformationStandard<double, 3> );
    fe_tet_lag.init (ansatz_tet_lag, dof_tet_lag, fe_trafo_tet_lag, modal_basis, FEType::LAGRANGE);


    ansatz_quad_lag->init(deg, 1);
    dof_quad_lag->init(deg, 1);
    std::cout << "initialize Lagrange quad element with " << dof_quad_lag->nb_dof_on_cell() << " dofs and " 
              << ansatz_quad_lag->dim() << " ansatz functions " << std::endl;
    RefElement<double, 2> fe_quad_lag;
    FETrafoSPtr<double, 2> fe_trafo_quad_lag (new FETransformationStandard<double, 2> );
    fe_quad_lag.init(ansatz_quad_lag,dof_quad_lag,fe_trafo_quad_lag,modal_basis, FEType::LAGRANGE);

    ansatz_hex_lag->init(deg, 1);
    dof_hex_lag->init(deg, 1);
    CONSOLE_OUTPUT(rank, "initialize Lagrange hex  element of degree " << deg << " with " << dof_hex_lag->nb_dof_on_cell() << " dofs and " 
              << ansatz_hex_lag->dim() << " ansatz functions ");
    RefElement<double, 3> fe_hex_lag;
    FETrafoSPtr<double, 3> fe_trafo_hex_lag (new FETransformationStandard<double, 3> );
    fe_hex_lag.init (ansatz_hex_lag, dof_hex_lag, fe_trafo_hex_lag, modal_basis, FEType::LAGRANGE);

    size_t pyr_deg = std::min(2, deg);
    ansatz_pyr_lag->init(pyr_deg);
    dof_pyr_lag->init(pyr_deg, 1);
    
    CONSOLE_OUTPUT(rank, "initialize Lagrange pyramid element with " << dof_pyr_lag->nb_dof_on_cell() << " dofs and " 
              << ansatz_pyr_lag->dim() << " ansatz functions ");
    RefElement<double, 3> fe_pyr_lag;
    FETrafoSPtr<double, 3> fe_trafo_pyr_lag (new FETransformationStandard<double, 3> );
    fe_pyr_lag.init (ansatz_pyr_lag,dof_pyr_lag, fe_trafo_pyr_lag, modal_basis, FEType::LAGRANGE);

    // Test HEXAHEDRON
    for (int k = 0; k < nb_3d_pts_line; ++k)
    {
      for (int j = 0; j < nb_3d_pts_line; ++j)
      {
        for (int i = 0; i < nb_3d_pts_line; ++i) 
        {
          Coord3 coord;

          coord.set(0, h_3d * i);
          coord.set(1, h_3d * j);
          coord.set(2, h_3d * k);

          assert(coord[0] >= 0.0 && coord[0] <= 1.0);
          assert(coord[1] >= 0.0 && coord[1] <= 1.0);
          assert(coord[2] >= 0.0 && coord[2] <= 1.0);

          std::vector< double > weight   (fe_hex_lag.nb_dof_on_cell());
          std::vector< double > weight_x (fe_hex_lag.nb_dof_on_cell());
          std::vector< double > weight_y (fe_hex_lag.nb_dof_on_cell());
          std::vector< double > weight_z (fe_hex_lag.nb_dof_on_cell());
          std::vector< double > weight_xx(fe_hex_lag.nb_dof_on_cell());
          std::vector< double > weight_xy(fe_hex_lag.nb_dof_on_cell());
          std::vector< double > weight_xz(fe_hex_lag.nb_dof_on_cell());
          std::vector< double > weight_yy(fe_hex_lag.nb_dof_on_cell());
          std::vector< double > weight_yz(fe_hex_lag.nb_dof_on_cell());
          std::vector< double > weight_zz(fe_hex_lag.nb_dof_on_cell());

          fe_hex_lag.N   (coord, weight);
          fe_hex_lag.N_x (coord, weight_x);
          fe_hex_lag.N_y (coord, weight_y);
          fe_hex_lag.N_z (coord, weight_z);
          fe_hex_lag.N_xx(coord, weight_xx);
          fe_hex_lag.N_xy(coord, weight_xy);
          fe_hex_lag.N_xz(coord, weight_xz);
          fe_hex_lag.N_yy(coord, weight_yy);
          fe_hex_lag.N_yz(coord, weight_yz);
          fe_hex_lag.N_zz(coord, weight_zz);

          // Check
          double sum = 0.0;
          double sum_x = 0.0;
          double sum_y = 0.0;
          double sum_z = 0.0;
          double sum_xx = 0.0;
          double sum_xy = 0.0;
          double sum_xz = 0.0;
          double sum_yy = 0.0;
          double sum_yz = 0.0;
          double sum_zz = 0.0;

          for (int w = 0; w < static_cast< int >(weight.size()); ++w) 
          {
            sum += weight[w];
            sum_x += weight_x[w];
            sum_y += weight_y[w];
            sum_z += weight_z[w];
            sum_xx += weight_xx[w];
            sum_xy += weight_xy[w];
            sum_xz += weight_xz[w];
            sum_yy += weight_yy[w];
            sum_yz += weight_yz[w];
            sum_zz += weight_zz[w];
          }

          BOOST_TEST(sum == 1.0);
          BOOST_TEST(sum_x == 0.0);
          BOOST_TEST(sum_y == 0.0);
          BOOST_TEST(sum_z == 0.0);
          BOOST_TEST(sum_xx == 0.0);
          BOOST_TEST(sum_xy == 0.0);
          BOOST_TEST(sum_xz == 0.0);
          BOOST_TEST(sum_yy == 0.0);
          BOOST_TEST(sum_yz == 0.0);
          BOOST_TEST(sum_zz == 0.0);
        }
      }
    }

    CONSOLE_OUTPUT(rank, "hex  lag test: passed " );
    // Test TETRAHEDRON

    for (int k = 0; k < nb_3d_pts_line; ++k)
    {
      for (int j = 0; j < nb_3d_pts_line - k; ++j)
      {
        for (int i = 0; i < nb_3d_pts_line - k - j; ++i) 
        {
          Coord3 coord;

          coord.set(2, h_3d * k);
          coord.set(1, (1.0 - coord[2]) / (nb_3d_pts_line - 1.) * j);
          coord.set(0, (1.0 - coord[2] - coord[1]) / (nb_3d_pts_line - 1.) * i);

          assert(coord[0] >= 0.0 && coord[0] <= (1.0 - coord[1] - coord[2]));
          assert(coord[1] >= 0.0 && coord[1] <= (1.0 - coord[2]));
          assert(coord[2] >= 0.0 && coord[2] <= 1.0);

          std::vector< double > weight(fe_tet_lag.nb_dof_on_cell());
          std::vector< double > weight_x(fe_tet_lag.nb_dof_on_cell());
          std::vector< double > weight_y(fe_tet_lag.nb_dof_on_cell());
          std::vector< double > weight_z(fe_tet_lag.nb_dof_on_cell());
          std::vector< double > weight_xx(fe_tet_lag.nb_dof_on_cell());
          std::vector< double > weight_xy(fe_tet_lag.nb_dof_on_cell());
          std::vector< double > weight_xz(fe_tet_lag.nb_dof_on_cell());
          std::vector< double > weight_yy(fe_tet_lag.nb_dof_on_cell());
          std::vector< double > weight_yz(fe_tet_lag.nb_dof_on_cell());
          std::vector< double > weight_zz(fe_tet_lag.nb_dof_on_cell());

          fe_tet_lag.N(coord, weight);
          fe_tet_lag.N_x(coord, weight_x);
          fe_tet_lag.N_y(coord, weight_y);
          fe_tet_lag.N_z(coord, weight_z);
          fe_tet_lag.N_xx(coord, weight_xx);
          fe_tet_lag.N_xy(coord, weight_xy);
          fe_tet_lag.N_xz(coord, weight_xz);
          fe_tet_lag.N_yy(coord, weight_yy);
          fe_tet_lag.N_yz(coord, weight_yz);
          fe_tet_lag.N_zz(coord, weight_zz);

          // Check
          double sum = 0.0;
          double sum_x = 0.0;
          double sum_y = 0.0;
          double sum_z = 0.0;
          double sum_xx = 0.0;
          double sum_xy = 0.0;
          double sum_xz = 0.0;
          double sum_yy = 0.0;
          double sum_yz = 0.0;
          double sum_zz = 0.0;

          for (int w = 0; w < static_cast< int >(weight.size()); ++w) {
            sum += weight[w];
            sum_x += weight_x[w];
            sum_y += weight_y[w];
            sum_z += weight_z[w];
            sum_xx += weight_xx[w];
            sum_xy += weight_xy[w];
            sum_xz += weight_xz[w];
            sum_yy += weight_yy[w];
            sum_yz += weight_yz[w];
            sum_zz += weight_zz[w];
          }

          if (sum_z > 0.1) {
            std::cout << "Degree: " << deg << std::endl;
            std::cout << coord[0] << " " << coord[1] << "  " << coord[2]
                      << std::endl;
            for (int w = 0; w < static_cast< int >(weight_z.size()); ++w)
              std::cout << weight_z[w] << "  ";

            std::cout << std::endl;
          }

          BOOST_TEST(sum == 1.0);
          BOOST_TEST(sum_x == 0.0);
          BOOST_TEST(sum_y == 0.0);
          BOOST_TEST(sum_z == 0.0);
          BOOST_TEST(sum_xx == 0.0);
          BOOST_TEST(sum_xy == 0.0);
          BOOST_TEST(sum_xz == 0.0);
          BOOST_TEST(sum_yy == 0.0);
          BOOST_TEST(sum_yz == 0.0);
          BOOST_TEST(sum_zz == 0.0);
        }
      }
    }
    CONSOLE_OUTPUT(rank, "tet  lag test: passed ");

    // Test TRIANGLE

    for (int j = 0; j < nb_2d_pts_line; ++j)
    {
      for (int i = 0; i < nb_2d_pts_line - j; ++i) 
      {
        Coord2 coord;

        coord.set(1, h_2d * j);
        coord.set(0, (1.0 - coord[1]) / (nb_2d_pts_line - 1.) * i);

        assert(coord[0] >= 0.0 && coord[0] <= (1.0 - coord[1]));
        assert(coord[1] >= 0.0 && coord[1] <= 1.0);

        std::vector< double > weight(fe_tri_lag.nb_dof_on_cell());
        std::vector< double > weight_x(fe_tri_lag.nb_dof_on_cell());
        std::vector< double > weight_y(fe_tri_lag.nb_dof_on_cell());
        std::vector< double > weight_xx(fe_tri_lag.nb_dof_on_cell());
        std::vector< double > weight_xy(fe_tri_lag.nb_dof_on_cell());
        std::vector< double > weight_yy(fe_tri_lag.nb_dof_on_cell());

        fe_tri_lag.N(coord, weight);
        fe_tri_lag.N_x(coord, weight_x);
        fe_tri_lag.N_y(coord, weight_y);
        fe_tri_lag.N_xx(coord, weight_xx);
        fe_tri_lag.N_xy(coord, weight_xy);
        fe_tri_lag.N_yy(coord, weight_yy);

        // Check
        double sum = 0.0;
        double sum_x = 0.0;
        double sum_y = 0.0;
        double sum_xx = 0.0;
        double sum_xy = 0.0;
        double sum_yy = 0.0;

        for (int w = 0; w < static_cast< int >(weight.size()); ++w) {
          sum += weight[w];
          sum_x += weight_x[w];
          sum_y += weight_y[w];
          sum_xx += weight_xx[w];
          sum_xy += weight_xy[w];
          sum_yy += weight_yy[w];
        }

        BOOST_TEST(sum == 1.0);
        BOOST_TEST(sum_x == 0.0);
        BOOST_TEST(sum_y == 0.0);
        BOOST_TEST(sum_xx == 0.0);
        BOOST_TEST(sum_xy == 0.0);
        BOOST_TEST(sum_yy == 0.0);
      }
    }
    CONSOLE_OUTPUT(rank, "tri  lag test: passed ");

    // Test QUADRILATERAL
    for (int j = 0; j < nb_2d_pts_line; ++j)
    {
      for (int i = 0; i < nb_2d_pts_line; ++i) 
      {
        Coord2 coord;

        coord.set(1, h_2d * j);
        coord.set(0, h_2d * i);

        assert(coord[0] >= 0.0 && coord[0] <= 1.0);
        assert(coord[1] >= 0.0 && coord[1] <= 1.0);

        std::vector< double > weight(fe_quad_lag.nb_dof_on_cell());
        std::vector< double > weight_x(fe_quad_lag.nb_dof_on_cell());
        std::vector< double > weight_y(fe_quad_lag.nb_dof_on_cell());
        std::vector< double > weight_xx(fe_quad_lag.nb_dof_on_cell());
        std::vector< double > weight_xy(fe_quad_lag.nb_dof_on_cell());
        std::vector< double > weight_yy(fe_quad_lag.nb_dof_on_cell());

        fe_quad_lag.N(coord, weight);
        fe_quad_lag.N_x(coord, weight_x);
        fe_quad_lag.N_y(coord, weight_y);
        fe_quad_lag.N_xx(coord, weight_xx);
        fe_quad_lag.N_xy(coord, weight_xy);
        fe_quad_lag.N_yy(coord, weight_yy);

        // Check
        double sum = 0.0;
        double sum_x = 0.0;
        double sum_y = 0.0;
        double sum_xx = 0.0;
        double sum_xy = 0.0;
        double sum_yy = 0.0;

        for (int w = 0; w < static_cast< int >(weight.size()); ++w) {
          sum += weight[w];
          sum_x += weight_x[w];
          sum_y += weight_y[w];
          sum_xx += weight_xx[w];
          sum_xy += weight_xy[w];
          sum_yy += weight_yy[w];
        }

        BOOST_TEST(sum == 1.0);
        BOOST_TEST(sum_x == 0.0);
        BOOST_TEST(sum_y == 0.0);
        BOOST_TEST(sum_xx == 0.0);
        BOOST_TEST(sum_xy == 0.0);
        BOOST_TEST(sum_yy == 0.0);
      }
    }

    // Test LINE

    for (int i = 0; i < nb_1d_pts_line; ++i) 
    {
      Coord1 coord;

      coord.set(0, h_2d * i);

      assert(coord[0] >= 0.0 && coord[0] <= 1.0);

      std::vector< double > weight(fe_line_lag.nb_dof_on_cell());
      std::vector< double > weight_x(fe_line_lag.nb_dof_on_cell());
      std::vector< double > weight_xx(fe_line_lag.nb_dof_on_cell());

      fe_line_lag.N(coord, weight);
      fe_line_lag.N_x(coord, weight_x);
      fe_line_lag.N_xx(coord, weight_xx);

      // Check
      double sum = 0.0;
      double sum_x = 0.0;
      double sum_xx = 0.0;

      for (int w = 0; w < static_cast< int >(weight.size()); ++w) {
        sum += weight[w];
        sum_x += weight_x[w];
        sum_xx += weight_xx[w];
      }

      BOOST_TEST(sum == 1.0);
      BOOST_TEST(sum_x == 0.0);
      BOOST_TEST(sum_xx == 0.0);
    }
    CONSOLE_OUTPUT(rank, "line lag test: passed ");
  }
  MPI_Finalize();
}
