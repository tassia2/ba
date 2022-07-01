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

/// \author Michael Schick

#define BOOST_TEST_MODULE fe_weight_zero

#include <boost/test/unit_test.hpp>
#include <boost/test/included/unit_test.hpp>
namespace utf = boost::unit_test;

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

#include "common/macros.h"

#include "../src/fem/felagrange_hex.h"
#include "../src/fem/felagrange_line.h"
#include "../src/fem/felagrange_quad.h"
#include "../src/fem/felagrange_tet.h"
#include "../src/fem/felagrange_tri.h"

using namespace hiflow::doffem;

<<<<<<< HEAD:tests/old/fe_weight_zero_test.cc
typedef hiflow::Vec<1, double> Coord1;
typedef hiflow::Vec<2, double> Coord2;
typedef hiflow::Vec<3, double> Coord3;


int main(int argc, char **argv) {
=======
BOOST_AUTO_TEST_CASE(fe_weight_zero, *utf::tolerance(1.0e-10)) {
>>>>>>> master:tests/fe_weight_zero_test.cc
  // Test if shapefunctions are zero on dofs, which correspond to other
  // shapefunctions

  for (int deg = 1; deg < 8; ++deg) {
    std::cout << "Degree: " << deg << std::endl;

    FELagrangeLine< double, 1 > fe_line;
    FELagrangeHex< double, 3 > fe_hex;
    FELagrangeTet< double, 3 > fe_tet;
    FELagrangeTri< double, 2 > fe_tri;
    FELagrangeQuad< double, 2 > fe_quad;

    fe_line.set_my_deg(deg);
    fe_hex.set_my_deg(deg);
    fe_tet.set_my_deg(deg);
    fe_tri.set_my_deg(deg);
    fe_quad.set_my_deg(deg);

    fe_line.init();
    fe_hex.init();
    fe_tet.init();
    fe_tri.init();
    fe_quad.init();

    // Test HEXAHEDRON

    int nb_dof_line = deg + 1;

    for (int k = 0; k < nb_dof_line; ++k)
      for (int j = 0; j < nb_dof_line; ++j)
        for (int i = 0; i < nb_dof_line; ++i) {
          int shape_fct_index =
            (i + j * nb_dof_line + k * nb_dof_line * nb_dof_line);

          Coord3 coord;

          double h = 1.0 / deg;

          coord[0] = i * h;
          coord[1] = j * h;
          coord[2] = k * h;

          std::vector< double > weight(fe_hex.get_nb_dof_on_cell());

          fe_hex.N(coord, weight);

          for (int w = 0; w < static_cast< int >(weight.size()); ++w)
            if (w != shape_fct_index)
              BOOST_TEST(weight[w] == 0.0);
        }

    // Test TETRAHEDRON
    nb_dof_line = deg + 1;
    int offset = 0;

    for (int k = 0; k < nb_dof_line; ++k)
      for (int j = 0; j < nb_dof_line - k; ++j) {
        for (int i = 0; i < nb_dof_line - k - j; ++i) {
          int shape_fct_index = i + offset;

          Coord3 coord;

          double h = 1.0 / deg;

          coord[2] = h * k;
          coord[1] = h * j;
          coord[0] = h * i;

          std::vector< double > weight(fe_tet.get_nb_dof_on_cell());

          fe_tet.N(coord, weight);

          for (int w = 0; w < static_cast< int >(weight.size()); ++w)
            if (w != shape_fct_index)
              BOOST_TEST(weight[w] == 0.0);
        }
        offset += nb_dof_line - k - j;
      }

    // Test TRIANGLE

    nb_dof_line = deg + 1;
    offset = 0;

    for (int j = 0; j < nb_dof_line; ++j) {
      for (int i = 0; i < nb_dof_line - j; ++i) {
        int shape_fct_index = i + offset;

        double h = 1.0 / deg;

        Coord2 coord;

        coord[1] = h * j;
        coord[0] = h * i;

        std::vector< double > weight(fe_tri.get_nb_dof_on_cell());

        fe_tri.N(coord, weight);

        for (int w = 0; w < static_cast< int >(weight.size()); ++w)
          if (w != shape_fct_index)
            BOOST_TEST(weight[w] == 0.0);
      }
      offset += nb_dof_line - j;
    }

    // Test QUADRILATERAL

    nb_dof_line = deg + 1;

    for (int j = 0; j < nb_dof_line; ++j)
      for (int i = 0; i < nb_dof_line; ++i) {
        int shape_fct_index = i + j * nb_dof_line;

        Coord2 coord;

        double h = 1.0 / deg;

        coord[1] = h * j;
        coord[0] = h * i;

        std::vector< double > weight(fe_quad.get_nb_dof_on_cell());

        fe_quad.N(coord, weight);

        for (int w = 0; w < static_cast< int >(weight.size()); ++w)
          if (w != shape_fct_index)
            BOOST_TEST(weight[w] == 0.0);
      }

    // Test LINE

    nb_dof_line = deg + 1;

    for (int j = 0; j < nb_dof_line; ++j) {
      int shape_fct_index = j;

      Coord1 coord;

      double h = 1.0 / deg;

      coord[0] = h * j;

      std::vector< double > weight(fe_line.get_nb_dof_on_cell());

      fe_line.N(coord, weight);

      for (int w = 0; w < static_cast< int >(weight.size()); ++w)
        if (w != shape_fct_index)
          BOOST_TEST(weight[w] == 0.0);
    }
  }

}
