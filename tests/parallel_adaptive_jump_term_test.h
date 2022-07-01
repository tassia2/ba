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

#define BOOST_TEST_MODULE parallel_adaptive_jump_term

#include <boost/test/unit_test.hpp>
#include <boost/test/included/unit_test.hpp>

// System includes.
#include "hiflow.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include <mpi.h>
#include <string>
#include <vector>

// Dimension of the problem.
const int DIMENSION = 2;

// All names are imported for simplicity.
using namespace hiflow;
using namespace hiflow::doffem;
using namespace hiflow::la;
using namespace hiflow::mesh;

// Shorten some datatypes with typedefs.
typedef LADescriptorCoupledD LAD;
typedef LAD::DataType Scalar;
typedef LAD::VectorType CVector;
typedef LAD::MatrixType CMatrix;

typedef Vec<DIMENSION, double > Coord;

// Rank of the master process.
const int MASTER_RANK = 0;



// Choose mesh implementation
#define USE_MESH_P4EST

#ifndef WITH_P4EST
#undef USE_MESH_P4EST
#endif

struct QuadratureSelection {
  /// Constructor
  /// \param[in] order Desired order of quadrature rule

  QuadratureSelection(const int order) : order_(order) {}

  /// Operator to obtain quadrature rule on given element
  /// \param[in] elem Element on which quadrature should be done
  /// \param[out] quadrature Quadrature rule on given Element elem

  void operator()(const Element< double, DIMENSION > &elem,
                  Quadrature< double > &quadrature) {
    // Get ID of FE type
    const mesh::CellType::Tag fe_id =
        elem.get_cell().cell_type().tag();
       // elem.aaa_get_fe(0)->get_my_type();
    // Switch by FE type for quadrature selection
    switch (fe_id) {
    case  mesh::CellType::Tag::TRIANGLE: {
      quadrature.set_cell_tag(mesh::CellType::Tag::TRIANGLE);
      quadrature.set_quadrature_by_order("GaussTriangle", order_);
      break;
    }
    case  mesh::CellType::Tag::QUADRILATERAL: {
      quadrature.set_cell_tag(mesh::CellType::Tag::QUADRILATERAL);
      quadrature.set_quadrature_by_order("EconomicalGaussQuadrilateral",
                                         order_);
      break;
    }
    case  mesh::CellType::Tag::TETRAHEDRON: {
      quadrature.set_cell_tag(mesh::CellType::Tag::TETRAHEDRON);
      quadrature.set_quadrature_by_order("EconomicalGaussTetrahedron", order_);
      break;
    }
    case  mesh::CellType::Tag::HEXAHEDRON: {
      quadrature.set_cell_tag(mesh::CellType::Tag::HEXAHEDRON);
      quadrature.set_quadrature_by_order("GaussHexahedron", order_);
      break;
    }
    default: { assert(false); }
    };
  }
  // Order of quadrature
  const int order_;
};

// Jump Term Assembler for the jumps over the inner edges

class JumpTermAssembler : private DGAssemblyAssistant< DIMENSION, double > {
public:
  JumpTermAssembler(CoupledVector< Scalar > &u_h) : u_h_(u_h) {}

  void operator()(const Element< double, DIMENSION > &left_elem,
                  const Element< double, DIMENSION > &right_elem,
                  const Quadrature< double > &left_quad,
                  const Quadrature< double > &right_quad, int left_facet_number,
                  int right_facet_number, InterfaceSide left_if_side,
                  InterfaceSide right_if_side, double &local_val) {
    const bool is_boundary =
        (right_if_side == DGGlobalAssembler< double, DIMENSION >::InterfaceSide::BOUNDARY);

    if (is_boundary || (left_if_side == right_if_side)) {
      local_val = 0.;
      return;
    }

    this->initialize_for_interface(left_elem, right_elem, left_quad, right_quad,
                                   left_facet_number, right_facet_number,
                                   left_if_side, right_if_side);

    left_grad_u_h.clear();
    right_grad_u_h.clear();

    this->trial().evaluate_fe_function_gradients(u_h_, 0, left_grad_u_h);
    this->test().evaluate_fe_function_gradients(u_h_, 0, right_grad_u_h);
    const int num_q = num_quadrature_points();

    double h_E = 0.;
    std::vector< double > y_coord;
    for (int i = 0; i < num_q; ++i) {
      for (int j = 0; j < num_q; ++j) {
        h_E = std::max(h_E, norm(this->x(i) - this->x(j)));
      }
      y_coord.push_back(this->x(i)[1]);
    }

    // Loop over quadrature points on each edge
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    LOG_DEBUG(2, "[" << rank << "] trial cell = " << left_elem.cell_index()
                     << " test cell = " << right_elem.cell_index()
                     << " trial normal = " << this->trial().n(0)[0] << " , "
                     << this->trial().n(0)[1] << " test normal = "
                     << this->test().n(0)[0] << " , " << this->test().n(0)[1]
                     << " trial grad = " << left_grad_u_h[0][0] << " , "
                     << left_grad_u_h[0][1]
                     << " test grad = " << right_grad_u_h[0][0] << " , "
                     << right_grad_u_h[0][1]
                     //<< " if lenght = " << h_E
                     << " "
                     << string_from_range(y_coord.begin(), y_coord.end()));

    for (int q = 0.; q < num_q; ++q) {
      const double dS = std::abs(this->ds(q));
      local_val += this->w(q)

                   * std::abs(dot(this->trial().n(q), left_grad_u_h[q]) +
                              dot(this->test().n(q), right_grad_u_h[q])) *
                   std::abs(dot(this->trial().n(q), left_grad_u_h[q]) +
                            dot(this->test().n(q), right_grad_u_h[q])) *
                   dS;
    }
  }
  const CoupledVector< Scalar > &u_h_;
  FunctionValues< Vec< DIMENSION, double > > left_grad_u_h;
  FunctionValues< Vec< DIMENSION, double > > right_grad_u_h;
};
