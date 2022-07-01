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

/// \author Staffan Ronnas

#define BOOST_TEST_MODULE attributes

#include <boost/test/unit_test.hpp>
#include <boost/test/included/unit_test.hpp>


#include <fstream>
#include <iostream>
#include <string>

#include "hiflow.h"

using namespace std;
using namespace hiflow;
using namespace hiflow::mesh;

const TDim tdim = 3;
const GDim gdim = 3;

static const char *datadir = MESH_DATADIR;

BOOST_AUTO_TEST_CASE(fe_weight_zero) {
  std::ofstream debug_file("dbg_output.log");
  LogKeeper::get_log("debug").set_target(&debug_file);

  std::string filename = std::string(datadir) + std::string("two_hexas_3d.inp");
  ScopedPtr< MeshBuilder >::Type mb(new MeshDbViewBuilder(tdim, gdim));
  ScopedPtr< Reader >::Type reader(new UcdReader(mb.get()));
  // ScopedPtr<Reader>::Type reader(new VtkReader(mb.get()));

  MeshPtr mesh;
  reader->read(filename.c_str(), mesh);

  std::vector< int > attr_values(mesh->num_entities(tdim), 5);
  AttributePtr attr_ptr_int(
    new IntAttribute(std::vector< int >(mesh->num_entities(tdim), 5)));
  mesh->add_attribute("magic_int", tdim, attr_ptr_int);

  AttributePtr attr_ptr_double(new DoubleAttribute(
                                 std::vector< double >(mesh->num_entities(tdim), 3.14)));
  mesh->add_attribute("magic_double", tdim, attr_ptr_double);

  for (EntityIterator it = mesh->begin(tdim); it != mesh->end(tdim); ++it) {
    int magic_int;
    it->get("magic_int", &magic_int);
    LOG_DEBUG(1, "Cell " << it->id() << " has magic int " << magic_int << "\n");
    BOOST_CHECK_EQUAL(magic_int, 5);

    double magic_double;
    it->get("magic_double", &magic_double);
    LOG_DEBUG(1, "Cell " << it->id() << " has magic double " << magic_double
              << "\n");
    BOOST_CHECK_EQUAL(magic_double, 3.14);
  }

  // flush log here to avoid problems
  LogKeeper::get_log("debug").flush();
}
