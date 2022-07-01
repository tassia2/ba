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

/// \author Martin Baumann, Thomas Gengenbach, Michael Schick

#define BOOST_TEST_MODULE dof_identification

#include <boost/test/unit_test.hpp>
#include <boost/test/included/unit_test.hpp>


#include <mpi.h>
#include <string>
#include <vector>

#include "hiflow.h"
#include "test.h"

#define TEST_BDM
#define TEST_LAG
#define TEST_RT

#define TEST_TRI
#define TEST_QUAD
#define TEST_TET
#define TEST_HEX
#define nTEST_COMPLEX

int INIT_LEVEL = 0;
int NUM_LEVEL = 3;

int MAX_DEG_LAG = 2;
int MAX_DEG_BDM = 2;
int MAX_DEG_RT = 2;

using namespace std;
using namespace hiflow;
using namespace hiflow::mesh;

static const char *datadir = MESH_DATADIR;

/// DegreeOfFreedom Identification test
///
/// \brief For a list of files it is analysed, whether the number of DoFs
///        after identification is correct.
///
/// For a continuous FE ansatz (linear/quadratic) the number of DoFs after
/// identification of common DoFs are checked by the number of points of meshes.


BOOST_AUTO_TEST_CASE(dof_identification) {

    LogKeeper::get_log("info").set_target(&(std::cout));

    LogKeeper::get_log("debug").set_target(&(std::cout));

    LogKeeper::get_log("error").set_target(&(std::cout));


  int argc = boost::unit_test::framework::master_test_suite().argc;
  char** argv = boost::unit_test::framework::master_test_suite().argv;

  MPI_Init(&argc, &argv);

  int init_level = 0;
  int num_level = NUM_LEVEL;

  int rank, num_proc;
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
  MPI_Comm_size (MPI_COMM_WORLD, &num_proc);

  // Which files should be checked?

  std::vector< std::string > filenames;
  std::vector< TDim > tdims;
  std::vector< GDim > gdims;
  std::vector< RefCellType > cell_types;
  
#ifdef TEST_TRI
  filenames.push_back(std::string(datadir) + std::string("two_triangles_2d.inp"));
  tdims.push_back(2);
  gdims.push_back(2);
  cell_types.push_back(RefCellType::TRI_STD);
#endif
#ifdef TEST_QUAD
  filenames.push_back(std::string(datadir) + std::string("two_quads_2d.inp"));
  tdims.push_back(2);
  gdims.push_back(2);
  cell_types.push_back(RefCellType::QUAD_STD);
#endif
#ifdef TEST_TET
  filenames.push_back(std::string(datadir) + std::string("two_tetras_3d.inp"));
  tdims.push_back(3);
  gdims.push_back(3);
  cell_types.push_back(RefCellType::TET_STD);
#endif
#ifdef TEST_HEX
  filenames.push_back(std::string(datadir) + std::string("two_hexas_3d.inp"));
  tdims.push_back(3);
  gdims.push_back(3);
  cell_types.push_back(RefCellType::HEX_STD);
#endif
  
#ifdef TEST_COMPLEX
  filenames.push_back(std::string(datadir) + std::string("2d_lung_4g.inp"));
  tdims.push_back(2);
  gdims.push_back(2);
  cell_types.push_back(RefCellType::TRI_STD);
  
  filenames.push_back(std::string(datadir) + std::string("unit_cube_tetras_3d.inp"));
  tdims.push_back(3);
  gdims.push_back(3);
  cell_types.push_back(RefCellType::TET_STD);
#endif
  
  VectorSpace< double, 1 > space1;
  VectorSpace< double, 2 > space2;
  VectorSpace< double, 3 > space3; 
  VectorSpace< double, 2 > rt_space2;
  VectorSpace< double, 3 > rt_space3; 
  VectorSpace< double, 2 > bdm_space2;
  VectorSpace< double, 3 > bdm_space3; 

  for (int test_number = 0; test_number < filenames.size(); ++test_number) {

    std::string filename = filenames.at(test_number);
    TDim tdim = tdims.at(test_number);
    GDim gdim = gdims.at(test_number);

    /////////////////////////////////////
    // mesh

    /////////////////////////////////////
    // mesh
    std::vector<MeshPtr> master_mesh(num_level+1);
    
    if (rank == 0) 
    {
      master_mesh[0] = read_mesh_from_file(filename.c_str(), gdim, gdim, 0);
      int cur_level = 0;
      while (cur_level < init_level)
      {
        master_mesh[0] = master_mesh[0]->refine();
        cur_level++;
      }
      for (int i=1; i<num_level+1; ++i)
      {
        master_mesh[i] = master_mesh[i-1]->refine();
      }
    }

    std::vector< MeshPtr > mesh(num_level+1);
    for (int i=0; i<num_level+1; ++i)
    {
      int num_ref_seq_steps;
      MeshPtr l_mesh = partition_and_distribute(master_mesh[i], 0, MPI_COMM_WORLD, num_ref_seq_steps);
      assert(l_mesh != 0);
  
      SharedVertexTable shared_verts;
      mesh[i] = compute_ghost_cells(*l_mesh, MPI_COMM_WORLD, shared_verts);
    }
    
    for (int i = 0; i < num_level; ++i) 
    {
      // tests
      CONSOLE_OUTPUT(rank, "Testing " << filename << " on mesh level: " << init_level + i);
    
      std::vector< FEType > fe_ansatz (1, FEType::LAGRANGE);
      std::vector< FEType > rt_fe_ansatz (1, FEType::RT);
      std::vector< FEType > bdm_fe_ansatz (1, FEType::BDM);
      std::vector< bool > is_cg(1, true);
      std::vector< int > degrees1(1,1);
      std::vector< int > degrees2(1,2);
      
      std::vector< std::vector< int > > rt_degrees(MAX_DEG_RT);
      for (int l=0; l < rt_degrees.size(); ++l)
      {
        rt_degrees[l].resize(1,l);
      }
      std::vector< std::vector< int > > bdm_degrees(MAX_DEG_BDM);
      for (int l=0; l < bdm_degrees.size(); ++l)
      {
        bdm_degrees[l].resize(1,l+1);
      }

#ifdef TEST_LAG
      // -> Linear Lagrange Ansatz on Mesh_Level 0
      CONSOLE_OUTPUT(rank, "test LAG space of degree " << degrees1[0]);
      bool do_test = false;
      int l_nb_dof = 0;
      int q1_dofs = 0;
      if (rank == 0)
      {
        q1_dofs = master_mesh[i]->num_entities(0);
      }
      
      if (gdim == 1)
      {
        space1.Init(*mesh[i], fe_ansatz, is_cg, degrees1);
        l_nb_dof = space1.dof().nb_dofs_local();
        do_test = true;
      }
      
      else if (gdim == 2)
      {
        space2.Init(*mesh[i], fe_ansatz, is_cg, degrees1);
        l_nb_dof = space2.dof().nb_dofs_local();
        do_test = true;
      }
      else if (gdim == 4)
      {
        space3.Init(*mesh[i], fe_ansatz, is_cg, degrees1);
        l_nb_dof = space3.dof().nb_dofs_local();
        do_test = true;
      }
      
      if (do_test)
      {
        int g_nb_dof = 0;
        MPI_Reduce(&l_nb_dof, &g_nb_dof, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0)
        {
          BOOST_TEST(q1_dofs == g_nb_dof);
        }
      }
      
      // -> Quadratic Lagrange Ansatz on Mesh_Level 0
      CONSOLE_OUTPUT(rank, "test LAG space of degree " << degrees2[0]);
      do_test = false;
      l_nb_dof = 0;
      int q2_dofs = -1;
      if (rank == 0)
      {
        q2_dofs = master_mesh[i+1]->num_entities(0);
      }
      
      if (gdim == 1)
      {
        space1.Init(*mesh[i], fe_ansatz, is_cg, degrees2);
        l_nb_dof = space1.dof().nb_dofs_local();
        do_test = true;
      }
      else if (gdim == 2)
      {
        space2.Init(*mesh[i], fe_ansatz, is_cg, degrees2);
        l_nb_dof = space2.dof().nb_dofs_local();
        do_test = true;
      }
      else if (gdim == 4)
      {
        space3.Init(*mesh[i], fe_ansatz, is_cg, degrees2);
        l_nb_dof = space3.dof().nb_dofs_local();
        do_test = true;
      }
      if (do_test)
      {
        int g_nb_dof = 0;
        MPI_Reduce(&l_nb_dof, &g_nb_dof, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0)
        {
          BOOST_TEST(q2_dofs == g_nb_dof);
        }
      }
#endif
#ifdef TEST_BDM
      // BDM element
      l_nb_dof = 0;
      if (gdim == 2 && cell_types[test_number] == RefCellType::TRI_STD)
      {
        size_t nb_facet = 0;
        size_t nb_cell = 0;
        
        if (rank == 0)
        {
          nb_facet = master_mesh[i]->num_entities(1);
          nb_cell = master_mesh[i]->num_entities(2);
        }
        
        for (int l=0; l<bdm_degrees.size(); ++l)
        {
          CONSOLE_OUTPUT(rank, "test BDM space of degree " << bdm_degrees[l][0]);
          bdm_space2.Init(*mesh[i], bdm_fe_ansatz, is_cg, bdm_degrees[l]);
          l_nb_dof = bdm_space2.dof().nb_dofs_local();
        
          size_t deg = bdm_degrees[l][0];
          size_t nb_dof_per_facet = deg+1;
          size_t nb_dof_per_cell = 0;
          if (deg >= 2)
          {
            nb_dof_per_cell = (deg-1) * (deg + 1);
          }
        
          size_t ref_dof = nb_facet * nb_dof_per_facet + nb_cell * nb_dof_per_cell;
          int g_nb_dof = 0;
          MPI_Reduce(&l_nb_dof, &g_nb_dof, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
          if (rank == 0)
          {
            BOOST_TEST(ref_dof == g_nb_dof);
            //CONSOLE_OUTPUT(rank, " #dofs in space: " << g_nb_dof << " , exact number: " << ref_dof); 
          }
        }
      }
#endif
#ifdef TEST_RT
      // RT element
      l_nb_dof = 0;
      if (gdim == 2 && cell_types[test_number] == RefCellType::TRI_STD)
      {
        size_t nb_facet = 0;
        size_t nb_cell = 0;
        
        if (rank == 0)
        {
          nb_facet = master_mesh[i]->num_entities(1);
          nb_cell = master_mesh[i]->num_entities(2);
        }
        
        for (int l=0; l<rt_degrees.size(); ++l)
        {
          CONSOLE_OUTPUT(rank, "test RT space of degree " << rt_degrees[l][0]);
          rt_space2.Init(*mesh[i], rt_fe_ansatz, is_cg, rt_degrees[l]);
          l_nb_dof = rt_space2.dof().nb_dofs_local();
        
          size_t deg = rt_degrees[l][0];
          size_t nb_dof_per_facet = deg+1;
          size_t nb_dof_per_cell = 0;
          if (deg >= 1)
          {
            nb_dof_per_cell = deg * (deg + 1);
          }
        
          size_t ref_dof = nb_facet * nb_dof_per_facet + nb_cell * nb_dof_per_cell;
          int g_nb_dof = 0;
          MPI_Reduce(&l_nb_dof, &g_nb_dof, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
          if (rank == 0)
          {
            BOOST_TEST(ref_dof == g_nb_dof);
            //CONSOLE_OUTPUT(rank, " #dofs in space: " << g_nb_dof << " , exact number: " << ref_dof); 
          }
        }
      }
      
      // RT element on tet
      l_nb_dof = 0;
      if (gdim == 3 && cell_types[test_number] == RefCellType::TET_STD)
      {
        size_t nb_facet = 0;
        size_t nb_cell = 0;
        
        if (rank == 0)
        {
          nb_facet = master_mesh[i]->num_entities(2);
          nb_cell = master_mesh[i]->num_entities(3);
        }
        
        for (int l=0; l<rt_degrees.size(); ++l)
        {
          CONSOLE_OUTPUT(rank, "test RT space of degree " << rt_degrees[l][0]);
          rt_space3.Init(*mesh[i], rt_fe_ansatz, is_cg, rt_degrees[l]);
          l_nb_dof = rt_space3.dof().nb_dofs_local();
        
          size_t deg = rt_degrees[l][0];
          size_t nb_dof_per_facet = (deg+1) * (deg+2) / 2;
          size_t nb_dof_per_cell = 0;
          if (deg >= 1)
          {
            nb_dof_per_cell = deg * (deg + 1) * (deg + 2) / 2;
          }
        
          size_t ref_dof = nb_facet * nb_dof_per_facet + nb_cell * nb_dof_per_cell;
          int g_nb_dof = 0;
          MPI_Reduce(&l_nb_dof, &g_nb_dof, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
          if (rank == 0)
          {
            BOOST_TEST(ref_dof == g_nb_dof);
            //CONSOLE_OUTPUT(rank, " #dofs in space: " << g_nb_dof << " , exact number: " << ref_dof); 
          }
        }
      }

      // RT element on hex
      l_nb_dof = 0;
      if (gdim == 3 && cell_types[test_number] == RefCellType::HEX_STD)
      {
        size_t nb_facet = 0;
        size_t nb_cell = 0;
        
        if (rank == 0)
        {
          nb_facet = master_mesh[i]->num_entities(2);
          nb_cell = master_mesh[i]->num_entities(3);
        }
        
        for (int l=0; l<rt_degrees.size(); ++l)
        {
          CONSOLE_OUTPUT(rank, "test RT space of degree " << rt_degrees[l][0]);
          rt_space3.Init(*mesh[i], rt_fe_ansatz, is_cg, rt_degrees[l]);
          l_nb_dof = rt_space3.dof().nb_dofs_local();
        
          size_t deg = rt_degrees[l][0];
          size_t nb_dof_per_facet = (deg+1) * (deg+1);
          size_t nb_dof_per_cell = 0;
          if (deg >= 1)
          {
            nb_dof_per_cell = 3 * deg * (deg + 1) * (deg + 1);
          }
        
          size_t ref_dof = nb_facet * nb_dof_per_facet + nb_cell * nb_dof_per_cell;
          int g_nb_dof = 0;
          MPI_Reduce(&l_nb_dof, &g_nb_dof, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
          if (rank == 0)
          {
            BOOST_TEST(ref_dof == g_nb_dof);
            //CONSOLE_OUTPUT(rank, " #dofs in space: " << g_nb_dof << " , exact number: " << ref_dof); 
          }
        }
      }
#endif
      CONSOLE_OUTPUT(rank, "===============================");
    }
  }

  MPI_Finalize();

}
