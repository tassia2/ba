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

#include <mpi.h>
#include <string>
#include <vector>

#include "hiflow.h"
#include "test.h"

#define BOOST_TEST_MODULE parallel_dof

#include <boost/test/unit_test.hpp>
#include <boost/test/included/unit_test.hpp>
#include <boost/mpl/list.hpp>

namespace utf = boost::unit_test;
namespace tt = boost::test_tools;

using namespace std;
using namespace hiflow;
using namespace hiflow::mesh;
using namespace hiflow::la;

/// 
///
/// \brief 
///

const int DEBUG_OUT = 1;
const double pi = 3.14159265;

template <class DataType> 
void print (std::vector< std::vector<DataType> > vals)
{
  for (size_t l=0; l<vals.size(); ++l)
  {
    for (size_t i=0; i<vals[l].size(); ++i)
    {
      std::cout << vals[l][i] << " ";
    }
    std::cout << std::endl;
  }
}

template <class DataType, int DIM>
void set_dof_values (VectorSpace<DataType,DIM> &space, 
                     MeshPtr mesh_ptr,
                     CoupledVector<DataType>& sol)
{
  // define fe coeff vector 
  sol.Init(MPI_COMM_WORLD, space.la_couplings());
  sol.Zeros();

  int first_ind = sol.ownership_begin();
  int last_ind = sol.ownership_begin();
  int num_local = sol.size_local();
  
  for (int i=0; i<num_local; ++i)
  {
    sol.SetValue(first_ind+i, static_cast<DataType>(first_ind+i));
  }
  sol.Update();
}

template <class DataType, int DIM>
bool cmp (Vec<DIM, DataType> lhs, Vec<DIM, DataType> rhs, DataType eps)
{
  for (int l=0; l<DIM; ++l)
  {
    if (std::abs(lhs[l] - rhs[l]) > eps)
    {
      return false;
    }
  }
  return true;
}

template <class DataType, int DIM>
void check_dof_factors (VectorSpace<DataType,DIM> &space)
{
  MPI_Barrier (space.get_mpi_comm());
  
  MeshPtr meshptr = space.meshPtr();
  int nb_sub = space.nb_subdom();
  int my_rank = space.rank();
  
  BOOST_TEST (nb_sub > 1);
  
  /*
  mesh::InterfaceList interface_list = mesh::InterfaceList::create(meshptr);
  for (int p=0; p<nb_sub; ++p)
  {
    if (p == my_rank)
    //if (DEBUG_LEVEL > 0) 
    {
      std::cout << interface_list << std::endl;
    }
    MPI_Barrier (space.get_mpi_comm());
  }
  MPI_Barrier (space.get_mpi_comm());
*/

  std::vector< std::vector<int> > remote_2_send (nb_sub);
  std::vector< std::vector< std::vector<DataType> > > factors_2_send (nb_sub);
  std::vector< std::vector< DataType > > own_factors(meshptr->num_entities(DIM));


  int nb_dof_factors;
  
  // loop over cells
  for (mesh::EntityIterator it = meshptr->begin(DIM);
       it != meshptr->end(DIM); ++it) 
  {
    int cell_index = it->index();
    int remote, domain;
    meshptr->get_attribute_value("_remote_index_", DIM, cell_index, &remote);
    meshptr->get_attribute_value("_sub_domain_", DIM, it->index(), &domain);

    std::vector<DataType> c_factors;
    space.dof().get_dof_factors_on_cell(cell_index, c_factors);
    
    own_factors[cell_index] = c_factors;
    
    if (remote >= 0)
    {
      remote_2_send[domain].push_back(remote);
      factors_2_send[domain].push_back(c_factors);
    }
    nb_dof_factors = c_factors.size();
  }
  
  // exchange data
  std::vector<int> num_cells_2_recv(nb_sub, 0);
  
  for (int p=0; p<nb_sub; ++p)
  {
    if (p == my_rank)
      continue;
      
    int num_cells_2_send = remote_2_send[p].size();   
    MPI_Send(&num_cells_2_send, 1, MPI_INT, p, 0, space.get_mpi_comm());
    MPI_Send(&remote_2_send[p][0], num_cells_2_send, MPI_INT, p, 1, space.get_mpi_comm());
   
    for (int i=0; i<num_cells_2_send; ++i)
    {
      MPI_Send(&factors_2_send[p][i][0], nb_dof_factors, MPI_DOUBLE, p, 2+i, space.get_mpi_comm());
    }
  
  }

  
  std::vector< int > local_shared_ind;
  std::vector< std::vector<DataType> > local_shared_factors;
  for (int p=0; p<nb_sub; ++p)
  {
    if (p == my_rank)
      continue;
    
    int num_cells_from_p = 0;
    MPI_Status state;   
    MPI_Recv(&num_cells_from_p, 1, MPI_INT, p, 0, space.get_mpi_comm(), &state);
  
    std::vector<int> remote_from_p(num_cells_from_p);
    MPI_Recv(&remote_from_p[0], num_cells_from_p, MPI_INT, p, 1, space.get_mpi_comm(), &state);
  
    std::vector< std::vector<DataType> > factors_from_p(num_cells_from_p);
    for (int i=0; i<num_cells_from_p; ++i)
    {
      factors_from_p[i].resize(nb_dof_factors, -99);
      MPI_Recv(&factors_from_p[i][0], nb_dof_factors, MPI_DOUBLE, p, 2+i, space.get_mpi_comm(), &state);

      local_shared_ind.push_back(remote_from_p[i]);
      local_shared_factors.push_back(factors_from_p[i]);
    }
  }
  
  MPI_Barrier (space.get_mpi_comm());
  for (int p=0; p<nb_sub; ++p)
  {
    if (p == my_rank)
    {
      for (mesh::EntityIterator it = meshptr->begin(DIM);
          it != meshptr->end(DIM); ++it) 
      {
        int cell_index = it->index();
        int remote, domain;
        meshptr->get_attribute_value("_remote_index_", DIM, cell_index, &remote);
        meshptr->get_attribute_value("_sub_domain_", DIM, it->index(), &domain);

        std::vector<DataType> c_factors;
        space.dof().get_dof_factors_on_cell(cell_index, c_factors);
/*            
        std::cout << "[" << my_rank << "] cell index " << cell_index << " remote index " << remote << " : local  factors = " 
                  << string_from_range(c_factors.begin(), c_factors.end()) << std::endl;
*/
      }
    }
    MPI_Barrier(space.get_mpi_comm());
  }
  
  
  MPI_Barrier (space.get_mpi_comm());
  for (int p=0; p<nb_sub; ++p)
  {
    if (p == my_rank)
    {
      for (int i=0; i<local_shared_ind.size(); ++i)
      {
        int index = local_shared_ind[i];
/*
        std::cout << "[" << my_rank << "] cell index " << index << " : local  factors = " 
                  << string_from_range(own_factors[index].begin(), own_factors[index].end()) << std::endl;
        std::cout << "[" << my_rank << "] cell index " << index << " : remote factors = " 
                  << string_from_range(local_shared_factors[i].begin(), local_shared_factors[i].end()) << std::endl;
*/
        for (int l=0; l<nb_dof_factors; ++l)
        {
          BOOST_CHECK(std::abs(own_factors[index][l]- local_shared_factors[i][l])< 1e-6);
        }
      }
    }
    MPI_Barrier(space.get_mpi_comm());
  }
}

template <class DataType, int DIM>
void visualize_dofs (VectorSpace<DataType,DIM> &space, 
                     CoupledVector<DataType>& sol,
                     std::string& name)
{
  int nb_var = space.nb_var();
  MeshPtr meshptr = space.meshPtr();
  const int tdim = meshptr->tdim();
  const int num_cells = meshptr->num_entities(tdim);
  
  // prepare cell attributes to be visualized
  std::vector< DataType > remote_index(num_cells, 0);
  std::vector< DataType > cell_index(num_cells, 0);
  std::vector< DataType > sub_domain(num_cells, 0);
  std::vector< DataType > material_number(num_cells, 0);

  for (mesh::EntityIterator it = meshptr->begin(tdim);
       it != meshptr->end(tdim); ++it) 
  {
    int temp1, temp2;
    meshptr->get_attribute_value("_remote_index_", tdim, it->index(), &temp1);
    meshptr->get_attribute_value("_sub_domain_", tdim, it->index(), &temp2);
    material_number.at(it->index()) = meshptr->get_material_number(tdim, it->index());
    remote_index.at(it->index()) = temp1;
    sub_domain.at(it->index()) = temp2;
    cell_index.at(it->index()) = it->index();
  }
  
  // Setup visualization object.
  int num_intervals = 1;
  
  std::vector<size_t> visu_vars;
  std::vector< std::string > names;
  for (int l=0; l<nb_var; ++l)
  {
    visu_vars.push_back(l);
    names.push_back("u_" + std::to_string(l));
  }
  
  CellVisualization< DataType, DIM > visu(space, num_intervals);

  visu.visualize (FeEvalCell< DataType, DIM >(space, sol, visu_vars), names);
  visu.visualize_cell_data(material_number, "Material Id");
  visu.visualize_cell_data(remote_index, "_remote_index_");
  visu.visualize_cell_data(sub_domain, "_sub_domain_");
  visu.visualize_cell_data(cell_index, "_cell_index_");
  
  // writer object
  VTKWriter< DataType, DIM> vtk_writer (visu, MPI_COMM_WORLD, 0);
  vtk_writer.write(name);
}

static const char *datadir = MESH_DATADIR;

BOOST_AUTO_TEST_CASE(parallel_dof) {

  int argc = boost::unit_test::framework::master_test_suite().argc;
  char** argv = boost::unit_test::framework::master_test_suite().argv;
  
  int init_level = 2;
  int num_level = 3;
  int rank;
  MPI_Init(&argc, &argv);

  MPI_Comm_rank(MPI_COMM_WORLD,  &rank);
  
  // Which files should be checked?

  std::vector< std::string > filenames;
  std::vector< TDim > tdims;
  std::vector< GDim > gdims;
  std::vector< RefCellType > cell_types;
  
  filenames.push_back(std::string(datadir) + std::string("unit_square-2_tri.inp"));
  tdims.push_back(2);
  gdims.push_back(2);
  cell_types.push_back(RefCellType::TRI_STD);

  filenames.push_back(std::string(datadir) + std::string("two_quads_2d.inp"));
  tdims.push_back(2);
  gdims.push_back(2);
  cell_types.push_back(RefCellType::QUAD_STD);
  
  filenames.push_back(std::string(datadir) + std::string("two_tetras_3d.inp"));
  tdims.push_back(3);
  gdims.push_back(3);
  cell_types.push_back(RefCellType::TET_STD);
  
  filenames.push_back(std::string(datadir) + std::string("two_hexas_3d.inp"));
  tdims.push_back(3);
  gdims.push_back(3);
  cell_types.push_back(RefCellType::HEX_STD);
 
  VectorSpace< double, 2 > dg_lag_space2;
  VectorSpace< double, 2 > lag_space2;
  VectorSpace< double, 3 > lag_space3;
  VectorSpace< double, 2 > rt_space2;
  VectorSpace< double, 2 > bdm_space2;
  VectorSpace< double, 2 > bdm_dg_space2;

  std::vector< FEType > lag_fe_ansatz (1, FEType::LAGRANGE);
  std::vector< FEType > lag_fe_ansatz2 (2, FEType::LAGRANGE);
  std::vector< FEType > lag_fe_ansatz3 (3, FEType::LAGRANGE);
  std::vector< FEType > rt_fe_ansatz (1, FEType::RT);
  std::vector< FEType > bdm_fe_ansatz (1, FEType::BDM);
  std::vector< FEType > bdm_dg_fe_ansatz (2, FEType::BDM);
  bdm_dg_fe_ansatz[1] = FEType::LAGRANGE;
  
  std::vector< bool > is_cg(1, true);
  std::vector< bool > is_cg2(2, true);
  std::vector< bool > is_cg3(3, true);
  std::vector< bool > is_dg(1, false);
  std::vector< bool > bdm_dg_is_cg(2, true);
  bdm_dg_is_cg[1] = false;
  
  std::vector< std::vector< int > > dg_lag_degrees(3);
  for (int l=0; l < dg_lag_degrees.size(); ++l)
  {
    dg_lag_degrees[l].resize(1,l);
  }
  
  std::vector< std::vector< int > > lag_degrees2(2);
  for (int l=0; l < lag_degrees2.size(); ++l)
  {
    lag_degrees2[l].resize(2,l+1);
  }
  
  std::vector< std::vector< int > > lag_degrees3(3);
  for (int l=0; l < lag_degrees3.size(); ++l)
  {
    lag_degrees3[l].resize(3,l+1);
  }
  
  std::vector< std::vector< int > > rt_degrees(3);
  for (int l=0; l < rt_degrees.size(); ++l)
  {
    rt_degrees[l].resize(1,l);
  }
  
  std::vector< std::vector< int > > bdm_degrees(2);
  for (int l=0; l < bdm_degrees.size(); ++l)
  {
    bdm_degrees[l].resize(1,l+1);
  }

  std::vector< std::vector< int > > bdm_dg_degrees(2);
  for (int l=0; l < bdm_dg_degrees.size(); ++l)
  {
    bdm_dg_degrees[l].resize(2,l+1);
    bdm_dg_degrees[l][1] = l;
  }
  
  for (int test_number = 0; test_number < filenames.size(); ++test_number) 
  {
    std::string filename = filenames.at(test_number);
    TDim tdim = tdims.at(test_number);
    GDim gdim = gdims.at(test_number);

    /////////////////////////////////////
    // mesh
    std::vector<MeshPtr> master_mesh(num_level);
    
    if (rank == 0) 
    {
      master_mesh[0] = read_mesh_from_file(filename.c_str(), gdim, gdim, 0);
      int cur_level = 0;
      while (cur_level < init_level)
      {
        master_mesh[0] = master_mesh[0]->refine();
        cur_level++;
      }
      for (int i=1; i<num_level; ++i)
      {
        master_mesh[i] = master_mesh[i-1]->refine();
      }
    }
    
    std::vector< MeshPtr > mesh(num_level);
    for (int i=0; i<num_level; ++i)
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
      CONSOLE_OUTPUT(rank, "Testing " << filename << " on mesh level: " << init_level + i );
#if 1
      // BDM element
      if (gdim == 2 && cell_types[test_number] == RefCellType::TRI_STD)
      {
        for (int l=0; l<bdm_degrees.size(); ++l)
        {
          std::string name = "bdm_tri_" + std::to_string(i) + "_"  + std::to_string(bdm_degrees[l][0]);
          CONSOLE_OUTPUT(rank, "test BDM      space of degree " << bdm_degrees[l][0]);
          bdm_space2.Init(*mesh[i], bdm_fe_ansatz, is_cg, bdm_degrees[l]);

          // interpolate function
          CoupledVector<double> sol;
          set_dof_values<double, 2> (bdm_space2, mesh[i], sol);
          check_dof_factors<double, 2> (bdm_space2);
          //visualize_dofs<double, 2> (bdm_space2, sol, name);
        }
      }
#endif
#if 1
      // discontinuous Lagrange element
      if (gdim == 2 && cell_types[test_number] == RefCellType::TRI_STD)
      {
        for (int l=0; l<dg_lag_degrees.size(); ++l)
        {
          // setup space
          std::string name = "dg_lag_tri_" + std::to_string(i) + "_"  + std::to_string(dg_lag_degrees[l][0]);
          
          CONSOLE_OUTPUT(rank, "test DG       space of degree " << dg_lag_degrees[l][0]);
          dg_lag_space2.Init(*mesh[i], lag_fe_ansatz, is_dg, dg_lag_degrees[l]);
          
          // interpolate function
          CoupledVector<double> sol;
          set_dof_values<double, 2> (dg_lag_space2, mesh[i], sol);
          check_dof_factors<double, 2> (dg_lag_space2);
          //visualize_dofs<double, 2> (lag_space2, sol, name);
        }
      }
#endif
#if 1
      // mixed BDM / discontinuous Lagrange element
      if (gdim == 2 && cell_types[test_number] == RefCellType::TRI_STD)
      {
        for (int l=0; l<bdm_dg_degrees.size(); ++l)
        {
          // setup space
          std::string name = "bdm_dg_tri_" + std::to_string(i) + "_"  + std::to_string(bdm_dg_degrees[l][0]);
          
          CONSOLE_OUTPUT(rank, "test BDM / DG space of degree " << bdm_dg_degrees[l][0] << " / " << bdm_dg_degrees[l][1]);
          bdm_dg_space2.Init(*mesh[i], bdm_dg_fe_ansatz, bdm_dg_is_cg, bdm_dg_degrees[l]);
          
          // interpolate function
          CoupledVector<double> sol;
          set_dof_values<double, 2> (bdm_dg_space2, mesh[i], sol);
          check_dof_factors<double, 2> (bdm_dg_space2);
          //visualize_dofs<double, 2> (lag_space2, sol, name);
        }
      }
#endif
#if 1
      // Lagrange element
      if (gdim == 2)
      {
        for (int l=0; l<lag_degrees2.size(); ++l)
        {
          // setup space
          std::string name = "lag_2D_" + std::to_string(i) + "_"  + std::to_string(lag_degrees2[l][0]);
          
          lag_space2.Init(*mesh[i], lag_fe_ansatz2, is_cg2, lag_degrees2[l]);
          CONSOLE_OUTPUT(rank, "test " << lag_space2.nb_fe() << " x " << lag_space2.nb_var() / lag_space2.nb_fe()
                    << " CG space of degree " << lag_degrees2[l][0]);
          
          // interpolate function
          CoupledVector<double> sol;
          set_dof_values<double, 2> (lag_space2, mesh[i], sol);
          check_dof_factors<double, 2> (lag_space2);
          //visualize_dofs<double, 2> (lag_space2, sol, name);
        }
      }
      if (gdim == 3)
      {
        for (int l=0; l<lag_degrees3.size(); ++l)
        {
          // setup space
          std::string name = "lag_3D_" + std::to_string(i) + "_"  + std::to_string(lag_degrees3[l][0]);
          
          lag_space3.Init(*mesh[i], lag_fe_ansatz3, is_cg3, lag_degrees3[l]);
          CONSOLE_OUTPUT(rank, "test " << lag_space3.nb_fe() << " x " << lag_space3.nb_var() / lag_space3.nb_fe()
                    << " CG space of degree " << lag_degrees3[l][0]);
          
          // interpolate function
          CoupledVector<double> sol;
          set_dof_values<double, 3> (lag_space3, mesh[i], sol);
          check_dof_factors<double, 3> (lag_space3);
          //visualize_dofs<double, 3> (lag_space3, sol, name);
        }
      }
#endif
#if 1
      // RT element
      if (gdim == 2 && cell_types[test_number] == RefCellType::TRI_STD)
      {
        for (int l=0; l<rt_degrees.size(); ++l)
        {
          std::string name = "rt_tri_" + std::to_string(i) + "_"  + std::to_string(rt_degrees[l][0]);

          CONSOLE_OUTPUT(rank, "test RT       space of degree " << rt_degrees[l][0]);
          rt_space2.Init(*mesh[i], rt_fe_ansatz, is_cg, rt_degrees[l]);

          // interpolate function
          CoupledVector<double> sol;
          set_dof_values<double, 2> (rt_space2, mesh[i], sol);
          check_dof_factors<double, 2> (rt_space2);
          //visualize_dofs<double, 2> (rt_space2, sol, name);
        }
      }
#endif
      CONSOLE_OUTPUT(rank, "===============================");
    }
  }

  MPI_Finalize();
}
