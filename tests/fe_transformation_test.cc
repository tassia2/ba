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

#define BOOST_TEST_MODULE fe_transformation

#include <mpi.h>
#include <string>
#include <vector>
#include <boost/test/unit_test.hpp>
#include <boost/test/included/unit_test.hpp>

#include "hiflow.h"
#include "test.h"

using namespace std;
using namespace hiflow;
using namespace hiflow::mesh;

#define TEST_BDM
#define TEST_LAG
#define TEST_RT

#define TEST_TRI
#define TEST_QUAD
#define TEST_TET
#define TEST_HEX

int INIT_LEVEL = 0;
int NUM_LEVEL = 2;

int MAX_DEG_LAG = 2;
int MAX_DEG_BDM = 2;
int MAX_DEG_RT = 2;

/// FE tranformation test
///
/// \brief For a list of files it is analysed, whether the FE transformation applied 
/// to its inverse yields the identity operator
///


static const char *datadir = MESH_DATADIR;

template <class DataType, int DIM>
void check_test_points (std::vector<Vec<DIM,DataType> >& test_points, 
                        RefElement<DataType, DIM> * ref_fe,  
                        MappingPhys2Ref <DataType, DIM, MappingRef2Phys<DataType, DIM, RefElement<DataType, DIM> > > * eval_phys_on_ref)
{       
  for (int q=0; q<test_points.size(); ++q)
  {
    
    Vec<DIM,DataType> pt = test_points[q];
    size_t weight_size = ref_fe->dim() * ref_fe->nb_comp();
                        
    //std::cout << "pt " << pt << std::endl;
    std::vector< DataType > weight_unmapped (weight_size, 0.);
    std::vector< DataType > weight_mapped (weight_size, 0.);
               
    ref_fe->N(pt, weight_unmapped);
    eval_phys_on_ref->evaluate(pt, weight_mapped);
            
//std::cout << " ========================= " << std::endl;
//std::cout << "point " << pt[0] << " , " << pt[1] << std::endl;
//std::cout << std::scientific << std::setprecision(2);

    for (size_t i=0; i<weight_size; ++i)
    {
//std::cout << std::setw(5) << weight_unmapped[i] << " ";
    }
//std::cout << std::endl;
    for (size_t i=0; i<weight_size; ++i)
    {
//std::cout << std::setw(5) << weight_mapped[i] << " ";
    }
//std::cout << std::endl;

    for (size_t i=0; i<weight_size; ++i)
    {
      BOOST_TEST(std::abs(weight_unmapped[i]- weight_mapped[i])<1e-10);
/*
      if (std::abs(weight_unmapped[i]- weight_mapped[i])>1e-10)
      {
        std::cout << "i " << i << " : fail " << weight_unmapped[i] << " vs " << weight_mapped[i] << std::endl;
      }
*/
    }
  }
}

template <class DataType, int DIM>
void check_test_points (std::vector<Vec<DIM,DataType> >& test_points, 
                        CellTrafoSPtr<DataType, DIM> trafo1,
                        CellTrafoSPtr<DataType, DIM> trafo2)
{       
  for (int q=0; q<test_points.size(); ++q)
  {
    Vec<DIM,DataType> ref_pt = test_points[q];
    Vec<DIM, DataType> ref_pt1, ref_pt2;
    
    Vec<DIM,DataType> pt1, pt2, pt;
    
    pt1 = trafo1->transform(ref_pt);
    pt2 = trafo2->transform(ref_pt);

    //std::cout << ref_pt << " : " << pt1 << " <> " << pt2 << std::endl;
    

    pt = pt1;
    
    bool found1 = trafo1->inverse(pt, ref_pt1);
    bool found2 = trafo2->inverse(pt, ref_pt2);
  
    //std::cout << pt << " : " << ref_pt1 << " <> " << ref_pt2 << " found " << found1 << " <> " << found2 << std::endl;

    BOOST_TEST(norm(pt1 - pt2)<1e-10);
    BOOST_TEST(found1 == found2);
    BOOST_TEST(norm(ref_pt1 - ref_pt2)<1e-10);
    
    Mat<DIM, DIM, DataType> J1, J2;
    DataType detJ1, detJ2;
    
    trafo1->J_and_detJ (ref_pt, J1, detJ1 );
    trafo2->J_and_detJ (ref_pt, J2, detJ2 );
    
    BOOST_TEST(std::sqrt(dot(J1 - J2, J1 - J2))<1e-10);
    BOOST_TEST(std::abs(detJ1 - detJ2)<1e-10);
    //std::cout << " ------------ " << std::endl;
  }
}

template <class DataType, int DIM>
void check_dofs (DofContainer<DataType, DIM> const * dof, 
                 RefElement<DataType, DIM> * ref_fe,  
                 MappingPhys2Ref <DataType, DIM, MappingRef2Phys<DataType, DIM, RefElement<DataType, DIM> > > * eval_phys_on_ref)
{
  size_t nb_dof = ref_fe->nb_dof_on_cell();
  std::vector<cDofId> all_dofs(nb_dof);
  for (size_t k=0; k<nb_dof; ++k)
  {
    all_dofs[k] = k;
  }
            
  // evaluate all dofs for unmapped shape functions
  std::vector< std::vector<DataType> > dof_vals_unmapped;
  dof->evaluate (ref_fe, all_dofs, dof_vals_unmapped);
                        
  // evaluate all dofs at inversly mapped shape functions
  // -> should yield identity matrix 
  std::vector< std::vector<DataType> > dof_vals_mapped;
  dof-> evaluate (eval_phys_on_ref, all_dofs, dof_vals_mapped);
        
  BOOST_TEST(dof_vals_unmapped.size() == dof_vals_mapped.size()); 
            
//  std::cout << std::scientific << std::setprecision(2);

  for (size_t l=0; l<dof_vals_unmapped.size(); ++l)
  {
    //std::cout << " dof " << l << " : ";
    BOOST_TEST(dof_vals_unmapped[l].size() == dof_vals_mapped[l].size());

    for (size_t i=0; i<dof_vals_unmapped[l].size(); ++i)
    {
//std::cout << std::setw(5) << dof_vals_unmapped[l][i] << " ";
    }
//    std::cout << std::endl;
    for (size_t i=0; i<dof_vals_unmapped[l].size(); ++i)
    {
//std::cout << std::setw(5) << dof_vals_mapped[l][i] << " ";
    }
//    std::cout << std::endl << std::endl;

    for (size_t i=0; i<dof_vals_unmapped[l].size(); ++i)
    {
      /*if (std::abs(dof_vals_unmapped[l][i]-dof_vals_mapped[l][i])>1e-10)
      {
        std::cout << " " << i;
      }*/
      BOOST_TEST(std::abs(dof_vals_unmapped[l][i]-dof_vals_mapped[l][i])<1e-10);
    }
    //std::cout << std::endl;
  }
}

BOOST_AUTO_TEST_CASE(fe_transformation) {

  int argc = boost::unit_test::framework::master_test_suite().argc;
  char** argv = boost::unit_test::framework::master_test_suite().argv;
  
  int init_level = INIT_LEVEL;
  int num_level = NUM_LEVEL;
  int rank;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD,  &rank);

  //LogKeeper::get_log ( "info" ).set_target  ( &( std::cout ) );    
  LogKeeper::get_log ( "debug" ).set_target ( &( std::cout ) );
  LogKeeper::get_log ( "error" ).set_target ( &( std::cout ) );
  
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
  std::vector< Vec<2,double> > tri_test_points (6);
  tri_test_points[0].set(0, 0.);
  tri_test_points[0].set(1, 0.);
  tri_test_points[1].set(0, 0.5);
  tri_test_points[1].set(1, 0.);
  tri_test_points[2].set(0, 1.);
  tri_test_points[2].set(1, 0.);
  tri_test_points[3].set(0, 0.);
  tri_test_points[3].set(1, 0.5);
  tri_test_points[4].set(0, 0.5);
  tri_test_points[4].set(1, 0.5);
  tri_test_points[5].set(0, 0.);
  tri_test_points[5].set(1, 1.);
 
  std::vector< Vec<2,double> > quad_test_points (9);
  quad_test_points[0].set(0, 0.);
  quad_test_points[0].set(1, 0.);
  quad_test_points[1].set(0, 0.5);
  quad_test_points[1].set(1, 0.);
  quad_test_points[2].set(0, 1.);
  quad_test_points[2].set(1, 0.);
  quad_test_points[3].set(0, 0.);
  quad_test_points[3].set(1, 0.5);
  quad_test_points[4].set(0, 0.5);
  quad_test_points[4].set(1, 0.5);
  quad_test_points[5].set(0, 1.);
  quad_test_points[5].set(1, 0.5);
  quad_test_points[6].set(0, 0.0);
  quad_test_points[6].set(1, 1.);
  quad_test_points[7].set(0, 0.5);
  quad_test_points[7].set(1, 1.);
  quad_test_points[8].set(0, 1.);
  quad_test_points[8].set(1, 1.);

  std::vector< Vec<3,double> > tet_test_points (11);

  tet_test_points[0].set(0, 0.);
  tet_test_points[0].set(1, 0.);
  tet_test_points[0].set(2, 0.);
  tet_test_points[1].set(0, 0.5);
  tet_test_points[1].set(1, 0.);
  tet_test_points[1].set(2,0.);  
  tet_test_points[2].set(0, 1.);
  tet_test_points[2].set(1, 0.);
  tet_test_points[2].set(2, 0.);
  tet_test_points[3].set(0, 0.);
  tet_test_points[3].set(1, 0.);
  tet_test_points[3].set(2, 0.5); 
  tet_test_points[4].set(0, 0.);
  tet_test_points[4].set(1, 0.5);
  tet_test_points[4].set(2, 0.5);
  tet_test_points[5].set(0, 0.);
  tet_test_points[5].set(1, 0.);
  tet_test_points[5].set(2, 1.);
  tet_test_points[6].set(0, 0.);
  tet_test_points[6].set(1, 0.5);
  tet_test_points[6].set(2, 0.);
  tet_test_points[7].set(0, 0.);
  tet_test_points[7].set(1, 1.);
  tet_test_points[7].set(2, 0.);
  tet_test_points[8].set(0, 0.);
  tet_test_points[8].set(1, 0.5);
  tet_test_points[8].set(2, 0.5);
  tet_test_points[9].set(0, 0.5);
  tet_test_points[9].set(1, 0.5);
  tet_test_points[9].set(2,0.); 
  tet_test_points[10].set(0, 0.25);
  tet_test_points[10].set(1, 0.25);
  tet_test_points[10].set(2, 0.5);

  std::vector< Vec<3,double> > hex_test_points (27);
  hex_test_points[0].set(0, 0.);
  hex_test_points[0].set(1, 0.);
  hex_test_points[0].set(2, 0.);
  hex_test_points[1].set(0, 0.5);
  hex_test_points[1].set(1, 0.);
  hex_test_points[1].set(2, 0.);
  hex_test_points[2].set(0, 1.);
  hex_test_points[2].set(1, 0.);
  hex_test_points[2].set(0, 0.);
  hex_test_points[3].set(0, 0.);
  hex_test_points[3].set(1, 0.);
  hex_test_points[3].set(2, 0.5);
  hex_test_points[4].set(0, 0.5);
  hex_test_points[4].set(1, 0.);
  hex_test_points[4].set(2, 0.5);
  hex_test_points[5].set(0, 1.);
  hex_test_points[5].set(1, 0.);
  hex_test_points[5].set(2, 0.5);
  hex_test_points[6].set(0, 0.);
  hex_test_points[6].set(1, 0.);
  hex_test_points[6].set(2, 1.);
  hex_test_points[7].set(0, 0.5);
  hex_test_points[7].set(1, 0.);
  hex_test_points[7].set(2, 1.);
  hex_test_points[8].set(0, 1.);
  hex_test_points[8].set(1, 0.);
  hex_test_points[8].set(2, 1.);
  hex_test_points[9].set(0, 0.);
  hex_test_points[9].set(1, 0.5);
  hex_test_points[9].set(2, 0.);
  hex_test_points[10].set(0, 0.5);
  hex_test_points[10].set(1, 0.5);
  hex_test_points[10].set(2, 0.);
  hex_test_points[11].set(0, 1.);
  hex_test_points[11].set(1, 0.5);
  hex_test_points[11].set(0, 0.);
  hex_test_points[12].set(0, 0.);
  hex_test_points[12].set(1, 0.5);
  hex_test_points[12].set(2, 0.5);
  hex_test_points[13].set(0, 0.5);
  hex_test_points[13].set(1, 0.5);
  hex_test_points[13].set(2, 0.5);
  hex_test_points[14].set(0, 1.);
  hex_test_points[14].set(1, 0.5);
  hex_test_points[14].set(2, 0.5);
  hex_test_points[15].set(0, 0.);
  hex_test_points[15].set(1, 0.5);
  hex_test_points[15].set(2, 1.);
  hex_test_points[16].set(0, 0.5);
  hex_test_points[16].set(1, 0.5);
  hex_test_points[16].set(2, 1.);
  hex_test_points[17].set(0, 1.);
  hex_test_points[17].set(1, 0.5);
  hex_test_points[17].set(2, 1.);
  hex_test_points[18].set(0, 0.);
  hex_test_points[18].set(1, 1.);
  hex_test_points[18].set(2, 0.);
  hex_test_points[19].set(0, 0.5);
  hex_test_points[19].set(1, 1.);
  hex_test_points[19].set(2, 0.);
  hex_test_points[20].set(0, 1.);
  hex_test_points[20].set(1, 1.);
  hex_test_points[20].set(0, 0.);
  hex_test_points[21].set(0, 0.);
  hex_test_points[21].set(1, 1.);
  hex_test_points[21].set(2, 0.5);
  hex_test_points[22].set(0, 0.5);
  hex_test_points[22].set(1, 1.);
  hex_test_points[22].set(2, 0.5);
  hex_test_points[23].set(0, 1.);
  hex_test_points[23].set(1, 1.);
  hex_test_points[23].set(2, 0.5);
  hex_test_points[24].set(0, 0.);
  hex_test_points[24].set(1, 1.);
  hex_test_points[24].set(2, 1.);
  hex_test_points[25].set(0, 0.5);
  hex_test_points[25].set(1, 1.);
  hex_test_points[25].set(2, 1.);
  hex_test_points[26].set(0, 1.);
  hex_test_points[26].set(1, 1.);
  hex_test_points[26].set(2, 1.);

       
  std::vector< std::vector< int > > lag_degrees(MAX_DEG_LAG);
  for (int l=0; l < lag_degrees.size(); ++l)
  {
    lag_degrees[l].resize(1,l);
  }
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

      CONSOLE_OUTPUT(rank, "Testing " << filename << " on mesh level: " << init_level + i);
#ifdef TEST_LAG
      // LAG element
      if (gdim == 2 && cell_types[test_number] == RefCellType::TRI_STD)
      { 
        for (int l=0; l<lag_degrees.size(); ++l)
        {
          int deg = lag_degrees[l][0];
          CONSOLE_OUTPUT(rank, "test LAG space of degree " << deg);

          CRefCellSPtr<double, 2> ref_cell_tri = CRefCellSPtr<double, 2>( new RefCellTriStd <double, 2> );
          AnsatzSpaceSPtr<double, 2> ansatz_tri_lag_2 (new PTriLag <double, 2> (ref_cell_tri));
          DofContainerLagSPtr<double, 2> dof_tri_lag (new DofContainerLagrange<double, 2>  (ref_cell_tri));

          ansatz_tri_lag_2->init(deg, 2);
          dof_tri_lag->init(deg, 2);
          RefElement<double, 2> ref_fe;
          FETrafoSPtr<double, 2> fe_trafo(new FETransformationStandard<double, 2> ());
          ref_fe.init (ansatz_tri_lag_2, dof_tri_lag, fe_trafo, false, FEType::LAGRANGE);

          // loop over cells
          for (mesh::EntityIterator it = mesh[i]->begin(2), e_it = mesh[i]->end(2); it != e_it; ++it) 
          {
//          std::cout << "cell index " << it->index() << std::endl;
//          std::cout << "===============" << std::endl;
  
            // Cell
            Entity cell = mesh[i]->get_entity(2, it->index());
            
            // cell trafo
            //const CellTransformation< double, 2 > *c_trafo = rt_space2.fe_manager().get_cell_transformation(it->index());
            CellTrafoSPtr< double, 2 > c_trafo = CellTrafoSPtr< double, 2 >(new LinearTriangleTransformation<double, 2>(ref_cell_tri));
            std::vector<double> coord_vtx;
            it->get_coordinates(coord_vtx);
            c_trafo->reinit(coord_vtx, *it);
                   
            // DOF container
            auto dof = ref_fe.dof_container();
            
            // apply FE transformation (Piola in this case) to map reference nodal basis to physical cell 
            MappingRef2Phys<double, 2, RefElement<double, 2> > * eval_phys 
              = new MappingRef2Phys<double, 2, RefElement<double, 2> > ( &ref_fe, fe_trafo, c_trafo );
    
            // inverse FE transformation
            MappingPhys2Ref <double, 2, MappingRef2Phys<double, 2, RefElement<double, 2> > > * eval_phys_on_ref
              = new MappingPhys2Ref <double, 2, MappingRef2Phys<double, 2, RefElement<double, 2> > > (eval_phys, &cell, fe_trafo, c_trafo);
    
            check_test_points<double, 2> (tri_test_points, &ref_fe, eval_phys_on_ref);
            check_dofs<double, 2> (dof.get(), &ref_fe, eval_phys_on_ref); 
          }
          CONSOLE_OUTPUT(rank, "     passed");
        }
      }
      if (gdim == 3 && cell_types[test_number] == RefCellType::HEX_STD)
      { 
        for (int l=0; l<lag_degrees.size(); ++l)
        {
          int deg = lag_degrees[l][0];
          CONSOLE_OUTPUT(rank, "test LAG space of degree " << deg);

          CRefCellSPtr<double, 3> ref_cell_hex = CRefCellSPtr<double, 3>( new RefCellHexStd <double, 3> );
          AnsatzSpaceSPtr<double, 3> ansatz_hex_lag_3 (new QHexLag <double, 3>  (ref_cell_hex));
          DofContainerLagSPtr<double, 3> dof_hex_lag (new DofContainerLagrange<double, 3>  (ref_cell_hex));

          ansatz_hex_lag_3->init(deg, 3);
          dof_hex_lag->init(deg, 3);
          RefElement<double, 3> ref_fe;
          FETrafoSPtr<double, 3> fe_trafo(new FETransformationStandard<double, 3> ());
          ref_fe.init (ansatz_hex_lag_3, dof_hex_lag, fe_trafo, false, FEType::LAGRANGE);

          // loop over cells
          for (mesh::EntityIterator it = mesh[i]->begin(3), e_it = mesh[i]->end(3); it != e_it; ++it) 
          {
//          std::cout << "cell index " << it->index() << std::endl;
//          std::cout << "===============" << std::endl;
  
            // Cell
            Entity cell = mesh[i]->get_entity(3, it->index());
            
            // cell trafo
            //const CellTransformation< double, 2 > *c_trafo = rt_space2.fe_manager().get_cell_transformation(it->index());
            CellTrafoSPtr< double, 3 > c_trafo = CellTrafoSPtr< double, 3 >(new TriLinearHexahedronTransformation<double, 3>(ref_cell_hex));
            std::vector<double> coord_vtx;
            it->get_coordinates(coord_vtx);
            c_trafo->reinit(coord_vtx, *it);
                   
            // DOF container
            auto dof = ref_fe.dof_container();
            
            // apply FE transformation (Piola in this case) to map reference nodal basis to physical cell 
            MappingRef2Phys<double, 3, RefElement<double, 3> > * eval_phys 
              = new MappingRef2Phys<double, 3, RefElement<double, 3> > ( &ref_fe, fe_trafo, c_trafo );
    
            // inverse FE transformation
            MappingPhys2Ref <double, 3, MappingRef2Phys<double, 3, RefElement<double, 3> > > * eval_phys_on_ref
              = new MappingPhys2Ref <double, 3, MappingRef2Phys<double, 3, RefElement<double, 3> > > (eval_phys, &cell, fe_trafo, c_trafo);
    
            check_test_points<double, 3> (hex_test_points, &ref_fe, eval_phys_on_ref);
            check_dofs<double, 3> (dof.get(), &ref_fe, eval_phys_on_ref); 
          }
          CONSOLE_OUTPUT(rank, "     passed");
        }
      }
      if (gdim == 3 && cell_types[test_number] == RefCellType::TET_STD)
      { 
        for (int l=0; l<lag_degrees.size(); ++l)
        {
          int deg = lag_degrees[l][0];
          CONSOLE_OUTPUT(rank, "test LAG space of degree " << deg);

          CRefCellSPtr<double, 3> ref_cell_tet = CRefCellSPtr<double, 3>( new RefCellTetStd <double, 3> );
          AnsatzSpaceSPtr<double, 3> ansatz_tet_lag_3 (new PTetLag <double, 3> (ref_cell_tet));
          DofContainerLagSPtr<double, 3> dof_tet_lag (new DofContainerLagrange<double, 3>  (ref_cell_tet));

          ansatz_tet_lag_3->init(deg, 3);
          dof_tet_lag->init(deg, 3);
          RefElement<double, 3> ref_fe;
          FETrafoSPtr<double, 3> fe_trafo(new FETransformationStandard<double, 3> ());
          ref_fe.init (ansatz_tet_lag_3, dof_tet_lag, fe_trafo, false, FEType::LAGRANGE);

          // loop over cells
          for (mesh::EntityIterator it = mesh[i]->begin(3), e_it = mesh[i]->end(3); it != e_it; ++it) 
          {
//          std::cout << "cell index " << it->index() << std::endl;
//          std::cout << "===============" << std::endl;
  
            // Cell
            Entity cell = mesh[i]->get_entity(3, it->index());
            
            // cell trafo
            //const CellTransformation< double, 2 > *c_trafo = rt_space2.fe_manager().get_cell_transformation(it->index());
            CellTrafoSPtr< double, 3 > c_trafo = CellTrafoSPtr< double, 3 >(new LinearTetrahedronTransformation<double, 3>(ref_cell_tet));
            std::vector<double> coord_vtx;
            it->get_coordinates(coord_vtx);
            c_trafo->reinit(coord_vtx, *it);
                   
            // DOF container
            auto dof = ref_fe.dof_container();
            
            // apply FE transformation (Piola in this case) to map reference nodal basis to physical cell 
            MappingRef2Phys<double, 3, RefElement<double, 3> > * eval_phys 
              = new MappingRef2Phys<double, 3, RefElement<double, 3> > ( &ref_fe, fe_trafo, c_trafo );
    
            // inverse FE transformation
            MappingPhys2Ref <double, 3, MappingRef2Phys<double, 3, RefElement<double, 3> > > * eval_phys_on_ref
              = new MappingPhys2Ref <double, 3, MappingRef2Phys<double, 3, RefElement<double, 3> > > (eval_phys, &cell, fe_trafo, c_trafo);
    
            check_test_points<double, 3> (tet_test_points, &ref_fe, eval_phys_on_ref);
            check_dofs<double, 3> (dof.get(), &ref_fe, eval_phys_on_ref); 
          }
          CONSOLE_OUTPUT(rank, "     passed");
        }
      }
#endif
#ifdef TEST_BDM
      /// BDM element
      if (gdim == 2 && cell_types[test_number] == RefCellType::TRI_STD)
      { 
        for (int l=0; l<bdm_degrees.size(); ++l)
        {
          int deg = bdm_degrees[l][0];
          CONSOLE_OUTPUT(rank, "test BDM space of degree " << deg);

          CRefCellSPtr<double, 2> ref_cell_tri = CRefCellSPtr<double, 2>( new RefCellTriStd <double, 2> );
          AnsatzSpaceSPtr<double, 2> ansatz_tri_lag_2 (new PTriLag <double, 2> (ref_cell_tri));
          ansatz_tri_lag_2->init(deg, 2);
  
          DofContainerRtBdmSPtr<double, 2> dof_tri_bdm (new DofContainerRTBDM<double, 2> (ref_cell_tri));
          dof_tri_bdm->init(deg, DofContainerType::BDM); 
          
          RefElement<double, 2> ref_fe;
          FETrafoSPtr<double, 2> fe_trafo(new FETransformationContraPiola<double, 2> ());
          ref_fe.init (ansatz_tri_lag_2, dof_tri_bdm, fe_trafo, false, FEType::BDM);
  
          // loop over cells
          for (mesh::EntityIterator it = mesh[i]->begin(2), e_it = mesh[i]->end(2); it != e_it; ++it) 
          {
//          std::cout << "cell index " << it->index() << std::endl;
//          std::cout << "===============" << std::endl;
  
            // Cell
            Entity cell = mesh[i]->get_entity(2, it->index());
            
            // cell trafo
            //const CellTransformation< double, 2 > *c_trafo = rt_space2.fe_manager().get_cell_transformation(it->index());
            CellTrafoSPtr< double, 2 > c_trafo = CellTrafoSPtr< double, 2 >(new LinearTriangleTransformation<double, 2>(ref_cell_tri));
            std::vector<double> coord_vtx;
            it->get_coordinates(coord_vtx);
            c_trafo->reinit(coord_vtx, *it);
            
            // DOF container
            auto dof = ref_fe.dof_container();
            
            // apply FE transformation (Piola in this case) to map reference nodal basis to physical cell 
            MappingRef2Phys<double, 2, RefElement<double, 2> > * eval_phys 
              = new MappingRef2Phys<double, 2, RefElement<double, 2> > ( &ref_fe, fe_trafo, c_trafo );
    
            // inverse FE transformation
            MappingPhys2Ref <double, 2, MappingRef2Phys<double, 2, RefElement<double, 2> > > * eval_phys_on_ref
              = new MappingPhys2Ref <double, 2, MappingRef2Phys<double, 2, RefElement<double, 2> > > (eval_phys, &cell, fe_trafo, c_trafo);
    
            check_test_points<double,2> (tri_test_points, &ref_fe, eval_phys_on_ref);
            check_dofs<double,2> (dof.get(), &ref_fe, eval_phys_on_ref);     
          }
          CONSOLE_OUTPUT(rank, "     passed");
        }
      }
#endif
#ifdef TEST_RT
      // RT element
      if (gdim == 2 && cell_types[test_number] == RefCellType::TRI_STD)
      { 
        for (int l=0; l<rt_degrees.size(); ++l)
        {
          int deg = rt_degrees[l][0];
          CONSOLE_OUTPUT(rank, "test RT space of degree " << deg);

          CRefCellSPtr<double, 2> ref_cell_tri = CRefCellSPtr<double, 2>( new RefCellTriStd <double, 2> );
          AnsatzSpaceSPtr<double, 2> ansatz_tri_lag_2 (new PTriLag <double, 2> (ref_cell_tri));

          AnsatzSpaceSumSPtr<double, 2> ansatz_tri_rt (new AnsatzSpaceSum<double, 2> (ref_cell_tri));
          DofContainerRtBdmSPtr<double, 2> dof_tri_rt (new DofContainerRTBDM<double, 2> (ref_cell_tri));
        
          AnsatzSpaceSPtr<double, 2> ansatz1(new PTriLag<double, 2> (ref_cell_tri));
          AnsatzSpaceSPtr<double, 2> ansatz2(new AugPTriMono<double, 2> (ref_cell_tri));
          
          ansatz1->init(deg,2);
          ansatz2->init(deg);

          ansatz_tri_rt->init(ansatz1, ansatz2, AnsatzSpaceType::RT);
          dof_tri_rt->init(deg, DofContainerType::RT);  
          RefElement<double, 2> ref_fe;
          FETrafoSPtr<double, 2> fe_trafo(new FETransformationContraPiola<double, 2> ());
          ref_fe.init (ansatz_tri_rt, dof_tri_rt, fe_trafo, false, FEType::RT);
  
          // loop over cells
          for (mesh::EntityIterator it = mesh[i]->begin(2), e_it = mesh[i]->end(2); it != e_it; ++it) 
          {
//          std::cout << "cell index " << it->index() << std::endl;
//          std::cout << "===============" << std::endl;
  
            // Cell
            Entity cell = mesh[i]->get_entity(2, it->index());
            
            // cell trafo
            //const CellTransformation< double, 2 > *c_trafo = rt_space2.fe_manager().get_cell_transformation(it->index());
            CellTrafoSPtr< double, 2 > c_trafo = CellTrafoSPtr< double, 2 > (new LinearTriangleTransformation<double, 2>(ref_cell_tri));
            std::vector<double> coord_vtx;
            it->get_coordinates(coord_vtx);
            c_trafo->reinit(coord_vtx, *it);
            
            // DOF container
            auto dof = ref_fe.dof_container();
            
            // apply FE transformation (Piola in this case) to map reference nodal basis to physical cell 
            MappingRef2Phys<double, 2, RefElement<double, 2> > * eval_phys 
              = new MappingRef2Phys<double, 2, RefElement<double, 2> > ( &ref_fe, fe_trafo, c_trafo );
    
            // inverse FE transformation
            MappingPhys2Ref <double, 2, MappingRef2Phys<double, 2, RefElement<double, 2> > > * eval_phys_on_ref
              = new MappingPhys2Ref <double, 2, MappingRef2Phys<double, 2, RefElement<double, 2> > > (eval_phys, &cell, fe_trafo, c_trafo);
    
            check_test_points<double,2> (tri_test_points, &ref_fe, eval_phys_on_ref);
            check_dofs<double,2> (dof.get(), &ref_fe, eval_phys_on_ref); 
          }
          CONSOLE_OUTPUT(rank, "     passed");
        }
      }
      // RT element on quad
      if (gdim == 2 && cell_types[test_number] == RefCellType::QUAD_STD)
      { 
        for (int l=0; l<rt_degrees.size(); ++l)
        {
          int deg = rt_degrees[l][0];
          CONSOLE_OUTPUT(rank, "test RT space of degree " << deg);

          CRefCellSPtr<double, 2> ref_cell_quad = CRefCellSPtr<double, 2>( new RefCellQuadStd <double, 2> );
          AnsatzSpaceSPtr<double, 2> ansatz_quad_rt (new QQuadLag <double, 2> (ref_cell_quad));

          std::vector< std::vector <size_t> > degrees;
          degrees.resize(2);
          degrees[0].resize(2);
          degrees[1].resize(2);
          degrees[0][0] = deg + 1;
          degrees[0][1] = deg;
          degrees[1][0] = deg;
          degrees[1][1] = deg + 1;
          ansatz_quad_rt->init(degrees);

          DofContainerRtBdmSPtr<double, 2> dof_quad_rt (new DofContainerRTBDM<double, 2> (ref_cell_quad));
        
          dof_quad_rt->init(deg, DofContainerType::RT);  
          RefElement<double, 2> ref_fe;
          FETrafoSPtr<double, 2> fe_trafo(new FETransformationContraPiola<double, 2> ());
          ref_fe.init (ansatz_quad_rt, dof_quad_rt, fe_trafo, false, FEType::RT);
  
          // loop over cells
          for (mesh::EntityIterator it = mesh[i]->begin(2), e_it = mesh[i]->end(2); it != e_it; ++it) 
          {
//          std::cout << "cell index " << it->index() << std::endl;
//          std::cout << "===============" << std::endl;
  
            // Cell
            Entity cell = mesh[i]->get_entity(2, it->index());
            
            // cell trafo
            //const CellTransformation< double, 2 > *c_trafo = rt_space2.fe_manager().get_cell_transformation(it->index());
            CellTrafoSPtr< double, 2 > c_trafo = CellTrafoSPtr< double, 2 > (new BiLinearQuadTransformation<double, 2>(ref_cell_quad));
            std::vector<double> coord_vtx;
            it->get_coordinates(coord_vtx);
            c_trafo->reinit(coord_vtx, *it);
            
            // DOF container
            auto dof = ref_fe.dof_container();
            
            // apply FE transformation (Piola in this case) to map reference nodal basis to physical cell 
            MappingRef2Phys<double, 2, RefElement<double, 2> > * eval_phys 
              = new MappingRef2Phys<double, 2, RefElement<double, 2> > ( &ref_fe, fe_trafo, c_trafo );
    
            // inverse FE transformation
            MappingPhys2Ref <double, 2, MappingRef2Phys<double, 2, RefElement<double, 2> > > * eval_phys_on_ref
              = new MappingPhys2Ref <double, 2, MappingRef2Phys<double, 2, RefElement<double, 2> > > (eval_phys, &cell, fe_trafo, c_trafo);
    
            check_test_points<double,2> (quad_test_points, &ref_fe, eval_phys_on_ref);
            check_dofs<double,2> (dof.get(), &ref_fe, eval_phys_on_ref); 
          }
          CONSOLE_OUTPUT(rank, "     passed");
        }
      }

      // RT element on tet
      if (gdim == 3 && cell_types[test_number] == RefCellType::TET_STD)
      { 
        for (int l=0; l<rt_degrees.size(); ++l)
        {
          int deg = rt_degrees[l][0];
          CONSOLE_OUTPUT(rank, "test RT space of degree " << deg);

          CRefCellSPtr<double, 3> ref_cell_tet = CRefCellSPtr<double, 3>( new RefCellTetStd <double, 3> );
          AnsatzSpaceSPtr<double, 3> ansatz_tet_lag_2 (new PTetLag <double, 3> (ref_cell_tet));

          AnsatzSpaceSumSPtr<double, 3> ansatz_tet_rt (new AnsatzSpaceSum<double, 3> (ref_cell_tet));
          DofContainerRtBdmSPtr<double, 3> dof_tet_rt (new DofContainerRTBDM<double, 3> (ref_cell_tet));
        
          AnsatzSpaceSPtr<double, 3> ansatz1( new PTetLag<double, 3> (ref_cell_tet));
          AnsatzSpaceSPtr<double, 3> ansatz2( new AugPTetMono<double, 3> (ref_cell_tet));
          
          ansatz1->init(deg,3);
          ansatz2->init(deg);

          ansatz_tet_rt->init(ansatz1, ansatz2, AnsatzSpaceType::RT);
          dof_tet_rt->init(deg, DofContainerType::RT);  
          RefElement<double, 3> ref_fe;
          FETrafoSPtr<double, 3> fe_trafo(new FETransformationContraPiola<double, 3> ());
          ref_fe.init (ansatz_tet_rt, dof_tet_rt, fe_trafo, false, FEType::RT);
  
          // loop over cells
          for (mesh::EntityIterator it = mesh[i]->begin(3), e_it = mesh[i]->end(3); it != e_it; ++it) 
          {
//          std::cout << "cell index " << it->index() << std::endl;
//          std::cout << "===============" << std::endl;
  
            // Cell
            Entity cell = mesh[i]->get_entity(3, it->index());
            
            // cell trafo
            //const CellTransformation< double, 2 > *c_trafo = rt_space2.fe_manager().get_cell_transformation(it->index());
            CellTrafoSPtr< double, 3 > c_trafo = CellTrafoSPtr< double, 3 > (new LinearTetrahedronTransformation<double, 3>(ref_cell_tet));
            std::vector<double> coord_vtx;
            it->get_coordinates(coord_vtx);
            c_trafo->reinit(coord_vtx, *it);
            
            // DOF container
            auto dof = ref_fe.dof_container();
            
            // apply FE transformation (Piola in this case) to map reference nodal basis to physical cell 
            MappingRef2Phys<double, 3, RefElement<double, 3> > * eval_phys 
              = new MappingRef2Phys<double, 3, RefElement<double, 3> > ( &ref_fe, fe_trafo, c_trafo );
    
            // inverse FE transformation
            MappingPhys2Ref <double, 3, MappingRef2Phys<double, 3, RefElement<double, 3> > > * eval_phys_on_ref
              = new MappingPhys2Ref <double, 3, MappingRef2Phys<double, 3, RefElement<double, 3> > > (eval_phys, &cell, fe_trafo, c_trafo);
    
            check_test_points<double,3> (tet_test_points, &ref_fe, eval_phys_on_ref);
            check_dofs<double,3> (dof.get(), &ref_fe, eval_phys_on_ref); 
          }
          CONSOLE_OUTPUT(rank, "     passed");
        }
      }

      // RT element on hex
      if (gdim == 3 && cell_types[test_number] == RefCellType::HEX_STD)
      { 
        for (int l=0; l<rt_degrees.size(); ++l)
        {
          int deg = rt_degrees[l][0];
          CONSOLE_OUTPUT(rank, "test RT space of degree " << deg);

          CRefCellSPtr<double, 3> ref_cell_hex = CRefCellSPtr<double, 3>( new RefCellHexStd <double, 3> );
          AnsatzSpaceSPtr<double, 3> ansatz_hex_rt (new QHexLag <double, 3> (ref_cell_hex));

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

          DofContainerRtBdmSPtr<double, 3> dof_hex_rt (new DofContainerRTBDM<double, 3> (ref_cell_hex));
        
          dof_hex_rt->init(deg, DofContainerType::RT);  
          RefElement<double, 3> ref_fe;
          FETrafoSPtr<double, 3> fe_trafo(new FETransformationContraPiola<double, 3> ());
          ref_fe.init (ansatz_hex_rt, dof_hex_rt, fe_trafo, false, FEType::RT);
  
          const int num_q = dof_hex_rt->cell_quad_size();
          std::vector< Vec<3, double> > cell_quad_pts(num_q);
          for (int q=0; q!=num_q; ++q)
          {
            cell_quad_pts[q] = dof_hex_rt->cell_quad_point(q);
          }
          
          // loop over cells
          for (mesh::EntityIterator it = mesh[i]->begin(3), e_it = mesh[i]->end(3); it != e_it; ++it) 
          {
            //std::cout << "===============" << std::endl;
            //std::cout << "cell index " << it->index() << std::endl;
  
            // Cell
            Entity cell = mesh[i]->get_entity(3, it->index());
            
            // cell trafo
            //const CellTransformation< double, 2 > *c_trafo = rt_space2.fe_manager().get_cell_transformation(it->index());
            CellTrafoSPtr< double, 3 > c_trafo = CellTrafoSPtr< double, 3 > (new TriLinearHexahedronTransformation<double, 3>(ref_cell_hex));
            CellTrafoSPtr< double, 3 > c_trafo_lin = CellTrafoSPtr< double, 3 > (new LinearHexahedronTransformation<double, 3>(ref_cell_hex));
            
            std::vector<double> coord_vtx;
            it->get_coordinates(coord_vtx);
            c_trafo->reinit(coord_vtx, *it);
            c_trafo_lin->reinit(coord_vtx, *it);
            
            // DOF container
            auto dof = ref_fe.dof_container();
            
            // apply FE transformation (Piola in this case) to map reference nodal basis to physical cell 
            MappingRef2Phys<double, 3, RefElement<double, 3> > * eval_phys 
              = new MappingRef2Phys<double, 3, RefElement<double, 3> > ( &ref_fe, fe_trafo, c_trafo );
    
            // inverse FE transformation
            MappingPhys2Ref <double, 3, MappingRef2Phys<double, 3, RefElement<double, 3> > > * eval_phys_on_ref
              = new MappingPhys2Ref <double, 3, MappingRef2Phys<double, 3, RefElement<double, 3> > > (eval_phys, &cell, fe_trafo, c_trafo);
    
            //check_test_points<double,3> (cell_quad_pts, &ref_fe, eval_phys_on_ref);
            check_test_points<double,3> (hex_test_points, &ref_fe, eval_phys_on_ref);
            check_dofs<double,3>        (dof.get(), &ref_fe, eval_phys_on_ref); 
            
            //check_test_points<double,3> (cell_quad_pts, c_trafo, c_trafo_lin);
          }
          CONSOLE_OUTPUT(rank, "     passed");
        }
      }
#endif

      
      CONSOLE_OUTPUT(rank, "===============================");
    }

  }

  MPI_Finalize();

  return;
}
