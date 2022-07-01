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

/// \author Martin Baumann, Thomas Gengenbach, Michael Schick
#define BOOST_TEST_MODULE fe_hdiv

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
using namespace hiflow::la;

/// H(div) conformity test
///
/// \brief For a list of files it is analysed, whether the FE ansatz space
/// has continuous normal component
///

const int DEBUG_OUT = 0;
const double pi = 3.14159265;

template <class DataType, int DIM>
struct TestFunction
{
  size_t nb_func() const 
  {
    return 1;
  }
  
  size_t nb_comp() const 
  {
    return DIM;
  }
  
  size_t weight_size() const 
  {
    return DIM;
  }
  
  inline size_t iv2ind (size_t i, size_t var ) const 
  {
    assert (i==0);
    assert (var < DIM);
    return var;
  }
  
  void evaluate(const Entity& cell, const Vec<DIM, DataType> & x, std::vector<DataType>& vals) const
  {
    assert (DIM >= 2);
    assert (space_ != nullptr);
    vals.clear();
    vals.resize(this->nb_comp(), 0.);
    
    //Vec<DIM, DataType> x;
    //space_->get_cell_transformation(cell.index()).transform(ref_pt, x);
    
    vals[0] = std::sin(pi * x[0]) * std::sin(pi * x[0]) 
            * std::sin(2. * pi * x[1]);

    vals[1] = - std::sin(2. * pi * x[0]) 
            * std::sin(pi * x[1]) * std::sin(pi * x[1]);
    
    if (DIM == 3)
    {
      vals[2] = 1.;
    }
  }
  
  VectorSpace<DataType, DIM>* space_;
};

template <class DataType, int DIM>
void check_normal_comp (VectorSpace<DataType,2> &space2,
		        VectorSpace<DataType,3> &space3,
                        MeshPtr mesh_ptr,
                        RefCellType c_type)
{
  // define fe coeff vector
  CoupledVector<DataType> sol;
  if (DIM == 2)
    sol.Init(MPI_COMM_WORLD, space2.la_couplings());
  else  
    sol.Init(MPI_COMM_WORLD, space3.la_couplings());

  sol.Zeros();
   
  TestFunction<DataType, 2> test_func2;
  TestFunction<DataType, 3> test_func3;
  if (DIM == 2)
    test_func2.space_ = &space2;
  else
    test_func3.space_ = &space3;
     
  if (DIM == 2) 
  {
    FeInterNodal<DataType, 2, TestFunction<DataType, 2> > fe_inter (space2, &test_func2, 0);
    fe_inter.interpolate (sol);
  }
  else
  {
    FeInterNodal<DataType, 3, TestFunction<DataType, 3> > fe_inter (space3, &test_func3, 0);
    fe_inter.interpolate (sol);
  }
      
  //space.interpolate_function (0, test_func, sol);
  
  /*
  int nlocal = sol.size_local();
  for (int i=0; i<nlocal; ++i)
  {
    sol.SetValue(i, static_cast<DataType> (i));
  }
  */
        
  // loop over mesh interfaces
  for (mesh::EntityIterator it = mesh_ptr->begin(DIM-1), e_it = mesh_ptr->end(DIM-1); it != e_it; ++it) 
  {
    // get incident cells
    std::vector<int> cell_ind;
    for (mesh::IncidentEntityIterator c_it = it->begin_incident(DIM),
         ce_it = it->end_incident(DIM); c_it != ce_it; ++c_it)
    {
      cell_ind.push_back(c_it->index());
    }
    if (cell_ind.size() != 2)
    {
      continue;
    }
                        
    Entity l_cell = mesh_ptr->get_entity(DIM, cell_ind[0]);
    Entity r_cell = mesh_ptr->get_entity(DIM, cell_ind[1]);
            
    std::vector<DataType> l_coords;
    std::vector<DataType> r_coords;
            
    l_cell.get_coordinates(l_coords);
    r_cell.get_coordinates(r_coords);
          
    CellTransformation<DataType, 2> * l_trafo2;
    CellTransformation<DataType, 2> * r_trafo2;
    CellTransformation<DataType, 3> * l_trafo3;
    CellTransformation<DataType, 3> * r_trafo3;
    CRefCellSPtr<DataType, 2>  ref_cell_tri = CRefCellSPtr<DataType, 2> (new RefCellTriStd <DataType, 2>);
    CRefCellSPtr<DataType, 2>  ref_cell_quad = CRefCellSPtr<DataType, 2> (new RefCellQuadStd <DataType, 2>);
    CRefCellSPtr<DataType, 3>  ref_cell_tet = CRefCellSPtr<DataType, 3> (new RefCellTetStd <DataType, 3>);
    CRefCellSPtr<DataType, 3>  ref_cell_hex = CRefCellSPtr<DataType, 3> (new RefCellHexStd <DataType, 3>);
    
    if (c_type == RefCellType::TRI_STD)
    {
      l_trafo2 = new LinearTriangleTransformation<DataType, 2> (ref_cell_tri);
      r_trafo2 = new LinearTriangleTransformation<DataType, 2> (ref_cell_tri);
    }
    else if (c_type == RefCellType::QUAD_STD)
    {  
      l_trafo2 = new BiLinearQuadTransformation<DataType, 2> (ref_cell_quad);
      r_trafo2 = new BiLinearQuadTransformation<DataType, 2> (ref_cell_quad);
    }
    else if (c_type == RefCellType::TET_STD)
    {  
      l_trafo3 = new LinearTetrahedronTransformation<DataType, 3> (ref_cell_tet);
      r_trafo3 = new LinearTetrahedronTransformation<DataType, 3> (ref_cell_tet);
    }
    else if (c_type == RefCellType::HEX_STD)
    {  
      l_trafo3 = new TriLinearHexahedronTransformation<DataType, 3> (ref_cell_hex);
      r_trafo3 = new TriLinearHexahedronTransformation<DataType, 3> (ref_cell_hex);
    }

    else
    {
      assert(false);
    }
    
    if(DIM == 2) 
    {
      l_trafo2->reinit(l_coords);
      r_trafo2->reinit(r_coords);
    }
    else
    {
      l_trafo3->reinit(l_coords);
      r_trafo3->reinit(r_coords);
    }

    // get mid point and normal of edge
    std::vector< DataType > coords;
    it->get_coordinates(coords);
            
    //Vec<DIM,DataType> mid_point;
    Vec<2,DataType> tangent;
    Vec<3,DataType> tangent1;
    Vec<3,DataType> tangent2;
    Vec<2, DataType> n2;
    Vec<3, DataType> n3;
    
    if (c_type == RefCellType::TRI_STD || c_type == RefCellType::QUAD_STD)
    {
      /*mid_point.set(0, 0.5 *(coords[0] + coords[2]));
      mid_point[1] = 0.5 *(coords[1] + coords[3]);*/
      
      tangent.set(0, coords[2] - coords[0]);
      tangent.set(1, coords[3] - coords[1]);

      n2 = normal(tangent);
    }
    else if (c_type == RefCellType::TET_STD)
    { 
      tangent1.set(0, coords[3] - coords[0]);
      tangent1.set(1, coords[4] - coords[1]);
      tangent1.set(2, coords[5] - coords[2]);

      tangent2.set(0, coords[6] - coords[0]);
      tangent2.set(1, coords[7] - coords[1]);
      tangent2.set(2, coords[8] - coords[2]); 

      n3 = normal(tangent1, tangent2);
    }
    else if (c_type == RefCellType::HEX_STD)
    { 
      tangent1.set(0, coords[3] - coords[0]);
      tangent1.set(1, coords[4] - coords[1]);
      tangent1.set(2, coords[5] - coords[2]);

      tangent2.set(0, coords[9] - coords[0]);
      tangent2.set(1, coords[10] - coords[1]);
      tangent2.set(2, coords[11] - coords[2]); 

      n3 = normal(tangent1, tangent2);
    }
    else
    {
      assert(false);
    }    
    
    std::vector< DataType > midpoint_coords;
    it->get_midpoint(midpoint_coords);
    Vec<DIM, DataType> mid_point(midpoint_coords);
    
    if (DEBUG_OUT >= 2)
    {
      std::cout << " ============================== "<< std::endl;
      std::cout << "Mid Point ";
      for (size_t d=0; d<DIM; ++d)
      {
        std::cout << mid_point[d] << " ";
      }
      std::cout << std::endl;
      std::cout << "Tangent(s)";
      if (DIM == 2)
      {
        for (size_t d=0; d<DIM; ++d)
        {
          std::cout << tangent[d] << " ";
        }
        std::cout << std::endl;
      }
      else
      {
        for (size_t d=0; d<DIM; ++d)
        {
          std::cout << tangent1[d] << " ";
        }
        std::cout << std::endl;
        for (size_t d=0; d<DIM; ++d)
        {
          std::cout << tangent2[d] << " ";
        }
        std::cout << std::endl;
      }
      std::cout << "Normal ";
      if (DIM == 2)
      {
        for (size_t d=0; d<DIM; ++d)
        {
          std::cout << n2[d] << " ";
        }
        std::cout << std::endl;
      }
      else
      {
        for (size_t d=0; d<DIM; ++d)
        {
          std::cout << n3[d] << " ";
        }
        std::cout << std::endl;
      }
    }
    
    DataType eps = 1e-8;
    Vec<2,DataType> a_point2;
    Vec<3,DataType> a_point3;
    if(DIM == 2)
    {
      a_point2.set(0, mid_point[0]);
      a_point2.set(1, mid_point[1]);
    }
    else
    {
      a_point3.set(0, mid_point[0]);
      a_point3.set(1, mid_point[1]);
      a_point3.set(2, mid_point[2]);
    }
    if (DIM == 2)
      a_point2 += eps*n2;
    else  
      a_point3 += eps*n3;

    Vec<2,DataType> b_point2;
    Vec<3,DataType> b_point3;
    if(DIM == 2)
    {
      b_point2.set(0, mid_point[0]);
      b_point2.set(1, mid_point[1]);
    }
    else
    {
      b_point3.set(0, mid_point[0]);
      b_point3.set(1, mid_point[1]);
      b_point3.set(2, mid_point[2]);
    }

    if(DIM == 2)
      b_point2 -= eps*n2;
    else
      b_point3 -= eps*n3;

    Vec<2,DataType> l_point_ref2;
    Vec<2,DataType> r_point_ref2;
    Vec<2,DataType> tmp2;
    Vec<2,DataType> l_point2;
    Vec<2,DataType> r_point2;

    Vec<3,DataType> l_point_ref3;
    Vec<3,DataType> r_point_ref3;
    Vec<3,DataType> tmp3;
    Vec<3,DataType> l_point3;
    Vec<3,DataType> r_point3;

    if(DIM == 2) 
    {
      if (l_trafo2->contains_physical_point(a_point2, tmp2))
      {
        l_point2 = a_point2;
        r_point2 = b_point2;
      }
      else
      {
        l_point2 = b_point2;
        r_point2 = a_point2;
      }
    
      l_trafo2->inverse(l_point2, l_point_ref2);
      r_trafo2->inverse(r_point2, r_point_ref2);
    }
    else
    {
      if (l_trafo3->contains_physical_point(a_point3, tmp3))
      {
        l_point3 = a_point3;
        r_point3 = b_point3;
      }
      else
      {
        l_point3 = b_point3;
        r_point3 = a_point3;
      }
    
      l_trafo3->inverse(l_point3, l_point_ref3);
      r_trafo3->inverse(r_point3, r_point_ref3);
    }

    if (DEBUG_OUT >= 2)
    {
      std::cout << "Left Point ";
      if (DIM == 2)	 
      {	      
        for (size_t d=0; d<DIM; ++d)
        {
          std::cout << l_point2[d] << " ";
        }
        std::cout << std::endl;
        std::cout << "Right Point ";
        for (size_t d=0; d<DIM; ++d)
        {
          std::cout << r_point2[d] << " ";
        }
        std::cout << std::endl;
      }
      else
      {	      
        for (size_t d=0; d<DIM; ++d)
        {
          std::cout << l_point2[d] << " ";
        }
        std::cout << std::endl;
        std::cout << "Right Point ";
        for (size_t d=0; d<DIM; ++d)
        {
          std::cout << r_point2[d] << " ";
        }
        std::cout << std::endl;
      }
    }
    
    if(DIM == 2)
    {	    
      assert (l_trafo2->contains_physical_point(l_point2, tmp2));
      assert (r_trafo2->contains_physical_point(r_point2, tmp2));
    }
    else
    {
      assert (l_trafo3->contains_physical_point(l_point3, tmp3));
      assert (r_trafo3->contains_physical_point(r_point3, tmp3));
    }
    // evaluate FE function on both sides of interface
    std::vector<DataType> l_val; 
    std::vector<DataType> r_val;
    if (DIM == 2)
    { 
      FeEvalCell<DataType, 2> fe_eval2(space2, sol, 0); 
      fe_eval2.evaluate(r_cell, r_point2, r_val);
      fe_eval2.evaluate(l_cell, l_point2, l_val);
    }
    else
    { 
      FeEvalCell<DataType, 3> fe_eval3(space3, sol, 0); 
      fe_eval3.evaluate(r_cell, r_point3, r_val);
      fe_eval3.evaluate(l_cell, l_point3, l_val);
    }
       
    std::vector<DataType> func_r_val;
    std::vector<DataType> func_l_val;

    if (DIM == 2)
    {
      test_func2.evaluate (r_cell, r_point2, func_r_val);
      test_func2.evaluate (l_cell, l_point2, func_l_val);
    }
    else
    {
      test_func3.evaluate (r_cell, r_point3, func_r_val);
      test_func3.evaluate (l_cell, l_point3, func_l_val);
    }

    assert (l_val.size() == DIM);
    assert (r_val.size() == DIM);
    assert (func_l_val.size() == DIM);
    assert (func_r_val.size() == DIM);
                
    DataType n_times_lval = 0.;
    DataType n_times_rval = 0.;
    DataType n_times_func_lval = 0.;
    DataType n_times_func_rval = 0.;
    
    if(DIM == 2)
    {
      for (size_t d=0; d<DIM; ++d)
      {
        n_times_lval += n2[d] * l_val[d];
        n_times_rval += n2[d] * r_val[d];
      
        n_times_func_lval += n2[d] * func_l_val[d];
        n_times_func_rval += n2[d] * func_r_val[d];
      }
    }
    else
    {
      for (size_t d=0; d<DIM; ++d)
      {
        n_times_lval += n3[d] * l_val[d];
        n_times_rval += n3[d] * r_val[d];
      
        n_times_func_lval += n3[d] * func_l_val[d];
        n_times_func_rval += n3[d] * func_r_val[d];
      }
    }

    if (DEBUG_OUT >= 2)
    {
      for (size_t d=0; d<DIM; ++d)
      {
        std::cout << d << ": " << l_val[d] << " <> " << r_val[d] << std::endl;
      }
    }
    
    if (DEBUG_OUT >= 1)
    {
      std::cout << "  left normal comp " << std::fixed << std::setw( 7 ) << std::setprecision( 4 ) 
                << n_times_lval << " ,   right normal comp " << n_times_rval << std::endl
                << "f left normal comp " << std::fixed << std::setw( 7 ) << std::setprecision( 4 ) 
                << n_times_func_lval << " , f right normal comp " << n_times_func_rval 
                << std::endl;
    }
    BOOST_TEST(std::abs(n_times_lval - n_times_rval)< 1e-6);
  }
}
        
static const char *datadir = MESH_DATADIR;

BOOST_AUTO_TEST_CASE(fe_hdiv) {

  int argc = boost::unit_test::framework::master_test_suite().argc;
  char** argv = boost::unit_test::framework::master_test_suite().argv;
  
  int init_level = 3;
  int num_level = 1;
  int rank;
  MPI_Init(&argc, &argv);

  MPI_Comm_rank(MPI_COMM_WORLD,  &rank);

  // Which files should be checked?

  std::vector< std::string > filenames;
  std::vector< TDim > tdims;
  std::vector< GDim > gdims;
  std::vector< RefCellType > cell_types;
  
  filenames.push_back(std::string(datadir) + std::string("two_triangles_2d.inp"));
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
 
  VectorSpace< double, 2 > rt_space2;
  VectorSpace< double, 3 > rt_space3; 
  VectorSpace< double, 2 > bdm_space2;
  VectorSpace< double, 3 > bdm_space3; 

  std::vector< FEType > rt_fe_ansatz (1, FEType::RT);
  std::vector< FEType > bdm_fe_ansatz (1, FEType::BDM);
  std::vector< bool > is_cg(1, true);
      
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
        
      // BDM element
      if (gdim == 2 && cell_types[test_number] == RefCellType::TRI_STD)
      {
        for (int l=0; l<bdm_degrees.size(); ++l)
        {
          CONSOLE_OUTPUT(rank, "test BDM space of degree " << bdm_degrees[l][0]);
          bdm_space2.Init(*mesh[i], bdm_fe_ansatz, is_cg, bdm_degrees[l]);
          check_normal_comp<double, 2> (bdm_space2, bdm_space3, mesh[i], cell_types[test_number]);
        }
      }
      // RT element
      if (gdim == 2 && cell_types[test_number] == RefCellType::TRI_STD)
      {
        for (int l=0; l<rt_degrees.size(); ++l)
        {
          CONSOLE_OUTPUT(rank, "test RT space of degree " << rt_degrees[l][0]);
          rt_space2.Init(*mesh[i], rt_fe_ansatz, is_cg, rt_degrees[l]);
          check_normal_comp<double, 2> (rt_space2, rt_space3, mesh[i], cell_types[test_number]);
        }
      }
      // RT element on quad
      if (gdim == 2 && cell_types[test_number] == RefCellType::QUAD_STD)
      {
        for (int l=0; l<rt_degrees.size(); ++l)
        {
          CONSOLE_OUTPUT(rank, "test RT space of degree " << rt_degrees[l][0]);
          rt_space2.Init(*mesh[i], rt_fe_ansatz, is_cg, rt_degrees[l]);
          check_normal_comp<double, 2> (rt_space2, rt_space3, mesh[i], cell_types[test_number]);
        }
      }
      // RT element on tet
      if (gdim == 3 && cell_types[test_number] == RefCellType::TET_STD)
      {
        for (int l=0; l<rt_degrees.size(); ++l)
        {
          CONSOLE_OUTPUT(rank, "test RT space of degree " << rt_degrees[l][0]);
          rt_space3.Init(*mesh[i], rt_fe_ansatz, is_cg, rt_degrees[l]);
          check_normal_comp<double, 3> (rt_space2, rt_space3, mesh[i], cell_types[test_number]);
        }
      }
      // RT element on hex
      if (gdim == 3 && cell_types[test_number] == RefCellType::HEX_STD)
      {
        for (int l=0; l<rt_degrees.size(); ++l)
        {
          CONSOLE_OUTPUT(rank, "test RT space of degree " << rt_degrees[l][0]);
          rt_space3.Init(*mesh[i], rt_fe_ansatz, is_cg, rt_degrees[l]);
          check_normal_comp<double, 3> (rt_space2, rt_space3, mesh[i], cell_types[test_number]);
        }
      }

      CONSOLE_OUTPUT(rank, "===============================" );
    }
  }

  MPI_Finalize();

  return;
}
