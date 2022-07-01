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

#define BOOST_TEST_MODULE fe_interpol_eval

#include <boost/test/unit_test.hpp>
#include <boost/test/included/unit_test.hpp>
namespace utf = boost::unit_test;

#define USE_GPERF 

#ifdef USE_GPERF
#include "profiler.h"
#endif

#ifdef LIKWID_PERFMON
#include <likwid-marker.h>
#else
#define LIKWID_MARKER_INIT
#define LIKWID_MARKER_THREADINIT
#define LIKWID_MARKER_SWITCH
#define LIKWID_MARKER_REGISTER(regionTag)
#define LIKWID_MARKER_START(regionTag)
#define LIKWID_MARKER_STOP(regionTag)
#define LIKWID_MARKER_CLOSE
#define LIKWID_MARKER_GET(regionTag, nevents, events, time, count)
#endif

#include <mpi.h>
#include <string>
#include <vector>

#include "hiflow.h"
#include "test.h"

#define nVISUALIZE

#define nTEST_BDM
#define TEST_LAG
#define nTEST_RT

#define nTEST_TRI
#define TEST_QUAD
#define nTEST_TET
#define TEST_HEX

#define TEST_MAPPING_FULL 
#define TEST_MAPPING_REDUCED
#define TEST_GLOBAL
#define nTEST_GLOBAL_BASIS

#define TEST_MAP

<<<<<<< HEAD
int NUM_LEVEL = 6;

int MAX_DEG_LAG = 2;
=======
int NUM_LEVEL = 5;

int MAX_DEG_LAG = 1;
>>>>>>> v3_glASM
int MAX_DEG_BDM = 1;
int MAX_DEG_RT = 1;

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
struct TestFunction 
{
  void set_components(std::vector<size_t> comps)
  {
    this->comps_ = comps;
  }
  
  size_t nb_func() const 
  {
    return 1;
  }
  
  size_t nb_comp() const 
  {
    return comps_.size();
  }
  
  size_t weight_size() const 
  {
    return nb_func() * nb_comp();
  }
  
  inline size_t iv2ind (size_t i, size_t var ) const 
  {
    assert (i==0);
    assert (var < DIM);
    return var;
  }
  
  void evaluate(const Entity&, const Vec<DIM, DataType> & x, std::vector<DataType>& vals) const
  {
    assert (DIM >= 2);
    assert (this->comps_.size() > 0);
    vals.clear();
    vals.resize (this->nb_comp(), 0.);
    
    for (size_t d=0; d<comps_.size(); ++d)
    {
      int c = comps_[d];
      if (c == 0)
        vals[d] = std::sin(pi * x[0]) * std::sin(pi * x[0]) 
               * std::sin(2. * pi * x[1]);
      if (c == 1)
        vals[d] = - std::sin(2. * pi * x[0]) 
                 * std::sin(pi * x[1]) * std::sin(pi * x[1]);
      if (c == 2 )
        vals[d] = 1.;
       //std::cout << " point " << x[0] << " , " << x[1] << " comp " << c << " -> " << res[d] << std::endl;
    }
  }
  
  DataType evaluate(size_t var, const Vec<DIM, DataType> & x) const
  {
    if (var == 0)
      return std::sin(pi * x[0]) * std::sin(pi * x[0]) * std::sin(2. * pi * x[1]);
    if (var == 1)
      return - std::sin(2. * pi * x[0]) * std::sin(pi * x[1]) * std::sin(pi * x[1]);
    if (var == 2)
      return 1.;
    return 0.;
  }
  
  std::vector<size_t> comps_;
};

template <class DataType, int DIM>
void init_vector (VectorSpace<DataType,DIM> &space, 
                  CoupledVector<DataType>& sol)
{
  // define fe coeff vector
  sol.Init(space.get_mpi_comm(), space.la_couplings());
  sol.InitStructure();
  sol.Zeros();
  sol.Update();
}  

template <class DataType, int DIM>
void interpolate (VectorSpace<DataType,DIM> &space, 
                  bool vector_fe,
                  CoupledVector<DataType>& sol)
{ 
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD,  &rank);
  
  CONSOLE_OUTPUT(rank, "  Global Interpolation with Analytical Function");
  Timer timer;
  timer.start();
   
  if (vector_fe)
  {
    std::vector<size_t> comps(DIM, 0);
    for (size_t d=0; d<DIM; ++d)
    {
      comps[d] = d;
    }
    
    TestFunction<DataType, DIM> test_func;
    test_func.set_components(comps);
    
    FeInterNodal<DataType, DIM, TestFunction<DataType, DIM> > fe_inter (space, &test_func, 0);
    fe_inter.interpolate (sol);
  }
  else
  {
    for (size_t d=0; d<DIM; ++d)
    {
      std::vector<size_t> comps(1, d);
      
      TestFunction<DataType, DIM> test_func;
      test_func.set_components(comps);
      
      FeInterNodal<DataType, DIM, TestFunction<DataType, DIM> > fe_inter (space, &test_func, d);
      fe_inter.interpolate (sol);
    }
  }
  
  sol.Update();
  timer.stop();
  CONSOLE_OUTPUT(rank, "   interpolate -> " << timer.get_duration() << " sec");
}

template <class DataType, int DIM>
void interpolate_mapping_full (VectorSpace<DataType,DIM> &in_space,
                               VectorSpace<DataType,DIM> &out_space, 
                               bool vector_fe,
                               CoupledVector<DataType>& in_sol,
                               CoupledVector<DataType>& out_sol)
{ 
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD,  &rank);
  
  CONSOLE_OUTPUT(rank, "  Full Mapping Interpolation with BasisEvalLocal, #in cells = " 
                  << in_space.meshPtr()->num_entities(DIM) << " , # out cells = " << out_space.meshPtr()->num_entities(DIM));
  Timer timer;
  timer.start();
  
  std::vector<size_t> in_fe_ind;
  std::vector<size_t> out_fe_ind;
  
  if (vector_fe)
  {
    in_fe_ind.resize(1,0);
    out_fe_ind.resize(1,0);
  }
  else
  {
    in_fe_ind.resize(DIM,0);
    out_fe_ind.resize(DIM,0);
    for (size_t d=0; d<DIM; ++d)
    {
      in_fe_ind[d] = d;
      out_fe_ind[d] = d;
    }
  }
  
  FeInterMapFullNodal<LADescriptorCoupledD, DIM> fe_mapping;

  LIKWID_MARKER_START("map");

  fe_mapping.init(&in_space, &out_space, true, in_sol, out_sol, in_fe_ind, out_fe_ind);

  LIKWID_MARKER_STOP("map");

  timer.stop();
  CONSOLE_OUTPUT(rank, "   initialize  -> " << timer.get_duration() << " sec");
  timer.reset();
  timer.start();
  
  fe_mapping.interpolate (in_sol, out_sol);
  
  timer.stop();
  CONSOLE_OUTPUT(rank, "   interpolate -> " << timer.get_duration() << " sec");
}

template <class DataType, int DIM>
void interpolate_mapping_reduced (VectorSpace<DataType,DIM> &in_space,
                                  VectorSpace<DataType,DIM> &out_space, 
                                  bool vector_fe,
                                  CoupledVector<DataType>& in_sol,
                                  CoupledVector<DataType>& out_sol)
{ 
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD,  &rank);
  
  CONSOLE_OUTPUT(rank, "  Reduced Mapping Interpolation with FeEvalLocal");
  Timer timer;
  timer.start();
  
  std::vector<size_t> in_fe_ind;
  std::vector<size_t> out_fe_ind;
  
  if (vector_fe)
  {
    in_fe_ind.resize(1,0);
    out_fe_ind.resize(1,0);
  }
  else
  {
    in_fe_ind.resize(DIM,0);
    out_fe_ind.resize(DIM,0);
    for (size_t d=0; d<DIM; ++d)
    {
      in_fe_ind[d] = d;
      out_fe_ind[d] = d;
    }
  }
  
  FeInterMapRedNodal<LADescriptorCoupledD, DIM> fe_mapping;
  fe_mapping.init(&in_space, &out_space, in_fe_ind, out_fe_ind);
  
  timer.stop();
  CONSOLE_OUTPUT(rank, "   initialize  -> " << timer.get_duration() << " sec");
  timer.reset();
  timer.start();
  
  fe_mapping.interpolate (in_sol, out_sol);
  
  timer.stop();
  CONSOLE_OUTPUT(rank, "   interpolate -> " << timer.get_duration() << " sec");
}

template <class DataType, int DIM>
void interpolate_global (VectorSpace<DataType,DIM> &in_space,
                          VectorSpace<DataType,DIM> &out_space, 
                          bool vector_fe,
                          CoupledVector<DataType>& in_sol,
                          CoupledVector<DataType>& out_sol)
{ 
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD,  &rank);
  
  CONSOLE_OUTPUT(rank, "  Global  Interpolation with FeEvalLocal");
  Timer timer;
  timer.start();
  
  std::vector<size_t> in_fe_ind;
  std::vector<size_t> out_fe_ind;
  
  if (vector_fe)
  {
    in_fe_ind.resize(1,0);
    out_fe_ind.resize(1,0);
  }
  else
  {
    in_fe_ind.resize(DIM,0);
    out_fe_ind.resize(DIM,0);
    for (size_t d=0; d<DIM; ++d)
    {
      in_fe_ind[d] = d;
      out_fe_ind[d] = d;
    }
  }
  
  for (size_t l=0; l<out_fe_ind.size(); ++l)
  {
    FeEvalLocal<DataType, DIM> fe_eval(in_space, in_sol, in_fe_ind[l]);
    FeInterNodal<DataType, DIM, FeEvalLocal<DataType, DIM> > fe_inter (out_space, &fe_eval, out_fe_ind[l]);
    
    fe_inter.interpolate (out_sol); 
  }
  timer.stop();
  CONSOLE_OUTPUT(rank, "   interpolate -> " << timer.get_duration() << " sec");
}

template <class DataType, int DIM>
void interpolate_global_basis (VectorSpace<DataType,DIM> &in_space,
                               VectorSpace<DataType,DIM> &out_space, 
                               bool vector_fe,
                               CoupledVector<DataType>& in_sol,
                               CoupledVector<DataType>& out_sol)
{
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD,  &rank);
  
  CONSOLE_OUTPUT(rank, "  Global  Interpolation with FeEvalBasisLocal");
  Timer timer;
  timer.start();
  
  std::vector<size_t> in_fe_ind;
  std::vector<size_t> out_fe_ind;
  
  if (vector_fe)
  {
    in_fe_ind.resize(1,0);
    out_fe_ind.resize(1,0);
  }
  else
  {
    in_fe_ind.resize(DIM,0);
    out_fe_ind.resize(DIM,0);
    for (size_t d=0; d<DIM; ++d)
    {
      in_fe_ind[d] = d;
      out_fe_ind[d] = d;
    }
  }
  
  for (size_t l=0; l<out_fe_ind.size(); ++l)
  {
    FeEvalBasisLocal<DataType, DIM> fe_eval(in_space, in_sol, in_fe_ind[l]);
    FeInterNodal<DataType, DIM, FeEvalBasisLocal<DataType, DIM> > fe_inter (out_space, &fe_eval, out_fe_ind[l]);
    
    fe_inter.interpolate (out_sol); 
  }
  timer.stop();
  CONSOLE_OUTPUT(rank, "   interpolate -> " << timer.get_duration() << " sec");
}

template <class DataType, int DIM>
void check_point_values (VectorSpace<DataType,DIM> &space, 
                         MeshPtr mesh_ptr,
                         bool vector_fe,
                         RefCellType c_type,
                         CoupledVector<DataType>& sol)
{
  TestFunction<DataType, DIM> func;
  FeEvalCell<DataType, DIM> fe_eval(space, sol);
  
  if (vector_fe)
  {
    std::vector<size_t> comps(DIM, 0);
    for (size_t d=0; d<DIM; ++d)
    {
      comps[d] = d;
    }
    func.set_components(comps);
  }
  else
  {
    std::vector<size_t> comps(1, 0);
    func.set_components(comps);
  }
  
  // loop over mesh interfaces
  for (mesh::EntityIterator it = mesh_ptr->begin(DIM), e_it = mesh_ptr->end(DIM); it != e_it; ++it) 
  {
    Entity cell = mesh_ptr->get_entity(DIM, it->index());
    
    std::vector<DataType> cell_coords;          
    cell.get_coordinates(cell_coords);
    size_t num_vert = cell.num_vertices();
    
    std::vector< Vec<DIM, DataType> > test_pts(num_vert);
    for (size_t l=0; l<num_vert; ++l)
    {
      for (size_t d=0; d<DIM; ++d)
      {
        test_pts[l][d] = cell_coords[l*DIM+d];
      }
    }

    for (size_t l=0; l<num_vert; ++l)
    {
      Vec<DIM, DataType> pt = test_pts[l];
      std::vector<DataType> fe_vals;
      
      fe_eval.evaluate(cell, pt, fe_vals);
      assert (fe_vals.size() == DIM);
      for (size_t v=0; v<DIM; ++v)
      {    
        DataType exact_val = func.evaluate(v, pt);
        //std::cout << fe_val << " <> " << exact_val << std::endl;
        TEST_EQUAL_EPS(fe_vals[v], exact_val, 1e-10);
      }
    }
  }
}

template <class DataType, int DIM>
void check_dof_values (VectorSpace<DataType,DIM> &space, 
                       MeshPtr mesh_ptr,
                       bool vector_fe,
                       CoupledVector<DataType>& sol)
{
  TestFunction<DataType, DIM> func;
  if (vector_fe)
  {
    std::vector<size_t> comps(DIM, 0);
    for (size_t d=0; d<DIM; ++d)
    {
      comps[d] = d;
    }
    func.set_components(comps);
  }
  else
  {
    std::vector<size_t> comps(1, 0);
    func.set_components(comps);
  }
  
  FeEvalCell<DataType, DIM> fe_eval(space, sol);
  int rank = space.rank();
  
  // loop over mesh interfaces
  for (mesh::EntityIterator it = mesh_ptr->begin(DIM), e_it = mesh_ptr->end(DIM); it != e_it; ++it) 
  {
    int temp1, temp2;
    mesh_ptr->get_attribute_value("_remote_index_", DIM, it->index(), &temp1);
    mesh_ptr->get_attribute_value("_sub_domain_", DIM, it->index(), &temp2);
        
    //std::cout << std::endl << " CELL " << it->index() << std::endl;
    Element<DataType,DIM> elem (space, it->index());

    Entity cell =elem.get_cell();

    MappingPhys2Ref<DataType, DIM, FeEvalCell<DataType, DIM> > elem_on_ref ( &fe_eval, 
                                                                             &cell, 
                                                                             elem.get_fe(0)->fe_trafo(),  
                                                                             elem.get_cell_transformation());
    
    MappingPhys2Ref<DataType, DIM, TestFunction<DataType, DIM> > func_on_ref ( &func, 
                                                                               &cell, 
                                                                               elem.get_fe(0)->fe_trafo(),  
                                                                               elem.get_cell_transformation());
                                                                          
    // evaluate dofs on current cell
    size_t num_dofs = elem.get_fe(0)->nb_dof_on_cell();
    
    std::vector<cDofId> all_dofs(num_dofs);
    for (size_t l=0; l<num_dofs; ++l)
    {
      all_dofs[l] = l;
    }
  
    std::vector< std::vector<DataType> > dof_fe;
    std::vector< std::vector<DataType> > dof_func;
    
    //std::cout << std::endl << std::endl;
    //std::cout << "------ eval dofs for mapped elem " << std::endl;
    elem.get_fe(0)->dof_container()->evaluate (&elem_on_ref, all_dofs, dof_fe);
    //std::cout << std::endl << std::endl;
    //std::cout << "------ eval dofs for mapped func " << std::endl;
    elem.get_fe(0)->dof_container()->evaluate (&func_on_ref, all_dofs, dof_func);
    /*
    std::cout << std::endl << std::endl;
    std::cout << std::endl << std::endl;
    
    std::cout << " =============== " << std::endl;
    std::cout << " DOF values " << std::endl;
    std::cout << string_from_range(dof_values.begin(), dof_values.end()) << std::endl;
    std::cout << " DOF FE " << dof_fe.size() << " x " << dof_fe[0].size() << std::endl;
    print (dof_fe);
    std::cout << " DOF Func " << dof_func.size() << " x " << dof_func[0].size() << std::endl;
    print (dof_func);
    */
    
    for (size_t l=0; l<dof_fe.size(); ++l)
    {
      for (size_t i=0; i<dof_fe[l].size(); ++i)
      {
        BOOST_TEST(std::abs(dof_fe[l][i] - dof_func[l][i])< 1e-8);
      }
    }
  }
}

template <class DataType, int DIM>
void perform_test (std::vector< MeshPtr > & mesh,
                   VectorSpace<DataType, DIM>& space,
                   VectorSpace<DataType, DIM>& space_c,
                   std::vector< FEType >& fe_ansatz,
                   std::vector< bool > & is_cg,
                   std::vector< int > & degrees,
                   std::string fe_name,
                   std::string cell_name,
                   int level,
                   int init_level)
{
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD,  &rank);
  
  space.Init(*mesh[level], fe_ansatz, is_cg, degrees);
  const int tdim = mesh[0]->tdim();
  int num_level = 0;
  num_level = mesh.size();
  
  const int nb_dof = space.dof().nb_dofs_global();
  CONSOLE_OUTPUT(rank, " # dofs " << nb_dof);
  
  bool vector_fe = false;
  if (space.nb_var() != space.nb_fe())
  {
    vector_fe = true;
  }
  
  // interpolate function
  CoupledVector<double> sol;
  init_vector<DataType, DIM> (space, sol);
  interpolate<DataType, DIM> (space, vector_fe, sol);
          
#ifdef VISUALIZE
  std::vector<size_t> u_vars(DIM, 0);
  std::vector< std::string> u_names;
  std::vector< std::string> u_names_c1;
  std::vector< std::string> u_names_c2;
  std::vector< std::string> u_names_c3;
  std::vector< std::string> u_names_c4;
  
  for (size_t d=0; d<DIM; ++d)
  {
    u_vars[d] = d;
    u_names.push_back("u_" + std::to_string(d));
    u_names_c1.push_back("u_c1_" + std::to_string(d));
    u_names_c2.push_back("u_c2_" + std::to_string(d));
    u_names_c3.push_back("u_c3_" + std::to_string(d));
    u_names_c4.push_back("u_c4_" + std::to_string(d));
  }
  
  std::string filename = "FeInterpolEvalStd_" + cell_name + "_lvl" + std::to_string(init_level + level) 
                        + "_" + fe_name  
                        + "_" + std::to_string(degrees[0]) + ".vtu";
  
  std::vector< std::vector< DataType> > cell_index (num_level);

  for (int i=0; i<num_level; ++i)
  {     
    cell_index[i].resize(mesh[i]->num_entities(tdim), 0);
    for (int c=0; c<mesh[i]->num_entities(tdim); ++c)
    {
      cell_index[i][c] = static_cast<double>(c);
    }
  }
    
  CellVisualization< DataType, DIM> visu(space, 1);
  visu.visualize(sol, u_vars, u_names);
  visu.visualize_cell_data(cell_index[level], "_cell_index_");
  VTKWriter< DataType, DIM> vtk_writer (visu, MPI_COMM_WORLD, 0);
  vtk_writer.write(filename);
#endif
          
  // check  values of dofs
  check_dof_values<DataType, DIM> (space, mesh[level], vector_fe, sol);
          
#ifdef TEST_MAP
  // check fe interpolation mapping
  if (level >= 1)
  {
    int coarse_lvl = level-1;
    space_c.Init(*mesh[coarse_lvl], fe_ansatz, is_cg, degrees);
    CoupledVector<DataType> sol_c_1;
    CoupledVector<DataType> sol_c_2;
    CoupledVector<DataType> sol_c_3;
    CoupledVector<DataType> sol_c_4;
    CoupledVector<DataType> sol_diff_12;
    CoupledVector<DataType> sol_diff_13;
    CoupledVector<DataType> sol_diff_23;
    CoupledVector<DataType> sol_diff_34;
            
    init_vector<DataType, DIM> (space_c, sol_c_1);
    init_vector<DataType, DIM> (space_c, sol_c_2);
    init_vector<DataType, DIM> (space_c, sol_c_3);
    init_vector<DataType, DIM> (space_c, sol_c_4);
            
    // map fine function to coarse function by std interpolator
#ifdef TEST_GLOBAL
    interpolate_global<DataType, DIM> (space, space_c, vector_fe, sol, sol_c_1);
            
    if (coarse_lvl == level)
    {
      check_dof_values<DataType, DIM> (space_c, mesh[coarse_lvl], vector_fe, sol_c_1);
    }
#endif
#ifdef TEST_GLOBAL_BASIS
    // map fine function to coarse function by std interpolator and BasisEvaluator
    interpolate_global_basis<DataType, DIM> (space, space_c, vector_fe, sol, sol_c_2);
            
    if (coarse_lvl == level)
    {
      check_dof_values<DataType, DIM> (space_c, mesh[coarse_lvl], vector_fe, sol_c_2);
    }
#endif
#ifdef TEST_MAPPING_FULL       
    // map fine function to coarse function by full mapping
    interpolate_mapping_full<DataType, DIM> (space, space_c, vector_fe, sol, sol_c_3);
#endif
#ifdef TEST_MAPPING_REDUCED
    // map fine function to coarse function by reduced mapping
    interpolate_mapping_reduced<DataType, DIM> (space, space_c, vector_fe, sol, sol_c_4);
#endif
#ifdef VISUALIZE
    std::string filename_c = "FeInterpolEvalMap_" + cell_name + "_lvl" 
                            + std::to_string(init_level + coarse_lvl) + "_" + fe_name 
                            + std::to_string(degrees[0]) + ".vtu";
                                 
    CellVisualization< DataType, DIM> visu(space_c, 1);
    visu.visualize(sol_c_1, u_vars, u_names_c1);
    visu.visualize(sol_c_2, u_vars, u_names_c2);
    visu.visualize(sol_c_3, u_vars, u_names_c3);
    visu.visualize(sol_c_4, u_vars, u_names_c4);
    visu.visualize_cell_data(cell_index[coarse_lvl], "_cell_index_");
    VTKWriter< DataType, DIM> vtk_writer (visu, MPI_COMM_WORLD, 0);
    vtk_writer.write(filename_c);
#endif

    if (coarse_lvl == level)
    {
      check_dof_values<DataType, DIM> (space_c, mesh[coarse_lvl], vector_fe, sol_c_2);
    }
            
#ifdef TEST_GLOBAL
#ifdef TEST_GLOBAL_BASIS
    sol_diff_12.CloneFrom(sol_c_2);
    sol_diff_12.Axpy(sol_c_1, -1);
    sol_diff_12.Update();
    DataType diff_12 = sol_diff_12.Norm2();
    BOOST_TEST(std::abs(diff_12 - 0.)< 1e-8);
#endif
#endif

#ifdef TEST_GLOBAL
#ifdef TEST_MAPPING_FULL
    sol_diff_13.CloneFrom(sol_c_3);
    sol_diff_13.Axpy(sol_c_1, -1);
    sol_diff_13.Update();
    DataType diff_13 = sol_diff_13.Norm2();
    BOOST_TEST(std::abs(diff_13 - 0.)< 1e-8);
#endif
#endif

#ifdef TEST_GLOBAL_BASIS
#ifdef TEST_MAPPING_FULL
    sol_diff_23.CloneFrom(sol_c_3);
    sol_diff_23.Axpy(sol_c_2, -1);
    sol_diff_23.Update();
    DataType diff_23 = sol_diff_23.Norm2();
    BOOST_TEST(std::abs(diff_23 - 0.)< 1e-8);
#endif
#endif

#ifdef TEST_MAPPING_FULL
#ifdef TEST_MAPPING_REDUCED
    sol_diff_34.CloneFrom(sol_c_3);
    sol_diff_34.Axpy(sol_c_4, -1);
    sol_diff_34.Update();
    DataType diff_34 = sol_diff_34.Norm2();
    BOOST_TEST(std::abs(diff_34 - 0.)< 1e-8);
#endif
#endif
    //std::cout << diff_12 << " " << diff_13 << " " << diff_23 << std::endl;
    
    
    
    
  }
#endif
}
          

static const char *datadir = MESH_DATADIR;

BOOST_AUTO_TEST_CASE(fe_interpol_eval) {

  int argc = boost::unit_test::framework::master_test_suite().argc;
  char** argv = boost::unit_test::framework::master_test_suite().argv;
  
  int init_level =0;
  int rank;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD,  &rank);

#ifdef USE_GPERF
    time_t rawtime;
    struct tm * timeinfo;
    char buffer[80];

    time (&rawtime);
    timeinfo = localtime(&rawtime);

    strftime(buffer,sizeof(buffer),"%d-%m-%Y %H:%M:%S",timeinfo);
    std::string str_time(buffer);
  
    std::string prof_name = "InterpolEval_" + str_time + ".log";
    ProfilerStart(prof_name.c_str());
#endif


  LIKWID_MARKER_INIT;


  //LogKeeper::get_log ( "info" ).set_target  ( &( std::cout ) );    
  LogKeeper::get_log ( "debug" ).set_target ( &( std::cout ) );
  LogKeeper::get_log ( "error" ).set_target ( &( std::cout ) );
  LogKeeper::get_log ( "info" ).set_target ( &( std::cout ) );
    
  // Which files should be checked?

  std::vector< std::string > filenames;
  std::vector< TDim > tdims;
  std::vector< GDim > gdims;
  std::vector< RefCellType > cell_types;

#ifdef TEST_QUAD  
  filenames.push_back(std::string(datadir) + std::string("two_quads_2d.inp"));
  tdims.push_back(2);
  gdims.push_back(2);
  cell_types.push_back(RefCellType::QUAD_STD);
#endif
#ifdef TEST_TRI
  filenames.push_back(std::string(datadir) + std::string("two_triangles_2d.inp"));
  tdims.push_back(2);
  gdims.push_back(2);
  cell_types.push_back(RefCellType::TRI_STD);
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
  VectorSpace< double, 2 > lag_space2,lag_space2_c; 
  VectorSpace< double, 3 > lag_space3, lag_space3_c;
  VectorSpace< double, 2 > rt_space2, rt_space2_c;
  VectorSpace< double, 3 > rt_space3, rt_space3_c; 
  VectorSpace< double, 2 > bdm_space2, bdm_space2_c;
  VectorSpace< double, 3 > bdm_space3, bdm_space3_c; 

  std::vector< FEType > lag_fe_ansatz2 (2, FEType::LAGRANGE);
  std::vector< FEType > lag_fe_ansatz3 (3, FEType::LAGRANGE);
  std::vector< FEType > rt_fe_ansatz (1, FEType::RT);
  std::vector< FEType > bdm_fe_ansatz (1, FEType::BDM);
  std::vector< bool > is_cg(1, true);
  std::vector< bool > is_cg2(2, true);
  std::vector< bool > is_cg3(3, true);
  
  std::vector< std::vector< int > > lag_degrees2(MAX_DEG_LAG);
  for (int l=0; l < lag_degrees2.size(); ++l)
  {
    lag_degrees2[l].resize(2,l+1);
  }
  
  std::vector< std::vector< int > > lag_degrees3(MAX_DEG_LAG);
  for (int l=0; l < lag_degrees3.size(); ++l)
  {
    lag_degrees3[l].resize(3,l+1);
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

    std::string cell_prefix;
    if (cell_types[test_number] == RefCellType::TRI_STD)
    {
      cell_prefix = "Triangle";
    }
    else if (cell_types[test_number] == RefCellType::TET_STD)
    {
      cell_prefix = "Tetra";
    }
    else if (cell_types[test_number] == RefCellType::QUAD_STD)
    {
      cell_prefix = "Quad";
    }
    else if (cell_types[test_number] == RefCellType::HEX_STD)
    {
      cell_prefix = "Hexa";
    }
    /////////////////////////////////////
    // mesh

    std::vector<MeshPtr> master_mesh(NUM_LEVEL);
    if (rank == 0) 
    {
      master_mesh[0] = read_mesh_from_file(filename.c_str(), gdim, gdim, 0);
      int cur_level = 0;
      while (cur_level < init_level)
      {
        master_mesh[0] = master_mesh[0]->refine();
        cur_level++;
      }
      for (int i=1; i<NUM_LEVEL; ++i)
      {
        master_mesh[i] = master_mesh[i-1]->refine();
      }
    }
    
    std::vector< MeshPtr > mesh(NUM_LEVEL);
    for (int i=0; i<NUM_LEVEL; ++i)
    {
      int num_ref_seq_steps;
      MeshPtr l_mesh = partition_and_distribute(master_mesh[i], 0, MPI_COMM_WORLD, num_ref_seq_steps);
      assert(l_mesh != 0);
  
      SharedVertexTable shared_verts;
      mesh[i] = compute_ghost_cells(*l_mesh, MPI_COMM_WORLD, shared_verts);
    }

    for (int i = 0; i < NUM_LEVEL; ++i) 
    {
      // tests
      CONSOLE_OUTPUT(rank, "test" << filename << " on mesh level: " << i);
#ifdef TEST_BDM
      // BDM element
      if (gdim == 2 && cell_types[test_number] == RefCellType::TRI_STD)
      {
        for (int l=0; l<bdm_degrees.size(); ++l)
        {
          CONSOLE_OUTPUT(rank, "test 2D BDM space of degree " << bdm_degrees[l][0]);
          
          perform_test<double, 2> (mesh,
                                   bdm_space2, bdm_space2_c,
                                   bdm_fe_ansatz, is_cg, bdm_degrees[l],
                                   "BDM", cell_prefix, i, init_level);
          CONSOLE_OUTPUT(rank, " ");
        }
      }
#endif
#ifdef TEST_LAG
      // Lagrange element
      if (gdim == 2)
      {
        for (int l=0; l<lag_degrees2.size(); ++l)
        {
          // setup space                   
          CONSOLE_OUTPUT(rank, "test 2D LAG space of degree " << lag_degrees2[l][0]);
          perform_test<double, 2> (mesh,
                                   lag_space2, lag_space2_c,
                                   lag_fe_ansatz2, is_cg2, lag_degrees2[l],
                                   "Lag", cell_prefix, i, init_level);
          CONSOLE_OUTPUT(rank, " ");
        }
      }
      if (gdim == 3)
      {
        for (int l=0; l<lag_degrees3.size(); ++l)
        {
          // setup space                   
          CONSOLE_OUTPUT(rank, "test 3D LAG space of degree " << lag_degrees3[l][0]);
          perform_test<double, 3> (mesh,
                                   lag_space3, lag_space3_c,
                                   lag_fe_ansatz3, is_cg3, lag_degrees3[l],
                                   "Lag", cell_prefix, i, init_level);
          CONSOLE_OUTPUT(rank, " ");
        }
      }
#endif
#ifdef TEST_RT
      // RT element
      if (gdim == 2 && cell_types[test_number] == RefCellType::TRI_STD)
      {
        for (int l=0; l<rt_degrees.size(); ++l)
        {
          CONSOLE_OUTPUT(rank, "test 2D RT  space of degree " << rt_degrees[l][0]);

          perform_test<double, 2> (mesh,
                                   rt_space2, rt_space2_c,
                                   rt_fe_ansatz, is_cg, rt_degrees[l],
                                   "RT", cell_prefix, i, init_level);
          CONSOLE_OUTPUT(rank, " ");
        }
      }
      // RT element on quad
      if (gdim == 2 && cell_types[test_number] == RefCellType::QUAD_STD)
      {
        for (int l=0; l<rt_degrees.size(); ++l)
        {
          CONSOLE_OUTPUT(rank, "test 2D RT  space of degree " << rt_degrees[l][0]);

          perform_test<double, 2> (mesh,
                                   rt_space2, rt_space2_c,
                                   rt_fe_ansatz, is_cg, rt_degrees[l],
                                   "RT", cell_prefix, i, init_level);
          CONSOLE_OUTPUT(rank, " ");
        }
      }
      // RT element on tet
      if (gdim == 3 && cell_types[test_number] == RefCellType::TET_STD)
      {
        for (int l=0; l<rt_degrees.size(); ++l)
        {
          CONSOLE_OUTPUT(rank, "test 3D RT  space of degree " << rt_degrees[l][0]);

          perform_test<double, 3> (mesh,
                                   rt_space3, rt_space3_c,
                                   rt_fe_ansatz, is_cg, rt_degrees[l],
                                   "RT", cell_prefix, i, init_level);
          CONSOLE_OUTPUT(rank, " ");
        }
      }
      // RT element on hex
      if (gdim == 3 && cell_types[test_number] == RefCellType::HEX_STD)
      {
        for (int l=0; l<rt_degrees.size(); ++l)
        {
          CONSOLE_OUTPUT(rank, "test 3D RT  space of degree " << rt_degrees[l][0]);

          perform_test<double, 3> (mesh,
                                   rt_space3, rt_space3_c,
                                   rt_fe_ansatz, is_cg, rt_degrees[l],
                                   "RT", cell_prefix, i, init_level);
          CONSOLE_OUTPUT(rank, " ");
        }
      }
#endif
      CONSOLE_OUTPUT(rank, "===============================");
    }
  }

  LIKWID_MARKER_CLOSE;

#ifdef USE_GPERF
    ProfilerStop();
#endif

  MPI_Finalize();

  return;
}
