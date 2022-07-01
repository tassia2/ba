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

using namespace std;
using namespace hiflow;
using namespace hiflow::mesh;
using namespace hiflow::la;

/// 
///
/// \brief 
///

#define VISUALIZE

const int DEBUG_OUT = 1;
const double pi = 3.14159265;

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
  
  Vec<DIM,DataType> evaluate_grad(size_t var, const Vec<DIM, DataType> & x) const
  {
    Vec<DIM,DataType> grad;
    if (var == 0)
    {
      grad[0] = 2. * pi * std::sin(pi * x[0]) * std::cos(pi * x[0]) * std::sin(2. * pi * x[1]);
      grad[1] = 2. * pi * std::sin(pi * x[0]) * std::sin(pi * x[0]) * std::cos(2. * pi * x[1]);
    }  
    if (var == 1)
    {
      grad[0] = - 2. * pi * std::cos(2. * pi * x[0]) * std::sin(pi * x[1]) * std::sin(pi * x[1]);
      grad[1] = - 2. * pi * std::sin(2. * pi * x[0]) * std::sin(pi * x[1]) * std::cos(pi * x[1]);
    }
    return grad;
  }
  
  std::vector<size_t> comps_;
};

template <class DataType, int DIM>
void interpolate (VectorSpace<DataType,DIM> &space, 
                  MeshPtr mesh_ptr,
                  bool vector_fe,
                  CoupledVector<DataType>& sol)
{
  // define fe coeff vector
  sol.Init(MPI_COMM_WORLD, space.la_couplings());
  sol.InitStructure();
  sol.Zeros();
  
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
}

template <class DataType, int DIM>
std::vector<DataType> compute_l2_error (VectorSpace<DataType,DIM> &space, 
                                        MeshPtr mesh_ptr,
                                        bool vector_fe,
                                        RefCellType c_type,
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
  
  std::vector<DataType> sum (3,0.);
  int number = 0;
  
  FeEvalCell<DataType, DIM> fe_eval(space, sol);
  
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
      number++;
      DataType err = 0.;
      DataType err_grad = 0.;
      DataType err_div = 0.;
      Vec<DIM, DataType> pt = test_pts[l];
      
      std::vector< DataType> fe_vals;
      std::vector< Vec<DIM, DataType> > fe_grads;
      
      fe_eval.evaluate(cell, pt, fe_vals);
      fe_eval.evaluate_grad(cell, pt, fe_grads);
      
      assert (fe_vals.size() == DIM);
      assert (fe_grads.size() == DIM);
      
      for (size_t v=0; v<DIM; ++v)
      {
        DataType exact_val = func.evaluate(v, pt);
        Vec<DIM, DataType> exact_grad = func.evaluate_grad(v, pt);
        
        err += (fe_vals[v] - exact_val) * (fe_vals[v] - exact_val);
        err_div += (fe_grads[v][v] - exact_grad[v]) * (fe_grads[v][v] - exact_grad[v]);
        
        for (size_t i=0; i<DIM; ++i)
        {
          err_grad += (fe_grads[v][i] - exact_grad[i]) * (fe_grads[v][i] - exact_grad[i]);
        }
      }
      
      sum[0] += err;
      sum[1] += err_grad;
      sum[2] += err_div;
    }
  }
  for (int l=0; l<3; ++l)
  {
    sum[l] = std::sqrt( sum[l] / number ); 
  }
  return sum;
}


template <class DataType, int DIM>
class L2ErrorIntegrator : private AssemblyAssistant< DIM, DataType > 
{
public:
  L2ErrorIntegrator(CoupledVector< DataType > &pp_sol, DataType c_l2, DataType c_h1, DataType c_div)
      : pp_sol_(pp_sol), c_l2_(c_l2), c_h1_(c_h1), c_div_(c_div) {}

  void operator()(const Element< DataType, DIM > &element,
                  const Quadrature< DataType > &quadrature, 
                  DataType &value) 
  {
    AssemblyAssistant< DIM, DataType >::initialize_for_element(element, quadrature, false);

    const int num_q = this->num_quadrature_points();
    for (int v = 0; v < DIM; ++v) 
    {
      approx_sol_[v].clear();
      this->evaluate_fe_function(pp_sol_, v, approx_sol_[v]);
    
      grad_approx_sol_[v].clear();
      this->evaluate_fe_function_gradients(pp_sol_, v, grad_approx_sol_[v]);
    }
    
    for (int q = 0; q < num_q; ++q) 
    {
      const DataType wq = this->w(q);
      const Vec<DIM, DataType> xq = this->x(q);
      const DataType dJ = std::abs(this->detJ(q));
      
      Vec<DIM, DataType> exact;
      std::vector< Vec<DIM, DataType> > exact_grad(DIM);
      DataType exact_div = 0.;
      DataType approx_div = 0.;
        
      for (size_t d=0; d<DIM; ++d)
      {
        exact[d] = func_.evaluate(d, this->x(q));
        exact_grad[d] = func_.evaluate_grad(d, this->x(q));
        exact_div += exact_grad[d][d];
        approx_div += this->grad_approx_sol_[d][q][d];
      }
      
      DataType err = 0.;
      DataType err_grad = 0.;
      DataType err_div = 0.;
      
      err_div += (exact_div - approx_div) * (exact_div - approx_div);
      for (size_t d=0; d<DIM; ++d)
      {
        err += (exact[d] - this->approx_sol_[d][q]) * (exact[d] - this->approx_sol_[d][q]);
        for (size_t l=0; l<DIM; ++l)  
        {
          err_grad += (exact_grad[d][l] - grad_approx_sol_[d][q][l]) 
                    * (exact_grad[d][l] - grad_approx_sol_[d][q][l]);
        }
      }
      
      value += wq *  (c_l2_ * err + c_h1_ * err_grad + c_div_ * err_div) * dJ;
    }
  }

private:
  // coefficients of the computed solution
  CoupledVector< DataType > &pp_sol_;
  // vector with values of computed solution evaluated at each quadrature point
  FunctionValues< DataType > approx_sol_[DIM];
  FunctionValues< Vec<DIM,DataType> > grad_approx_sol_[DIM];
  // variables for which to compute the norm
  TestFunction<DataType, DIM> func_;
  
  DataType c_l2_, c_h1_, c_div_;
};

        
static const char *datadir = MESH_DATADIR;

int main(int argc, char *argv[]) {
  int init_level = 1;
  int num_level;
  int num_level2 = 4;
  int num_level3 = 3;
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
 
  VectorSpace< double, 2 > lag_space2;
  VectorSpace< double, 3 > lag_space3;
  VectorSpace< double, 2 > rt_space2;
  VectorSpace< double, 3 > rt_space3; 
  VectorSpace< double, 2 > bdm_space2;
  VectorSpace< double, 3 > bdm_space3; 

  std::vector< FEType > lag_fe_ansatz2 (2, FEType::LAGRANGE);
  std::vector< FEType > lag_fe_ansatz3 (3, FEType::LAGRANGE);
  std::vector< FEType > rt_fe_ansatz (1, FEType::RT);
  std::vector< FEType > bdm_fe_ansatz (1, FEType::BDM);
  std::vector< bool > is_cg(1, true);
  std::vector< bool > is_cg2(2, true);
  std::vector< bool > is_cg3(3, true);

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
      
  for (int test_number = 0; test_number < filenames.size(); ++test_number) 
  {
    std::string filename = filenames.at(test_number);
    TDim tdim = tdims.at(test_number);
    GDim gdim = gdims.at(test_number);

    if (gdim == 3)
    {
      num_level = num_level3;
    }
    if (gdim == 2)
    {
      num_level = num_level2;
    }
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

    // tests
    CONSOLE_OUTPUT(rank, "===============");
    CONSOLE_OUTPUT(rank, "Testing " << filename);
        
#if 1
    // BDM element
    if (gdim == 2 && cell_types[test_number] == RefCellType::TRI_STD)
    {
      for (int l=0; l<bdm_degrees.size(); ++l)
      {
        CONSOLE_OUTPUT(rank, "------------------------ ");
        CONSOLE_OUTPUT(rank, "test BDM space of degree " << bdm_degrees[l][0]);
          
        for (int i = 0; i < num_level; ++i) 
        {
          bdm_space2.Init(*mesh[i], bdm_fe_ansatz, is_cg, bdm_degrees[l]);
          CoupledVector<double> sol;
          interpolate<double, 2> (bdm_space2, mesh[i], true, sol);
          
          StandardGlobalAssembler< double, 2 > global_asm;
          L2ErrorIntegrator<double, 2> local_asm_l2 (sol, 1., 0., 0.);
          L2ErrorIntegrator<double, 2> local_asm_h1 (sol, 0., 1., 0.);
          L2ErrorIntegrator<double, 2> local_asm_hd (sol, 0., 0., 1.);
          
          double L2_error = 0.;
          global_asm.integrate_scalar(bdm_space2, boost::ref(local_asm_l2), L2_error );
          double H1_error = 0.;
          global_asm.integrate_scalar(bdm_space2, boost::ref(local_asm_h1), H1_error );          
          double Hdiv_error = 0.;
          global_asm.integrate_scalar(bdm_space2, boost::ref(local_asm_hd), Hdiv_error );
          
          double global_L2_error = 0.;
          double global_H1_error = 0.;
          double global_Hdiv_error = 0.;
          
          MPI_Reduce(&L2_error, &global_L2_error, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
          MPI_Reduce(&H1_error, &global_H1_error, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
          MPI_Reduce(&Hdiv_error, &global_Hdiv_error, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

/*           
          std::vector<double> errors = compute_l2_error<double, 2> (bdm_space2, mesh[i], true, cell_types[test_number], sol);
    
          std::cout << std::scientific << std::setprecision(2);
          std::cout << rank << " : " << " l2 error = " << std::setw(7) << errors[0]
                    << " h1 error = " << std::setw(7) << errors[1]
                    << " hd error = " << std::setw(7) << errors[2] << std::endl;
*/       
          std::cout << std::scientific << std::setprecision(2);
          CONSOLE_OUTPUT(rank, "RefLevel [" << i << "]" 
                    << " L2 error = " << std::setw(7) << std::sqrt(global_L2_error)
                    << " H1 error = " << std::setw(7) << std::sqrt(global_H1_error)
                    << " Hdiv error = " << std::setw(7) << std::sqrt(global_Hdiv_error));
/*
                    << " l2 error = " << std::setw(7) << errors[0]
                    << " h1 error = " << std::setw(7) << errors[1]
                    << " hd error = " << std::setw(7) << errors[2]);*/
        }
      }
    }
#endif
#if 1
    // Lagrange element
    if (gdim == 2)
    {
      for (int l=0; l<lag_degrees2.size(); ++l)
      {
        CONSOLE_OUTPUT(rank, "------------------------ ");
        CONSOLE_OUTPUT(rank, "test Lag space of degree " << lag_degrees2[l][0]);
        for (int i = 0; i < num_level; ++i) 
        {
          lag_space2.Init(*mesh[i], lag_fe_ansatz2, is_cg2, lag_degrees2[l]);
          CoupledVector<double> sol;
          interpolate<double, 2> (lag_space2, mesh[i], false, sol);
          
          StandardGlobalAssembler< double, 2 > global_asm;
          L2ErrorIntegrator<double, 2> local_asm_l2 (sol, 1., 0., 0.);
          L2ErrorIntegrator<double, 2> local_asm_h1 (sol, 0., 1., 0.);
          L2ErrorIntegrator<double, 2> local_asm_hd (sol, 0., 0., 1.);
          
          double L2_error = 0.;
          global_asm.integrate_scalar(lag_space2, boost::ref(local_asm_l2), L2_error );
          double H1_error = 0.;
          global_asm.integrate_scalar(lag_space2, boost::ref(local_asm_h1), H1_error );          
          double Hdiv_error = 0.;
          global_asm.integrate_scalar(lag_space2, boost::ref(local_asm_hd), Hdiv_error );
          
          double global_L2_error = 0.;
          double global_H1_error = 0.;
          double global_Hdiv_error = 0.;
          
          MPI_Reduce(&L2_error, &global_L2_error, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
          MPI_Reduce(&H1_error, &global_H1_error, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
          MPI_Reduce(&Hdiv_error, &global_Hdiv_error, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
          MPI_Barrier(MPI_COMM_WORLD);
          //std::vector<double> errors = compute_l2_error<double, 2> (bdm_space2, mesh[i], true, cell_types[test_number], sol);
    
          std::cout << std::scientific << std::setprecision(2);
          CONSOLE_OUTPUT(rank, "RefLevel [" << i << "]" 
                    << " L2 error = " << std::setw(7) << std::sqrt(global_L2_error)
                    << " H1 error = " << std::setw(7) << std::sqrt(global_H1_error)
                    << " Hdiv error = " << std::setw(7) << std::sqrt(global_Hdiv_error));
          /*
                    << " l2 error = " << std::setw(7) << errors[0]
                    << " h1 error = " << std::setw(7) << errors[1]
                    << " hd error = " << std::setw(7) << errors[2]);*/
        }
      }
    }
    
    if (gdim == 3)
    {    
      for (int l=0; l<lag_degrees3.size(); ++l)
      {      
        CONSOLE_OUTPUT(rank, "------------------------ ");
        CONSOLE_OUTPUT(rank, "test Lag space of degree " << lag_degrees3[l][0]);
        for (int i = 0; i < num_level; ++i) 
        {
          lag_space3.Init(*mesh[i], lag_fe_ansatz3, is_cg3, lag_degrees3[l]);
          CoupledVector<double> sol;
          interpolate<double, 3> (lag_space3, mesh[i], false, sol);
          
          StandardGlobalAssembler< double, 3 > global_asm;
          L2ErrorIntegrator<double, 3> local_asm_l2 (sol, 1., 0., 0.);
          L2ErrorIntegrator<double, 3> local_asm_h1 (sol, 0., 1., 0.);
          L2ErrorIntegrator<double, 3> local_asm_hd (sol, 0., 0., 1.);
          
          double L2_error = 0.;
          global_asm.integrate_scalar(lag_space3, boost::ref(local_asm_l2), L2_error );
          double H1_error = 0.;
          global_asm.integrate_scalar(lag_space3, boost::ref(local_asm_h1), H1_error );          
          double Hdiv_error = 0.;
          global_asm.integrate_scalar(lag_space3, boost::ref(local_asm_hd), Hdiv_error );
          
          double global_L2_error = 0.;
          double global_H1_error = 0.;
          double global_Hdiv_error = 0.;
          
          MPI_Reduce(&L2_error, &global_L2_error, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
          MPI_Reduce(&H1_error, &global_H1_error, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
          MPI_Reduce(&Hdiv_error, &global_Hdiv_error, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
           
          //std::vector<double> errors = compute_l2_error<double, 2> (bdm_space2, mesh[i], true, cell_types[test_number], sol);
    
          std::cout << std::scientific << std::setprecision(2);
          CONSOLE_OUTPUT(rank, "RefLevel [" << i << "]" 
                    << " L2 error = " << std::setw(7) << std::sqrt(global_L2_error)
                    << " H1 error = " << std::setw(7) << std::sqrt(global_H1_error)
                    << " Hdiv error = " << std::setw(7) << std::sqrt(global_Hdiv_error));
          /*
                    << " l2 error = " << std::setw(7) << errors[0]
                    << " h1 error = " << std::setw(7) << errors[1]
                    << " hd error = " << std::setw(7) << errors[2]);*/
        }
      }
    }
#endif
#if 1
    // RT element
    if (gdim == 2 && cell_types[test_number] == RefCellType::TRI_STD)
    {
      for (int l=0; l<rt_degrees.size(); ++l)
      {
        CONSOLE_OUTPUT(rank, "------------------------ ");
        CONSOLE_OUTPUT(rank, "test RT space of degree " << rt_degrees[l][0]);
          
        for (int i = 0; i < num_level; ++i) 
        {
          rt_space2.Init(*mesh[i], rt_fe_ansatz, is_cg, rt_degrees[l]);
          CoupledVector<double> sol;
          interpolate<double, 2> (rt_space2, mesh[i], true, sol);
          
          StandardGlobalAssembler< double, 2 > global_asm;
          L2ErrorIntegrator<double, 2> local_asm_l2 (sol, 1., 0., 0.);
          L2ErrorIntegrator<double, 2> local_asm_h1 (sol, 0., 1., 0.);
          L2ErrorIntegrator<double, 2> local_asm_hd (sol, 0., 0., 1.);
          
          double L2_error = 0.;
          global_asm.integrate_scalar(rt_space2, boost::ref(local_asm_l2), L2_error );
          double H1_error = 0.;
          global_asm.integrate_scalar(rt_space2, boost::ref(local_asm_h1), H1_error );          
          double Hdiv_error = 0.;
          global_asm.integrate_scalar(rt_space2, boost::ref(local_asm_hd), Hdiv_error );
          
          double global_L2_error = 0.;
          double global_H1_error = 0.;
          double global_Hdiv_error = 0.;
          
          MPI_Reduce(&L2_error, &global_L2_error, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
          MPI_Reduce(&H1_error, &global_H1_error, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
          MPI_Reduce(&Hdiv_error, &global_Hdiv_error, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
           
          //std::vector<double> errors = compute_l2_error<double, 2> (bdm_space2, mesh[i], true, cell_types[test_number], sol);
    
          std::cout << std::scientific << std::setprecision(2);
          CONSOLE_OUTPUT(rank, "RefLevel [" << i << "]" 
                    << " L2 error = " << std::setw(7) << std::sqrt(global_L2_error)
                    << " H1 error = " << std::setw(7) << std::sqrt(global_H1_error)
                    << " Hdiv error = " << std::setw(7) << std::sqrt(global_Hdiv_error));
          /*
                    << " l2 error = " << std::setw(7) << errors[0]
                    << " h1 error = " << std::setw(7) << errors[1]
                    << " hd error = " << std::setw(7) << errors[2]);*/
        }
      }
    }
    // RT element on quad
    if (gdim == 2 && cell_types[test_number] == RefCellType::QUAD_STD)
    {
      for (int l=0; l<rt_degrees.size(); ++l)
      {
        CONSOLE_OUTPUT(rank, "------------------------ ");
        CONSOLE_OUTPUT(rank, "test RT space of degree " << rt_degrees[l][0]);
          
        for (int i = 0; i < num_level; ++i) 
        {
          rt_space2.Init(*mesh[i], rt_fe_ansatz, is_cg, rt_degrees[l]);
          CoupledVector<double> sol;
          interpolate<double, 2> (rt_space2, mesh[i], true, sol);
          
          StandardGlobalAssembler< double, 2 > global_asm;
          L2ErrorIntegrator<double, 2> local_asm_l2 (sol, 1., 0., 0.);
          L2ErrorIntegrator<double, 2> local_asm_h1 (sol, 0., 1., 0.);
          L2ErrorIntegrator<double, 2> local_asm_hd (sol, 0., 0., 1.);
          
          double L2_error = 0.;
          global_asm.integrate_scalar(rt_space2, boost::ref(local_asm_l2), L2_error );
          double H1_error = 0.;
          global_asm.integrate_scalar(rt_space2, boost::ref(local_asm_h1), H1_error );          
          double Hdiv_error = 0.;
          global_asm.integrate_scalar(rt_space2, boost::ref(local_asm_hd), Hdiv_error );
          
          double global_L2_error = 0.;
          double global_H1_error = 0.;
          double global_Hdiv_error = 0.;
          
          MPI_Reduce(&L2_error, &global_L2_error, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
          MPI_Reduce(&H1_error, &global_H1_error, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
          MPI_Reduce(&Hdiv_error, &global_Hdiv_error, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
           
          //std::vector<double> errors = compute_l2_error<double, 2> (bdm_space2, mesh[i], true, cell_types[test_number], sol);
    
          std::cout << std::scientific << std::setprecision(2);
          CONSOLE_OUTPUT(rank, "RefLevel [" << i << "]" 
                    << " L2 error = " << std::setw(7) << std::sqrt(global_L2_error)
                    << " H1 error = " << std::setw(7) << std::sqrt(global_H1_error)
                    << " Hdiv error = " << std::setw(7) << std::sqrt(global_Hdiv_error));
          /*
                    << " l2 error = " << std::setw(7) << errors[0]
                    << " h1 error = " << std::setw(7) << errors[1]
                    << " hd error = " << std::setw(7) << errors[2]);*/
        }
      }
    }
    // RT element on tet
    if (gdim == 3 && cell_types[test_number] == RefCellType::TET_STD)
    {
      for (int l=0; l<rt_degrees.size(); ++l)
      {
        CONSOLE_OUTPUT(rank, "------------------------ ");
        CONSOLE_OUTPUT(rank, "test RT space of degree " << rt_degrees[l][0]);
          
        for (int i = 0; i < num_level; ++i) 
        {
          rt_space3.Init(*mesh[i], rt_fe_ansatz, is_cg, rt_degrees[l]);
          CoupledVector<double> sol;
          interpolate<double, 3> (rt_space3, mesh[i], true, sol);
          
#ifdef VISUALIZE
	  std::vector<size_t> u_vars(3, 0);
	  std::vector< std::string> u_names;
	  for (size_t d=0; d<3; ++d)
	  {
	    u_vars[d] = d;
	    u_names.push_back("u_" + std::to_string(d));
	  }
	  
	  std::string filename = "FeInterpolConv_" + std::string("Tet") + "_lvl" + std::to_string(i) 
				+ "_" + "RT"  
				+ "_" + std::to_string(rt_degrees[l][0]) + ".vtu";
	  
	  std::vector< std::vector< double > > cell_index (num_level);

	  for (int i=0; i < num_level; ++i)
	  {     
	    cell_index[i].resize(mesh[i]->num_entities(tdim), 0);
	    for (int c=0; c<mesh[i]->num_entities(tdim); ++c)
	    {
	      cell_index[i][c] = static_cast<double>(c);
	    }
	  }
	    
	  CellVisualization< double, 3> visu(rt_space3, 1);
	  visu.visualize(sol, u_vars, u_names);
	  visu.visualize_cell_data(cell_index[i], "_cell_index_");
	  VTKWriter< double, 3> vtk_writer (visu, MPI_COMM_WORLD, 0);
	  vtk_writer.write(filename);
#endif

          StandardGlobalAssembler< double, 3 > global_asm;
          L2ErrorIntegrator<double, 3> local_asm_l2 (sol, 1., 0., 0.);
          L2ErrorIntegrator<double, 3> local_asm_h1 (sol, 0., 1., 0.);
          L2ErrorIntegrator<double, 3> local_asm_hd (sol, 0., 0., 1.);
          
          double L2_error = 0.;
          global_asm.integrate_scalar(rt_space3, boost::ref(local_asm_l2), L2_error );
          double H1_error = 0.;
          global_asm.integrate_scalar(rt_space3, boost::ref(local_asm_h1), H1_error );          
          double Hdiv_error = 0.;
          global_asm.integrate_scalar(rt_space3, boost::ref(local_asm_hd), Hdiv_error );
          
          double global_L2_error = 0.;
          double global_H1_error = 0.;
          double global_Hdiv_error = 0.;
          
          MPI_Reduce(&L2_error, &global_L2_error, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
          MPI_Reduce(&H1_error, &global_H1_error, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
          MPI_Reduce(&Hdiv_error, &global_Hdiv_error, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
           
          //std::vector<double> errors = compute_l2_error<double, 2> (bdm_space2, mesh[i], true, cell_types[test_number], sol);
    
          std::cout << std::scientific << std::setprecision(2);
          CONSOLE_OUTPUT(rank, "RefLevel [" << i << "]" 
                    << " L2 error = " << std::setw(7) << std::sqrt(global_L2_error)
                    << " H1 error = " << std::setw(7) << std::sqrt(global_H1_error)
                    << " Hdiv error = " << std::setw(7) << std::sqrt(global_Hdiv_error));
          /*
                    << " l2 error = " << std::setw(7) << errors[0]
                    << " h1 error = " << std::setw(7) << errors[1]
                    << " hd error = " << std::setw(7) << errors[2]);*/
        }
      }
    }
    // RT element on hex
    if (gdim == 3 && cell_types[test_number] == RefCellType::HEX_STD)
    {
      for (int l=0; l<rt_degrees.size(); ++l)
      {
        CONSOLE_OUTPUT(rank, "------------------------ ");
        CONSOLE_OUTPUT(rank, "test RT space of degree " << rt_degrees[l][0]);
          
        for (int i = 0; i < num_level; ++i) 
        {
          rt_space3.Init(*mesh[i], rt_fe_ansatz, is_cg, rt_degrees[l]);
          CoupledVector<double> sol;
          interpolate<double, 3> (rt_space3, mesh[i], true, sol);
          
          StandardGlobalAssembler< double, 3 > global_asm;
          L2ErrorIntegrator<double, 3> local_asm_l2 (sol, 1., 0., 0.);
          L2ErrorIntegrator<double, 3> local_asm_h1 (sol, 0., 1., 0.);
          L2ErrorIntegrator<double, 3> local_asm_hd (sol, 0., 0., 1.);
          
          double L2_error = 0.;
          global_asm.integrate_scalar(rt_space3, boost::ref(local_asm_l2), L2_error );
          double H1_error = 0.;
          global_asm.integrate_scalar(rt_space3, boost::ref(local_asm_h1), H1_error );          
          double Hdiv_error = 0.;
          global_asm.integrate_scalar(rt_space3, boost::ref(local_asm_hd), Hdiv_error );
          
          double global_L2_error = 0.;
          double global_H1_error = 0.;
          double global_Hdiv_error = 0.;
          
          MPI_Reduce(&L2_error, &global_L2_error, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
          MPI_Reduce(&H1_error, &global_H1_error, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
          MPI_Reduce(&Hdiv_error, &global_Hdiv_error, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
           
          //std::vector<double> errors = compute_l2_error<double, 2> (bdm_space2, mesh[i], true, cell_types[test_number], sol);
    
          std::cout << std::scientific << std::setprecision(2);
          CONSOLE_OUTPUT(rank, "RefLevel [" << i << "]" 
                    << " L2 error = " << std::setw(7) << std::sqrt(global_L2_error)
                    << " H1 error = " << std::setw(7) << std::sqrt(global_H1_error)
                    << " Hdiv error = " << std::setw(7) << std::sqrt(global_Hdiv_error));
          /*
                    << " l2 error = " << std::setw(7) << errors[0]
                    << " h1 error = " << std::setw(7) << errors[1]
                    << " hd error = " << std::setw(7) << errors[2]);*/
        }
      }
    }


#endif
  }

  MPI_Finalize();

  return 0;
}
