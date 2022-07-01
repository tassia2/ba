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

#define TEST_LAG
#define TEST_BDM
#define TEST_RT

#define TEST_TRI
#define TEST_QUAD
#define TEST_TET
#define TEST_HEX

#define VISUALIZE

const int NUM_LEVEL = 2;

const int NUM_DEG_LAG = 2;
const int NUM_DEG_RT = 2;
const int NUM_DEG_BDM = 2;

const int NUM_PROJECTION_TYPES = 1; // 1: L2, 2: L2+H1, 3: L2+H1+Hdiv

/// 
///
/// \brief 
///

const int DEBUG_OUT = 1;
const double pi = 3.14159265;

template <class DataType, int DIM>
struct QuadratureSelection 
{
  QuadratureSelection(int order) : order_(order) {}

  void operator()(const Element< DataType, DIM > &elem,
                  Quadrature< DataType > &quadrature) {
    const RefCellType cell_id = elem.ref_cell()->type();

    switch (cell_id) 
    {
    case RefCellType::TRI_STD:
      quadrature.set_cell_tag(CellType::TRIANGLE);
      quadrature.set_quadrature_by_order("GaussTriangle", order_);
      break;
    case RefCellType::QUAD_STD:
      quadrature.set_cell_tag(CellType::QUADRILATERAL);
      quadrature.set_quadrature_by_order("GaussQuadrilateral", order_);
      break;
    case RefCellType::TET_STD:
      quadrature.set_cell_tag(CellType::TETRAHEDRON);
      quadrature.set_quadrature_by_order("GaussTetrahedron", order_);
      break;
    case RefCellType::HEX_STD:
      quadrature.set_cell_tag(CellType::HEXAHEDRON);
      quadrature.set_quadrature_by_order("GaussHexahedron", order_);
      break;
    default:
      assert(false);
    };
  }

  int order_;
};

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
class ProjectionAssembler : private AssemblyAssistant< DIM, DataType > 
{
  typedef la::SeqDenseMatrix< DataType > LocalMatrix;
  typedef std::vector< DataType > LocalVector;
  
public:
  ProjectionAssembler(CoupledVector< DataType > &pp_sol, int type)
      : pp_sol_(pp_sol), type_(type) {}


  void operator()(const Element< DataType, DIM > &element,
                  const Quadrature< DataType > &quadrature, LocalMatrix &lm) 
  {
    AssemblyAssistant< DIM, DataType >::initialize_for_element(element, quadrature, false);

    const int num_q = this->num_quadrature_points();

    // loop q
    std::vector<DataType> phi_i (DIM, 0.);
    std::vector< Vec<DIM, DataType> > grad_phi_i (DIM);
    std::vector<DataType> phi_j (DIM, 0.);
    std::vector< Vec<DIM, DataType> > grad_phi_j (DIM);

    for (int q = 0; q < num_q; ++q) 
    {
      const DataType wq = this->w(q);
      const DataType dJ = std::abs(this->detJ(q));

      // loop over test functions
      for (int i = 0; i < this->num_dofs_total(); ++i) 
      {    
        // get ansatz function values 
        for (size_t var = 0; var < DIM; ++var)
        {
          phi_i[var] = this->Phi(i, q, var);
          grad_phi_i[var] = this->grad_Phi(i, q, var);
        }
  
        // precompute divergence of test function
        DataType div_i = 0.;
        for (size_t var = 0; var < DIM; ++var)
        {
          div_i += grad_phi_i[var][var];
        }
            
        // loop over trial functions
        for (int j = 0; j < this->num_dofs_total(); ++j) 
        {
          // get ansatz function values 
          for (size_t var = 0; var < DIM; ++var)
          {
            phi_j[var] = this->Phi(j, q, var);
            grad_phi_j[var] = this->grad_Phi(j, q, var);
          }

          // precompute divergence of test function
          DataType div_j = 0.;
          for (size_t var = 0; var < DIM; ++var)
          {
            div_j += grad_phi_j[var][var];
          }
        
          // ----- start assembly of individual terms in variational formulation ------
          // mass tern: a0(u,v) = \int { u * v }
          // laplace term : a1(u,v) = \int {\grad(u) : \grad(v)}
          // div term: a2(u,v) = \int {\div(u) * \div(v)}
            
          DataType laplace = 0.;
          DataType mass = 0.;
          DataType div = div_j * div_i;
          for (size_t var = 0; var < DIM; ++var)
          {
            mass += phi_j[var] * phi_i[var];
            laplace += dot(grad_phi_j[var], grad_phi_i[var]);
          }
          
          // L2 projection
          lm(i, j) += wq * mass * dJ;
          
          if (type_ == 1)
          {
          // H1 projection
            lm(i, j) += wq * laplace * dJ; 
          }
          if (type_ == 2)
          {
          // H(div) projection
            lm(i, j) += wq * div * dJ; 
          }
        }
      }
    }
  }
  
  void operator()(const Element< DataType, DIM > &element,
                  const Quadrature< DataType > &quadrature, LocalVector &lv) 
  {
    AssemblyAssistant< DIM, DataType >::initialize_for_element(element, quadrature, false);

    // indices j -> trial variable, i -> test variable
    // basis functions \phi -> velocity components, \eta -> pressure

    const int num_q = this->num_quadrature_points();
    std::vector<DataType> phi_i (DIM+1, 0.);
    std::vector< Vec<DIM, DataType> > grad_phi_i (DIM+1);
    std::vector<DataType> phi_j (DIM+1, 0.);
    std::vector< Vec<DIM, DataType> > grad_phi_j (DIM+1);

    // loop quadrature points
    for (int q = 0; q < num_q; ++q) 
    {
      const DataType wq = this->w(q);
      const DataType dJ = std::abs(this->detJ(q));

      // loop over test functions
      for (int i = 0; i < this->num_dofs_total(); ++i) 
      {    
        // get ansatz function values 
        for (size_t var = 0; var < DIM; ++var)
        {
          phi_i[var] = this->Phi(i, q, var);
          grad_phi_i[var] = this->grad_Phi(i, q, var);
        }
        
        DataType div_i = 0.;
        for (size_t var = 0; var < DIM; ++var)
        {
          div_i += grad_phi_i[var][var];
        }
          
        Vec<DIM, DataType> exact;
        std::vector< Vec<DIM, DataType> > exact_grad(DIM);
        DataType exact_div = 0.;
        
        for (size_t d=0; d<DIM; ++d)
        {
          exact[d] = func_.evaluate(d, this->x(q));
          exact_grad[d] = func_.evaluate_grad(d, this->x(q));
          exact_div += exact_grad[d][d];
        }
        
        DataType mass = 0.;
        DataType laplace = 0.;
        for (size_t var = 0; var < DIM; ++var)
        {
            mass += exact[var] * phi_i[var];
            laplace += dot( exact_grad[var], grad_phi_i[var]);
        }
        DataType div =  exact_div * div_i;
                
        // L2 projection
        lv[i] += wq * mass * dJ;
          
        if (type_ == 1)
        {
        // H1 projection
          lv[i] += wq * laplace * dJ; 
        }
        if (type_ == 2)
        {
        // H(div) projection
          lv[i] += wq * div * dJ; 
        }
      }
    }
  }
  
  
  void operator()(const Element< DataType, DIM > &element,
                  const Quadrature< DataType > &quadrature, 
                  DataType &value) 
  {
    AssemblyAssistant< DIM, DataType >::initialize_for_element(element, quadrature, false);

    const int num_q = this->num_quadrature_points();
    for (int v = 0; v < DIM; ++v) 
    {
      approx_sol_.clear();
      this->evaluate_fe_function(pp_sol_, v, approx_sol_);

      for (int q = 0; q < num_q; ++q) 
      {
        const DataType wq = this->w(q);
        const Vec<DIM, DataType> xq = this->x(q);
        const DataType dJ = std::abs(this->detJ(q));
        DataType exact = this->func_.evaluate (v, xq);
       
        value += wq * (exact - this->approx_sol_[q]) 
                    * (exact - this->approx_sol_[q]) * dJ;
      
      }
    }
  }

private:
  int type_;
  
  // coefficients of the computed solution
  CoupledVector< DataType > &pp_sol_;
  // vector with values of computed solution evaluated at each quadrature point
  FunctionValues< DataType > approx_sol_;
  // variables for which to compute the norm
  TestFunction<DataType, DIM> func_;
};

template <class DataType, int DIM>
void project (int projection_type,
              VectorSpace<DataType,DIM> &space, 
              MeshPtr mesh_ptr,
              CoupledVector<DataType>& sol)
{
  // setup LA
  sol.Init(MPI_COMM_WORLD, space.la_couplings());
  sol.InitStructure();
  sol.Zeros();

  CoupledVector<DataType> rhs;
  rhs.Init(MPI_COMM_WORLD, space.la_couplings());
  rhs.InitStructure();
  rhs.Zeros();

  SparsityStructure sparsity;
  int nb_fe = space.nb_fe();
    
  std::vector< std::vector< bool > > coupling_vars(nb_fe);
  for (int l=0; l<nb_fe; ++l)
  {
    for(int k=0; k<nb_fe; ++k)
    {
      coupling_vars[l].push_back(true);
    }
  }

  StandardGlobalAssembler< DataType, DIM > global_asm;
  compute_sparsity_structure(space, sparsity, coupling_vars, false);

  CoupledMatrix<DataType> A;
  A.Init(MPI_COMM_WORLD, space.la_couplings());
  A.InitStructure(vec2ptr(sparsity.diagonal_rows), vec2ptr(sparsity.diagonal_cols),
                  sparsity.diagonal_rows.size(), vec2ptr(sparsity.off_diagonal_rows),
                  vec2ptr(sparsity.off_diagonal_cols), sparsity.off_diagonal_rows.size());
  A.Zeros();
  
  // setup solver
  GMRES< LADescriptorCoupledD > gmres;
#ifdef WITH_ILUPP
  PreconditionerIlupp< LADescriptorCoupledD > ilupp;
  ilupp.InitParameter(0, 11, 20, 0.8, 2.75, 0.05);
  ilupp.SetupOperator(A);
  gmres.SetupPreconditioner(ilupp);
  gmres.InitParameter(300, "RightPreconditioning");
#else
  gmres.InitParameter(300, "NoPreconditioning");
#endif
  gmres.InitControl(500, 1e-10, 1e-6, 1e6);
  gmres.SetupOperator(A);

  // assemble system
  QuadratureSelection<DataType, DIM> q_sel (2 * space.fe_manager().get_fe(0,0)->max_deg());
  global_asm.set_quadrature_selection_function(q_sel);
    
  ProjectionAssembler<DataType, DIM> local_asm (sol, projection_type);
  global_asm.assemble_matrix(space, local_asm, A);
  global_asm.assemble_vector(space, local_asm, rhs);
  rhs.Update();

  // solve system
  gmres.Solve(rhs, &sol);
  sol.Update();
}
 
template <class DataType, int DIM>
DataType compute_l2_error (VectorSpace<DataType,DIM> &space, 
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
  
  DataType sum = 0.;
  int number = 0;
  
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
      
      Vec<DIM, DataType> pt = test_pts[l];
      std::vector< DataType> fe_vals;
      fe_eval.evaluate(cell, pt, fe_vals);
      assert (fe_vals.size() == DIM);
      
      for (size_t v=0; v<DIM; ++v)
      {    
        DataType exact_val = func.evaluate(v, pt);
      
        sum += (fe_vals[v] - exact_val) * (fe_vals[v] - exact_val);
      }
    }
  }
  return std::sqrt( sum / number);
}

        
static const char *datadir = MESH_DATADIR;

int main(int argc, char *argv[]) {
  int init_level = 0;
  int num_level = NUM_LEVEL;
  int rank;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD,  &rank);

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
  std::vector<size_t> u_vars2(2, 0);
  std::vector<size_t> u_vars3(3, 0);
  for (size_t d=0; d<2; ++d)
  {
    u_vars2[d] = d;
    u_vars3[d] = d;
  }
  u_vars3.push_back(2);
  
  std::vector< std::string> u_names2;
  std::vector< std::string> u_names3;
  for (size_t d=0; d<2; ++d)
  {
    u_names2.push_back("u_" + std::to_string(d));
    u_names3.push_back("u_" + std::to_string(d));
  }
  u_names3.push_back("u_" + std::to_string(2));
  
   
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

  std::vector< std::vector< int > > lag_degrees2(NUM_DEG_LAG);
  for (int l=0; l < lag_degrees2.size(); ++l)
  {
    lag_degrees2[l].resize(2,l+1);
  }
  
  std::vector< std::vector< int > > lag_degrees3(NUM_DEG_LAG);
  for (int l=0; l < lag_degrees3.size(); ++l)
  {
    lag_degrees3[l].resize(3,l+1);
  }
  
  std::vector< std::vector< int > > rt_degrees(NUM_DEG_RT);
  for (int l=0; l < rt_degrees.size(); ++l)
  {
    rt_degrees[l].resize(1,l);
  }
  
  std::vector< std::vector< int > > bdm_degrees(NUM_DEG_BDM);
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

#ifdef TEST_RT
    // RT element
    if (gdim == 2 && ( cell_types[test_number] == RefCellType::TRI_STD || cell_types[test_number] == RefCellType::QUAD_STD))
    {
      for (int projection_type = 0; projection_type < NUM_PROJECTION_TYPES; ++projection_type)
      {
        for (int l=0; l<rt_degrees.size(); ++l)
        {
          CONSOLE_OUTPUT(rank, "------------------------ ");
          CONSOLE_OUTPUT(rank, "test RT  space of degree " << rt_degrees[l][0] << " , projection type " << projection_type);
            
          for (int i = 0; i < num_level; ++i) 
          {
            rt_space2.Init(*mesh[i], rt_fe_ansatz, is_cg, rt_degrees[l]);
            CoupledVector<double> sol;
            project<double, 2> (projection_type, rt_space2, mesh[i], sol);
            
            ProjectionAssembler<double, 2> local_asm (sol, projection_type);
            StandardGlobalAssembler< double, 2 > global_asm;
            
            double L2_error = 0.;
            global_asm.integrate_scalar(rt_space2, boost::ref(local_asm), L2_error);
            double l2_error = compute_l2_error<double, 2> (rt_space2, mesh[i], true, cell_types[test_number], sol);
      
            std::cout << std::scientific << std::setprecision(2);
            CONSOLE_OUTPUT(rank, "RefLevel [" << init_level + i << "] L2 error = " << std::setw(7) << std::sqrt(L2_error) 
                            << " , l2 error = " << std::setw(7) << l2_error);
          
            std::string filename = "FeProjectionTest_" + cell_prefix + "_proj" + std::to_string(projection_type)
                                 + "_RT" + std::to_string(rt_degrees[l][0])  
                                 + "_lvl" + std::to_string(init_level + i) + ".vtu";
                                 
#ifdef VISUALIZE
            CellVisualization< double, 2> visu(rt_space2, 1);
            visu.visualize(FeEvalCell< double, 2 >(rt_space2, sol, u_vars2, nullptr), u_names2);
            VTKWriter< double, 2> vtk_writer (visu, MPI_COMM_WORLD, 0);
            vtk_writer.write(filename);
#endif
          }
        }
      }
    }
    // RT element on tet
    if (gdim == 3 && (cell_types[test_number] == RefCellType::TET_STD || cell_types[test_number] == RefCellType::HEX_STD))
    {
      for (int projection_type = 0; projection_type < NUM_PROJECTION_TYPES; ++projection_type)
      {
        for (int l=0; l<rt_degrees.size(); ++l)
        {
          CONSOLE_OUTPUT(rank, "------------------------ ");
          CONSOLE_OUTPUT(rank, "test RT  space of degree " << rt_degrees[l][0] << " , projection type " << projection_type);
            
          for (int i = 0; i < num_level; ++i) 
          {
            rt_space3.Init(*mesh[i], rt_fe_ansatz, is_cg, rt_degrees[l]);
            CoupledVector<double> sol;
            project<double, 3> (projection_type, rt_space3, mesh[i], sol);
            
            ProjectionAssembler<double, 3> local_asm (sol, projection_type);
            StandardGlobalAssembler< double, 3 > global_asm;
            
            double L2_error = 0.;
            global_asm.integrate_scalar(rt_space3, boost::ref(local_asm), L2_error);
            double l2_error = compute_l2_error<double, 3> (rt_space3, mesh[i], true, cell_types[test_number], sol);
      
            std::cout << std::scientific << std::setprecision(2);
            CONSOLE_OUTPUT(rank, "RefLevel [" << init_level + i << "] L2 error = " << std::setw(7) << std::sqrt(L2_error) 
                            << " , l2 error = " << std::setw(7) << l2_error);
          
            std::string filename = "FeProjectionTest_" + cell_prefix + "_proj" + std::to_string(projection_type)
                                 + "_RT" + std::to_string(rt_degrees[l][0])  
                                 + "_lvl" + std::to_string(init_level + i) + ".vtu";

#ifdef VISUALIZE                                 
            CellVisualization< double, 3> visu(rt_space3, 1);
            visu.visualize(FeEvalCell< double, 3 >(rt_space3, sol, u_vars3, nullptr), u_names3);
            VTKWriter< double, 3> vtk_writer (visu, MPI_COMM_WORLD, 0);
            vtk_writer.write(filename);
#endif
          }
        }
      }
    }
#endif
#ifdef TEST_BDM
    // BDM element
    if (gdim == 2 && cell_types[test_number] == RefCellType::TRI_STD)
    {
      for (int projection_type = 0; projection_type < NUM_PROJECTION_TYPES; ++projection_type)
      {
        for (int l=0; l<bdm_degrees.size(); ++l)
        {
          CONSOLE_OUTPUT(rank, "------------------------ ");
          CONSOLE_OUTPUT(rank, "test BDM space of degree " << bdm_degrees[l][0] << " , projection type " << projection_type);
            
          for (int i = 0; i < num_level; ++i) 
          {
            bdm_space2.Init(*mesh[i], bdm_fe_ansatz, is_cg, bdm_degrees[l]);
            CoupledVector<double> sol;
            project<double, 2> (projection_type, bdm_space2, mesh[i], sol);
            
            ProjectionAssembler<double, 2> local_asm (sol, projection_type);
            StandardGlobalAssembler< double, 2 > global_asm;
            
            double L2_error = 0.;
            global_asm.integrate_scalar(bdm_space2, boost::ref(local_asm), L2_error);
            double l2_error = compute_l2_error<double, 2> (bdm_space2, mesh[i], true, cell_types[test_number], sol);
      
            std::cout << std::scientific << std::setprecision(2);
            CONSOLE_OUTPUT(rank, "RefLevel [" << init_level + i << "] L2 error = " << std::setw(7) << std::sqrt(L2_error) 
                      << " , l2 error = " << std::setw(7) << l2_error);
          
            std::string filename = "FeProjectionTest_" + cell_prefix + "_proj" + std::to_string(projection_type)
                                 + "_BDM" + std::to_string(bdm_degrees[l][0])  
                                 + "_lvl" + std::to_string(init_level + i) + ".vtu";
#ifdef VISUALIZE                                 
            CellVisualization< double, 2> visu(bdm_space2, 1);
            visu.visualize(FeEvalCell< double, 2 >(bdm_space2, sol, u_vars2, nullptr), u_names2);
            VTKWriter< double, 2> vtk_writer (visu, MPI_COMM_WORLD, 0);
            vtk_writer.write(filename);
#endif
          }
        }
      }
    }
#endif
#ifdef TEST_LAG
    // Lagrange element
    if (gdim == 2)
    {
      for (int projection_type = 0; projection_type < NUM_PROJECTION_TYPES; ++projection_type)
      {
        for (int l=0; l<lag_degrees2.size(); ++l)
        {
          CONSOLE_OUTPUT(rank, "------------------------ ");
          CONSOLE_OUTPUT(rank, "test Lag space of degree " << lag_degrees2[l][0] << " , projection type " << projection_type);
          for (int i = 0; i < num_level; ++i) 
          {
            lag_space2.Init(*mesh[i], lag_fe_ansatz2, is_cg2, lag_degrees2[l]);
            CoupledVector<double> sol;
            project<double, 2> (projection_type, lag_space2, mesh[i], sol);
            
            ProjectionAssembler<double, 2> local_asm (sol, projection_type);
            StandardGlobalAssembler< double, 2 > global_asm;
            
            double L2_error = 0.;
            global_asm.integrate_scalar(lag_space2, boost::ref(local_asm), L2_error);
                          
            double l2_error = compute_l2_error<double, 2> (lag_space2, mesh[i], false, cell_types[test_number], sol);
      
            std::cout << std::scientific << std::setprecision(2);
            CONSOLE_OUTPUT(rank, "RefLevel [" << init_level + i << "] L2 error = " << std::setw(7) << std::sqrt(L2_error) 
                            << " , l2 error = " << std::setw(7) << l2_error);
                      
            std::string filename = "FeProjectionTest_" + cell_prefix + "_proj" + std::to_string(projection_type)
                                 + "_LAG" + std::to_string(lag_degrees2[l][0])  
                                 + "_lvl" + std::to_string(init_level + i) + ".vtu";
#ifdef VISUALIZE                                 
            CellVisualization< double, 2> visu(lag_space2, 1);
            visu.visualize(FeEvalCell< double, 2 >(lag_space2, sol, u_vars2, nullptr), u_names2);
            VTKWriter< double, 2> vtk_writer (visu, MPI_COMM_WORLD, 0);
            vtk_writer.write(filename);
#endif
          }
        }
      }
    }
    if (gdim == 3)
    {
      for (int projection_type = 0; projection_type < NUM_PROJECTION_TYPES; ++projection_type)
      {
        for (int l=0; l<lag_degrees3.size(); ++l)
        {
          CONSOLE_OUTPUT(rank, "------------------------ ");
          CONSOLE_OUTPUT(rank, "test Lag space of degree " << lag_degrees3[l][0] << " , projection type " << projection_type);
          for (int i = 0; i < num_level; ++i) 
          {
            lag_space3.Init(*mesh[i], lag_fe_ansatz3, is_cg3, lag_degrees3[l]);
            CoupledVector<double> sol;
            project<double, 3> (projection_type, lag_space3, mesh[i], sol);
            
            ProjectionAssembler<double, 3> local_asm (sol, projection_type);
            StandardGlobalAssembler< double, 3> global_asm;
            
            double L2_error = 0.;
            global_asm.integrate_scalar(lag_space3, boost::ref(local_asm), L2_error);
                          
            double l2_error = compute_l2_error<double, 3> (lag_space3, mesh[i], false, cell_types[test_number], sol);
      
            std::cout << std::scientific << std::setprecision(2);
            CONSOLE_OUTPUT(rank, "RefLevel [" << init_level + i << "] L2 error = " << std::setw(7) << std::sqrt(L2_error) 
                            << " , l2 error = " << std::setw(7) << l2_error);
                      
            std::string filename = "FeProjectionTest_" + cell_prefix + "_proj" + std::to_string(projection_type)
                                 + "_LAG" + std::to_string(lag_degrees3[l][0])  
                                 + "_lvl" + std::to_string(init_level + i) + ".vtu";
#ifdef VISUALIZE                                 
            CellVisualization< double, 3> visu(lag_space3, 1);
            visu.visualize(FeEvalCell< double, 3 >(lag_space3, sol, u_vars3, nullptr), u_names3);
            VTKWriter< double, 3> vtk_writer (visu, MPI_COMM_WORLD, 0);
            vtk_writer.write(filename);
#endif
          }
        }
      }
    }
#endif
  }

  MPI_Finalize();

  return 0;
}
