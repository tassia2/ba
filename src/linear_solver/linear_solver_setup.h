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

/// @author Philipp Gerstner

#ifndef HIFLOW_LINEARSOLVER_LINEAR_SOLVER_SETUP_H_
#define HIFLOW_LINEARSOLVER_LINEAR_SOLVER_SETUP_H_

#include "common/property_tree.h"
#include "linear_solver/linear_solver.h"
#include "linear_solver/preconditioner_bjacobi_standard.h"
#include "linear_solver/preconditioner_bjacobi_ext.h"
#include "linear_solver/preconditioner_vanka.h"
#include "linear_solver/richardson.h"
#include "linear_solver/gmres.h"
#include "linear_solver/fgmres.h"
#include "linear_solver/cg.h"

namespace hiflow {
namespace la {

template < class LAD >
void prepare_bjacobi_std(PreconditionerBlockJacobiStand< LAD >* precond,
                         const std::string local_solver_type, 
                         const PropertyTree &params) 
{
  typedef typename LAD::DataType DataType;

  PropertyTree cur_param = params[local_solver_type];
  assert (precond != nullptr);
  
  if (local_solver_type == "HiflowILU")
  {
    precond->Init_ILUp(cur_param["Bandwidth"].get<int>());
  }
  else if (local_solver_type == "SOR")
  {
    precond->Init_SOR(cur_param["Omega"].get<DataType>());
  }
  else if (local_solver_type == "SSOR")
  {
    precond->Init_SSOR(cur_param["Omega"].get<DataType>());
  }
  else if (local_solver_type == "Jacobi")
  {
    precond->Init_Jacobi();
  }
  else if (local_solver_type == "FSAI")
  {
    precond->Init_FSAI(cur_param["NumIter"].get<int>(),
                       cur_param["RelRes"].get<DataType>(),
                       cur_param["AbsRes"].get<DataType>(),
                       cur_param["Power"].get<int>() );
  }
  else
  {
    assert (false);
  }
}

template < class LAD >
void prepare_bjacobi_ext(PreconditionerBlockJacobiExt< LAD >* precond,
                         const std::string local_solver_type,  
                         const PropertyTree &params) 
{
  assert (precond != nullptr);

  precond->Init(local_solver_type, params);
  PropertyTree cur_param = params[local_solver_type];
}

template < class LAD, int DIM >
void prepare_vanka(PreconditionerVanka< LAD, DIM >* precond, 
                   const PropertyTree &params,
                   const VectorSpace<typename LAD::DataType, DIM>* space) 
{
  typedef typename LAD::DataType DataType;

  assert (precond != nullptr);
  //PropertyTree cur_param = params["Vanka"];
  PropertyTree cur_param = params;
  
  int patch_mode = cur_param["PatchMode"].get< int >();
  VankaPatchMode vanka_patch = VankaPatchMode::SingleCell;
  if (patch_mode == 1)
    vanka_patch = VankaPatchMode::VertexPatch;
  else if (patch_mode == 2)
    vanka_patch = VankaPatchMode::FacetPatch;
  else if (patch_mode == 3)
    vanka_patch = VankaPatchMode::CellPatch;
  
  precond->InitParameter(
      *space,
      cur_param["Damping"].get< DataType >(),
      cur_param["NumIter"].get< int >(),
      cur_param["UseILUPP"].get< bool >(),
      vanka_patch,
      cur_param["PrebuildMatrices"].get< bool >(),
      -1);
}

template < class LAD >
void prepare_richardson(Richardson< LAD >* precond, 
                        const PropertyTree &params) 
{
  assert (precond != nullptr);
  NOT_YET_IMPLEMENTED;
}

template < class LAD, int DIM >
void prepare_krylov_solver(LinearSolver< LAD >*& solver, 
                           Preconditioner< LAD>*& preconditioner,
                           const PropertyTree &param,
                           const PropertyTree &locsolver_param,
                           const VectorSpace<typename LAD::DataType, DIM>* space,
                           NonlinearProblem< LAD > *app) 
{
  //typedef typename LAD::DataType DataType;

  std::string solver_type = param["Type"].get<std::string>();
  std::string precond_type = param["PrecondType"].get<std::string>();
  
  if (solver_type == "GMRES")
  {
    GMRES<LAD>* gmres = new GMRES<LAD>();
    setup_GMRES_solver<LAD, LAD> (*gmres, param, app);
    solver = gmres;
  }
  else if (solver_type == "FGMRES")
  {
    FGMRES<LAD>* fgmres = new FGMRES<LAD>();
    setup_FGMRES_solver<LAD, LAD> (*fgmres, param, app);
    solver = fgmres;
  }
  else if (solver_type == "CG")
  {
    CG<LAD>* cg = new CG<LAD>();
    setup_CG_solver<LAD> (*cg, param, app);
    solver = cg;
  }
  else
  {
    assert (false);
  }
  
  if (precond_type == "BlockJacobiStd")
  { 
    PreconditionerBlockJacobiStand< LAD >* precond = new PreconditionerBlockJacobiStand< LAD >();
    
    prepare_bjacobi_std(precond, param["LocalSolverType"].get<std::string>(), locsolver_param);
    solver->SetupPreconditioner(*precond);
    preconditioner = precond;
  }
  else if (precond_type == "BlockJacobiExt")
  {
    PreconditionerBlockJacobiExt< LAD >* precond = new PreconditionerBlockJacobiExt< LAD >();
        
    prepare_bjacobi_ext(precond, param["LocalSolverType"].get<std::string>(), locsolver_param);
    solver->SetupPreconditioner(*precond);
    preconditioner = precond;
  }
  else if (precond_type == "Vanka")
  {
    PreconditionerVanka< LAD, DIM >* precond = new PreconditionerVanka< LAD, DIM >();
    prepare_vanka(precond, locsolver_param, space);
    solver->SetupPreconditioner(*precond);
    preconditioner = precond;
  }
}

template < class LAD, int DIM >
void prepare_krylov_solver(LinearSolver< LAD >*& solver, 
                           const PropertyTree &param,
                           NonlinearProblem< LAD > *app) 
{
  //typedef typename LAD::DataType DataType;

  std::string solver_type = param["Type"].get<std::string>();
  std::string precond_type = param["PrecondType"].get<std::string>();
  
  if (solver_type == "GMRES")
  {
    GMRES<LAD>* gmres = new GMRES<LAD>();
    setup_GMRES_solver<LAD, LAD> (*gmres, param, app);
    solver = gmres;
  }
  else if (solver_type == "FGMRES")
  {
    FGMRES<LAD>* fgmres = new FGMRES<LAD>();
    setup_FGMRES_solver<LAD, LAD> (*fgmres, param, app);
    solver = fgmres;
  }
  else if (solver_type == "CG")
  {
    CG<LAD>* cg = new CG<LAD>();
    setup_CG_solver<LAD> (*cg, param, app);
    solver = cg;
  }
  else
  {
    assert (false);
  }
}



} // namespace la
} // namespace hiflow

#endif // HIFLOW_LINEARSOLVER_LINEAR_SOLVER_SETUP_H_
