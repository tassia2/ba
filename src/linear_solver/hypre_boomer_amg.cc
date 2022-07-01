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

/// @author Simon Gawlok

#include "linear_solver/hypre_boomer_amg.h"
#include "common/macros.h"

namespace hiflow {
namespace la {

template < class LAD > HypreBoomerAMG< LAD >::HypreBoomerAMG() {
  this->initialized_ = false;
  this->res_ = 0.;
  this->iter_ = 0;
  this->op_ = nullptr;
  this->precond_ = nullptr;
  this->SetModifiedOperator(false);
  this->SetModifiedParam(false);
  this->SetState(false);
  this->reuse_ = false;

  this->max_coarse_size_ = 9;
  this->min_coarse_size_ = 1;
  this->max_levels_ = 25;
  this->agg_num_levels_ = 0;
  this->agg_interp_type_ = 4;
  this->coarsen_type_ = 6;
  this->num_functions_ = 1;
  this->dof_func_.clear();
  this->dof_func_temp_ = nullptr;
  this->strong_threshold_ = 0.25;
  this->cycle_type_ = 1;
  this->nodal_ = 0;
  this->nodal_diag_ = 0;
  this->interp_type_ = 0;
  this->trunc_factor_ = 0.;
  this->P_max_elem_ = 0;
  this->sep_weight_ = 0;
  this->relax_weight_ = 1.;
  this->schwarz_relax_weight_ = 1.;
  this->omega_ = 1.;
  this->cycle_num_sweeps_.clear();
  this->cycle_relax_type_.clear();

  this->smooth_type_ = 6;
  this->variant_ = 0;
  this->overlap_ = 1;
  this->domain_type_ = 2;
  this->smooth_num_levels_ = 0;
  this->smooth_num_sweeps_ = 1;
  this->use_nonsymm_ = 0;
  this->sym_ = 0;
  this->level_ = 1;
  this->threshold_ = 0.1;
  this->filter_ = 0.05;
  this->drop_tol_ = 0.0001;
  this->max_nz_per_row_ = 20;
  this->euclidfile_ = nullptr;
  this->eu_level_ = 1;
  this->eu_bj_ = 0;
  this->print_level_ = 0;
  this->keepTranspose_ = 0;
  this->amg_logging_ = 0;
  this->name_ = "AMG";
}

template < class LAD > void HypreBoomerAMG< LAD >::Init() {
#ifdef WITH_HYPRE
  if (!this->initialized_) {
    HYPRE_BoomerAMGCreate(&(this->solver_));
    this->initialized_ = true;
  }

  HYPRE_BoomerAMGSetPrintLevel(this->solver_, this->print_level_);
  HYPRE_BoomerAMGSetMaxIter(this->solver_, this->maxits_); /* max iterations */
  if (this->print_level_ > 2) {
    LOG_INFO("Maximum iterations", this->maxits_);
  }
  HYPRE_BoomerAMGSetTol(this->solver_,
                        this->reltol_); /* relative conv. tolerance */
  if (this->print_level_ > 2) {
    LOG_INFO("Relative tolerance [convergence]", this->reltol_);
  }
  HYPRE_BoomerAMGSetLogging(
      this->solver_, this->amg_logging_); /* needed to get run info later */

  HYPRE_BoomerAMGSetMaxCoarseSize(this->solver_, this->max_coarse_size_);
  if (this->print_level_ > 2) {
    LOG_INFO("Maximum size of coarse grid problem", this->max_coarse_size_);
  }
  HYPRE_BoomerAMGSetMinCoarseSize(this->solver_, this->min_coarse_size_);
  if (this->print_level_ > 2) {
    LOG_INFO("Minimum size of coarse grid problem", this->min_coarse_size_);
  }
  HYPRE_BoomerAMGSetMaxLevels(this->solver_, this->max_levels_);
  if (this->print_level_ > 2) {
    LOG_INFO("Maximum number of multigrid levels", this->max_levels_);
  }
  HYPRE_BoomerAMGSetAggNumLevels(this->solver_, this->agg_num_levels_);
  if (this->print_level_ > 2) {
    LOG_INFO("Levels of aggressive coarsening", this->agg_num_levels_);
  }
  HYPRE_BoomerAMGSetAggInterpType(this->solver_, this->agg_interp_type_);
  if (this->print_level_ > 2) {
    LOG_INFO("Interpolation on levels of aggressive coarsening",
             this->agg_interp_type_);
  }
  HYPRE_BoomerAMGSetCoarsenType(this->solver_, this->coarsen_type_);
  if (this->print_level_ > 2) {
    LOG_INFO("Coarsening type", this->coarsen_type_);
  }
  HYPRE_BoomerAMGSetNumFunctions(this->solver_, this->num_functions_);
  if (this->print_level_ > 2) {
    LOG_INFO("Number of functions", this->num_functions_);
  }
  if (!this->dof_func_.empty()) {
    // No delete needed of this pointer as it is done on destruction
    // of the BoomerAMG solver object
    this->dof_func_temp_ = new HYPRE_Int[this->dof_func_.size()];
    for (size_t i = 0; i < this->dof_func_.size(); ++i) {
      this->dof_func_temp_[i] = static_cast< HYPRE_Int >(this->dof_func_[i]);
    }
    HYPRE_BoomerAMGSetDofFunc(this->solver_, this->dof_func_temp_);
  }
  HYPRE_BoomerAMGSetStrongThreshold(this->solver_, this->strong_threshold_);
  if (this->print_level_ > 2) {
    LOG_INFO("Strong threshold", this->strong_threshold_);
  }
  HYPRE_BoomerAMGSetCycleType(this->solver_, this->cycle_type_);
  if (this->print_level_ > 2) {
    LOG_INFO("Multigrid cycle type", this->cycle_type_);
  }
  HYPRE_BoomerAMGSetNodal(this->solver_, this->nodal_);
  if (this->print_level_ > 2) {
    LOG_INFO("Nodal", this->nodal_);
  }
  HYPRE_BoomerAMGSetNodalDiag(this->solver_, this->nodal_diag_);
  if (this->print_level_ > 2) {
    LOG_INFO("Nodal diagonal", this->nodal_diag_);
  }
  HYPRE_BoomerAMGSetInterpType(this->solver_, this->interp_type_);
  if (this->print_level_ > 2) {
    LOG_INFO("Interpolation type", this->interp_type_);
  }
  HYPRE_BoomerAMGSetTruncFactor(this->solver_, this->trunc_factor_);
  HYPRE_BoomerAMGSetAggTruncFactor(this->solver_, this->trunc_factor_);
  if (this->print_level_ > 2) {
    LOG_INFO("Truncation factor", this->trunc_factor_);
  }
  HYPRE_BoomerAMGSetPMaxElmts(this->solver_, this->P_max_elem_);
  HYPRE_BoomerAMGSetAggPMaxElmts(this->solver_, this->P_max_elem_);
  if (this->print_level_ > 2) {
    LOG_INFO("Maximum number of elements per row", this->P_max_elem_);
  }
  HYPRE_BoomerAMGSetSepWeight(this->solver_, this->sep_weight_);
  if (this->print_level_ > 2) {
    LOG_INFO("Separation weight", this->sep_weight_);
  }
  HYPRE_BoomerAMGSetRelaxWt(this->solver_, this->relax_weight_);
  if (this->print_level_ > 2) {
    LOG_INFO("Relaxation weight", this->relax_weight_);
  }
  HYPRE_BoomerAMGSetSchwarzRlxWeight(this->solver_,
                                     this->schwarz_relax_weight_);
  if (this->print_level_ > 2) {
    LOG_INFO("Schwarz Relaxation weight", this->schwarz_relax_weight_);
  }
  HYPRE_BoomerAMGSetOuterWt(this->solver_, this->omega_);
  if (this->print_level_ > 2) {
    LOG_INFO("Outer relaxation weight", this->omega_);
  }
  if (!this->cycle_num_sweeps_.empty()) {
    for (std::map< int, int >::const_iterator
             it = this->cycle_num_sweeps_.begin(),
             e_it = this->cycle_num_sweeps_.end();
         it != e_it; ++it) {
      HYPRE_BoomerAMGSetCycleNumSweeps(this->solver_, it->second, it->first);
    }
  }
  if (!this->cycle_relax_type_.empty()) {
    for (std::map< int, int >::const_iterator
             it = this->cycle_relax_type_.begin(),
             e_it = this->cycle_relax_type_.end();
         it != e_it; ++it) {
      HYPRE_BoomerAMGSetCycleRelaxType(this->solver_, it->second, it->first);
    }
  }
  if (this->smooth_num_levels_ > 0) {
    HYPRE_BoomerAMGSetSmoothType(this->solver_, this->smooth_type_);
    HYPRE_BoomerAMGSetSmoothNumLevels(this->solver_, this->smooth_num_levels_);
    HYPRE_BoomerAMGSetSmoothNumSweeps(this->solver_, this->smooth_num_sweeps_);

    // ParaSails
    if (this->smooth_type_ == 8) {
      HYPRE_BoomerAMGSetSym(this->solver_, this->sym_);
      HYPRE_BoomerAMGSetLevel(this->solver_, this->level_);
      HYPRE_BoomerAMGSetThreshold(this->solver_, this->threshold_);
      HYPRE_BoomerAMGSetFilter(this->solver_, this->filter_);
    }
    // PILUT
    else if (this->smooth_type_ == 7) {
      HYPRE_BoomerAMGSetDropTol(this->solver_, this->drop_tol_);
      HYPRE_BoomerAMGSetMaxNzPerRow(this->solver_, this->max_nz_per_row_);
    }
    // Euclid
    else if (this->smooth_type_ == 9) {
      if (this->euclidfile_ != nullptr) {
        HYPRE_BoomerAMGSetEuclidFile(this->solver_, this->euclidfile_);
      } else {
        HYPRE_BoomerAMGSetEuLevel(this->solver_, this->eu_level_);
        HYPRE_BoomerAMGSetEuSparseA(this->solver_, this->eu_sparse_A_);
        HYPRE_BoomerAMGSetEuBJ(this->solver_, this->eu_bj_);
      }
      // Schwarz
    } else {
      if (this->print_level_ > 2) {
        LOG_INFO("Using Schwarz smoother", true);
      }
      HYPRE_BoomerAMGSetVariant(this->solver_, this->variant_);
      if (this->print_level_ > 2) {
        LOG_INFO("Schwarz variant", this->variant_);
      }
      HYPRE_BoomerAMGSetOverlap(this->solver_, this->overlap_);
      if (this->print_level_ > 2) {
        LOG_INFO("Schwarz overlap", this->overlap_);
      }
      HYPRE_BoomerAMGSetDomainType(this->solver_, this->domain_type_);
      if (this->print_level_ > 2) {
        LOG_INFO("Schwarz domain_type", this->domain_type_);
      }
      HYPRE_BoomerAMGSetSchwarzUseNonSymm(this->solver_, this->use_nonsymm_);
      if (this->print_level_ > 2) {
        LOG_INFO("Schwarz non-symmetric matrix", this->use_nonsymm_);
      }
    }
  }

  HYPRE_BoomerAMGSetKeepTranspose(this->solver_, this->keepTranspose_);
  if (this->print_level_ > 2) {
    LOG_INFO("Keep interpolation transposes", this->keepTranspose_);
  }

  this->SetModifiedParam(true);
  this->SetState(false);
#else
  LOG_ERROR("HiFlow was not compiled with HYPRE support!");
  quit_program();
#endif
}

template < class LAD > HypreBoomerAMG< LAD >::~HypreBoomerAMG() {
  this->Clear();
}

template < class LAD >
void HypreBoomerAMG< LAD >::SetPreconditioningParameters() {
  this->reltol_ = 0.;
  this->maxits_ = 1;
}

#ifdef WITH_HYPRE
template < class LAD >
void HypreBoomerAMG< LAD >::BuildImpl(VectorType const *b, VectorType *x) {
  assert(this->op_ != nullptr);

  // Create dummy vector, only needed for interface;
  HYPRE_ParVector tmp = nullptr;
  this->Init();

  HYPRE_BoomerAMGSetup(this->solver_, *(this->op_->GetParCSRMatrix()), tmp, tmp);

  this->SetModifiedParam(false);
}
#endif

template < class LAD >
LinearSolverState HypreBoomerAMG< LAD >::SolveImpl(const VectorType &b,
                                                   VectorType *x) {
#ifdef WITH_HYPRE
  if (this->print_level_ > 1) {
    LOG_INFO(this->name_, "starts with residual norm " << this->res_);
  }
  HYPRE_BoomerAMGSolve(this->solver_, *(this->op_->GetParCSRMatrix()),
                       *(b.GetParVector()), *(x->GetParVector()));
  HYPRE_Int iter_temp;
  HYPRE_Real res_temp, res_rel_temp;
  HYPRE_BoomerAMGGetNumIterations(this->solver_, &(iter_temp));
  HYPRE_BoomerAMGGetFinalRelativeResidualNorm(this->solver_, &(res_temp));
  HYPRE_BoomerAMGGetFinalRelativeResidualNorm(this->solver_, &(res_rel_temp));
  this->iter_ = static_cast< int >(iter_temp);
  this->res_ = static_cast< DataType >(res_temp);
  this->res_rel_ = static_cast< DataType >(res_rel_temp);

  if (this->print_level_ > 1) {
    LOG_INFO(this->name_, "final iterations   = " << this->iter_);
    LOG_INFO(this->name_, "final abs res norm = " << this->res_);
    LOG_INFO(this->name_, "final rel res norm = " << this->res_rel_);
  }

  return kSolverSuccess;
#else
  LOG_ERROR("HiFlow was not compiled with HYPRE support!");
  quit_program();
#endif
}

template < class LAD > void HypreBoomerAMG< LAD >::Clear() {
#ifdef WITH_HYPRE
  if (this->initialized_) {
    if (this->print_level_ >= 2) {
      LOG_INFO("destroy AMG object ", this);
    }
    HYPRE_BoomerAMGDestroy(this->solver_);
    HypreLinearSolver< LAD >::Clear();
  }

  this->cycle_num_sweeps_.clear();
  this->cycle_relax_type_.clear();
#else
  LOG_ERROR("HiFlow was not compiled with HYPRE support!");
  quit_program();
#endif
}

template class HypreBoomerAMG< LADescriptorHypreD >;

template < class LAD >
void setup_AMG_solver(HypreBoomerAMG< LAD > &solver, const PropertyTree &params,
                      const std::vector< int > &dof_func) {
  int print_level = params["PrintLevel"].get< int >(0);
  if (print_level >= 2) {
    LOG_INFO("setup AMG object ", &solver);
  }

#ifdef WITH_HYPRE
  HypreBoomerAMG< LADescriptorHypreD > *amg_solver =
      dynamic_cast< HypreBoomerAMG< LADescriptorHypreD > * >(&solver);
  assert(amg_solver != 0);

  amg_solver->Clear();
  amg_solver->SetPrintLevel(print_level);
  amg_solver->SetPreconditioningParameters();

  amg_solver->SetNumFunctions(params["NumFunctions"].get< int >(1));
  amg_solver->SetCycleType(params["CycleType"].get< int >(1));
  amg_solver->InitControl(params["MaxIterations"].get< int >(1), 0, 0);
  if (params.contains("RelaxTypeDown")) {
    amg_solver->SetCycleRelaxType(params["RelaxTypeDown"].get< int >(0), 1);
  } else {
    amg_solver->SetCycleRelaxType(params["RelaxType"].get< int >(0), 1);
  }
  if (params.contains("RelaxTypeUp")) {
    amg_solver->SetCycleRelaxType(params["RelaxTypeUp"].get< int >(0), 2);
  } else {
    amg_solver->SetCycleRelaxType(params["RelaxType"].get< int >(0), 2);
  }
  amg_solver->SetLogging(params["Logging"].get< int >(0));
  amg_solver->SetRelaxWt(params["RelaxWeight"].get< double >(0.5));
  amg_solver->SetInterpType(params["InterpolationType"].get< int >(4));
  amg_solver->SetStrongThreshold(params["StrongThreshold"].get< double >(0.55));
  amg_solver->SetAggInterpType(params["AggInterpType"].get< int >(4));
  amg_solver->SetAggNumLevels(params["AggNumLevels"].get< int >(0));
  amg_solver->SetCoarsenType(params["CoarsenType"].get< int >(6));
  amg_solver->SetCycleNumSweeps(params["NumDownSweeps"].get< int >(1), 1);
  amg_solver->SetCycleNumSweeps(params["NumUpSweeps"].get< int >(1), 2);
  amg_solver->SetSmoothType(params["SmoothType"].get< int >(0));
  amg_solver->SetSmoothNumLevels(params["SmoothNumLevels"].get< int >(-1));
  amg_solver->SetMaxCoarseSize(params["MaxCoarseSize"].get< int >(5));
  amg_solver->SetMaxLevels(params["MaxLevels"].get< int >(10));
  amg_solver->SetCycleRelaxType(params["CoarseSolver"].get< int >(9), 3);
  amg_solver->SetCycleNumSweeps(params["CoarseSweeps"].get< int >(1), 3);
  amg_solver->SetVariant(params["SchwarzVariant"].get< int >(3));
  amg_solver->SetOverlap(params["SchwarzOverlap"].get< int >(1));
  amg_solver->SetDomainType(params["SchwarzDomainType"].get< int >(1));
  amg_solver->SetSchwarzUseNonSymm(params["SchwarzUseNonSymm"].get< int >(0));
  
  //amg_solver->SetNodal(params["Nodal"].get< int >(1));
  //amg_solver->SetNodalDiag(1);
  
  amg_solver->SetDofFunc(dof_func);

  if (params["Reuse"].get< bool >(false)) {
    amg_solver->SetReuse(true);
  } else {
    amg_solver->SetReuse(false);
  }

  amg_solver->Init();
#endif
}

template void
setup_AMG_solver< LADescriptorHypreD >(HypreBoomerAMG< LADescriptorHypreD > &,
                                       const PropertyTree &,
                                       const std::vector< int > &);

} // namespace la
} // namespace hiflow
