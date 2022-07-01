#ifndef HIFLOW_LINEARSOLVER_GMG_H_
#define HIFLOW_LINEARSOLVER_GMG_H_

#include "common/timer.h"
#include "linear_algebra/vector.h"
#include "linear_algebra/coupled_vector.h"
#include "linear_solver/linear_solver.h"
#include "linear_solver/linear_solver_creator.h"
#include "linear_solver/linear_solver_factory.h"
#include "linear_solver/richardson.h"
#include "linear_solver/preconditioner_bjacobi_standard.h"
#include "linear_solver/preconditioner_bjacobi_ext.h"
#include "linear_solver/preconditioner_vanka.h"
#include "space/vector_space.h"
#include "space/fe_interpolation_map.h"
#include "assembly/assembly_assistant.h"
#include "space/fe_evaluation.h"
#include <cmath>
#include <string>
#include <vector>


namespace hiflow {
namespace la {

/// @brief GMG
///
/// Implementation of the geometric multigrid method 
/// 

enum class GMGCycleType 
{
  V = 0,
  W = 1,
  F = 2
};

template < class Application, class Res, class Pro, class LAD, class cLAD, int DIM >
class GMGBase : public LinearSolver< LAD, LAD > 
{
public:
  using OperatorType = typename LAD::MatrixType;
  using VectorType   = typename LAD::VectorType;
  using DataType     = typename LAD::DataType  ;
  using CoarseOperatorType = typename cLAD::MatrixType;
    
  GMGBase()
  {
    this->name_ = "GMG";
  }
  
  virtual ~GMGBase() 
  {
    this->Clear();
  }
  
  virtual void Clear();
    
  inline void set_application (Application * ap) 
  {
    assert(ap!= nullptr);
    this->ap_= ap;
    this->SetModifiedOperator(true);
  }
  
  void set_spaces(const std::vector < VectorSpace <DataType, DIM >*  >& spaces) 
  {
    this->nb_lvl_ = spaces.size();
    const size_t nb_fe = spaces[0]->nb_fe();
    
    this->spaces_.resize(this->nb_lvl_);
    for (int i = 0; i< this->nb_lvl_; ++i) 
    {
      assert(spaces[i]!=nullptr);
      assert(nb_fe == spaces[i]->nb_fe());
      
      this->spaces_[i] = spaces[i];
    }
    this->initialized_LA_ = false;
    this->initialized_R_ = false;
    this->initialized_P_ = false;
    this->SetModifiedOperator(true);
  }
  
  // TODO: if this function is called, then no restriction operator should be
  // setup in InitRestriction
  void set_restriction(const std::vector <Res *>& R, bool is_initialized = false) 
  {
    assert (R.size() == (this->nb_lvl_ - 1));
    
    this->R_.resize(R.size());
    for (int i = 0; i< R.size(); ++i) 
    {
      assert(R[i]!=nullptr);
      this->R_[i] = R[i];
    }

    this->initialized_R_ = is_initialized;
    this->SetState(false);
  }
  
   // TODO: if this function is called, then no restriction operator should be
  // setup in InitProlongation
  void set_prolongation(const std::vector < Pro *>& P, bool is_initialized = false) 
  {
    assert (P.size() == (this->nb_lvl_ - 1));
  
    this->P_.resize(P.size());
    for (int i = 0; i < P.size(); ++i) 
    {
      assert(P[i]!=nullptr);
      this->P_[i] = P[i];
    }
    this->initialized_P_ = is_initialized;
    this->SetState(false);
  }
  
  void set_smoothers(const std::vector <Preconditioner<LAD>* >& S) 
  {
    assert (S.size() == this->nb_lvl_ - 1);
    
    this->preS_.resize(S.size()+1, nullptr);
    for (int i = 0; i< S.size(); ++i) 
    {
      assert(S[i]!=nullptr);
      this->preS_[i+1] = S[i];
    }
    this->SetState(false);
  }
  
  inline void set_coarse_solver(LinearSolver<cLAD>* S) 
  {
    assert(S != nullptr);
    this->cS_ = S;
    this->SetState(false);
  }
  
  void set_operators(const std::vector<OperatorType*>& A,
                     CoarseOperatorType* cA) 
  {    
    assert (A.size() == this->nb_lvl_-1);
    
    //flag
    op_manually_set_ = true;
    
    this->A_.resize(A.size()+1, nullptr);
    for (int i = 0; i< A.size(); ++i) 
    {
      assert(A[i]!=nullptr);
      this->A_[i+1] = A[i];
    }
    
    assert (cA != nullptr);
    this->cA_ = cA;

    this->SetModifiedOperator(true);
  }

  void set_rhs (const std::vector<VectorType*>& b) 
  {
    assert (b.size() == this->nb_lvl_);
          
    //flag
    rhs_manually_set_ = true;
    this->b_.resize(b.size());
    for (int i = 0; i< b.size(); ++i) 
    {
      assert(b[i]!=nullptr);
      this->b_[i] = b[i];
    }
  }
  
  virtual void InitParameter(std::string cycle_type,
                             bool nested_iteration,
                             int pre_smooth_it,
                             int post_smooth_it,
                             DataType relax_omega,
                             bool use_approx_defect_correction,
                             bool use_transpose_prolongation,
                             bool update_bc,
                             bool interpolate_rhs)
  {
    this->iterative_initial_ = nested_iteration;
   
    assert (pre_smooth_it >= 0);
    assert (post_smooth_it >= 0);
    
    this->pre_it_= pre_smooth_it;
    this->post_it_ = post_smooth_it;
  
    assert (relax_omega > 0.);
    assert (relax_omega <= 1.);
    this->omega_ = relax_omega;
  
    assert(cycle_type == "V" || cycle_type == "W" || cycle_type == "F");
    if (cycle_type == "V")
    {
      this->cycle_type_ = GMGCycleType::V;
    }
    else if (cycle_type == "W")
    {
      this->cycle_type_ = GMGCycleType::W;
    }
    else if (cycle_type == "F")
    {
      this->cycle_type_ = GMGCycleType::F;
    }
    
    this->use_approx_defect_correction_ = use_approx_defect_correction;
    this->use_transpose_prolongation_ = use_transpose_prolongation;
    this->update_bc_ = update_bc;
    this->interpolate_rhs_ = interpolate_rhs;
  }
  
  void InitStructure(VectorType const *b, VectorType *x);

  virtual void InitLA(VectorType const *b, 
                      VectorType *x);
  
  virtual void InitRestriction();

  virtual void InitOperators();

  void ResetFixedDofs();

  void Restrict  (int level, const VectorType* in, VectorType* out);
  
  void Prolongate(int level, const VectorType* in, VectorType* out);
  
  
  inline DataType get_operator_duration() 
  {
    return this->timer_operator_.get_duration();
  }
  
  inline DataType get_rhs_duration() 
  {
    return this->timer_rhs_.get_duration();
  }
  
  inline DataType get_respro_duration() 
  {
    return this->timer_respro_.get_duration();
  }
  
  inline void reinit_LA()
  {
    this->initialized_LA_ = false;
  }

  inline void reinit_RP()
  {
    this->initialized_R_ = false;
    this->initialized_P_ = false;
  }

  inline void reinit_OP()
  {
    this->initialized_OP_ = false;
  }

  inline void reinit_all() 
  {
    this->reinit_LA();
    this->reinit_RP();
    this->reinit_OP();
  }
  
  virtual void setup_gmg_vectors ( std::vector<VectorType*>& vec);
    
  virtual void GetStatistics(int &acc_iter, int &num_build, int &num_solve,
                             DataType &time_build, DataType &time_solve,
                             DataType &time_res, DataType& time_pro,
                             bool erase = false) 
  {
    Preconditioner<LAD>::GetStatistics(acc_iter, num_build, num_solve,
                                       time_build, time_solve, erase);
    time_res = this->time_res_;
    time_pro = this->time_pro_;
    if (erase) 
    {
      this->time_res_ = 0.;
      this->time_pro_ = 0.;
    }
  }

  inline LinearSolver<cLAD>* coarse_solver() 
  {
    return this->cS_;
  }

  inline const LinearSolver<cLAD>* coarse_solver() const 
  {
    return this->cS_;
  }

  inline Preconditioner<LAD> * smoother(int level) 
  {
    assert (level >= 1);
    assert (level < this->nb_lvl_);
    return this->preS_[level];
  }

  inline const Preconditioner<LAD> * smoother(int level) const
  {
    assert (level >= 1);
    assert (level < this->nb_lvl_);
    return this->preS_[level];
  }

protected:
  
  LinearSolverState SolveImpl(const VectorType &b, 
                              VectorType *x); 
                                                                           
  virtual void BuildImpl(VectorType const *b, 
                         VectorType *x);
  
  virtual void UpdateOperators(); 
  
  virtual void UpdateFixedDofs();
    
  virtual void ApplyFixedDofs(int level, bool zeros, VectorType *x) const;
  
  virtual void BuildRhsLvl(const VectorType &b);
  
  bool op_manually_set_ = false;
  bool rhs_manually_set_ = false;
  bool iterative_initial_ = true;
  bool initialized_LA_ = false;
  bool initialized_R_ = false;
  bool initialized_P_ = false;
  bool initialized_OP_ = false;
  bool use_approx_defect_correction_ = false;
  bool use_transpose_prolongation_ = true;
  bool update_bc_ = true;
  bool interpolate_rhs_ = true;
  
  Application * ap_ = nullptr;
  std::vector < VectorSpace <DataType , DIM >*  > spaces_;
  
  // smoothers and coarse solver
  std::vector < Richardson<LAD>* > S_;
  std::vector < Preconditioner<LAD> * > preS_;
  LinearSolver<cLAD>* cS_ = nullptr;

  //Restriction Operator between two successive levels
  std::vector < Res* > R_;
  //Prolongation Operator between two successive levels
  std::vector < Pro* > P_;

  //coefficient matrices of linear system on each level
  std::vector < OperatorType *> A_;
  CoarseOperatorType* cA_ = nullptr;

  //right-hand side on each level
  std::vector < VectorType *> b_;
  
  //defect on each level
  std::vector < VectorType *> d_;
  //error on each level
  std::vector < VectorType *> e_;
  //solution on each level
  std::vector < VectorType *> x_;
  
  // residual on each level
  std::vector < VectorType *> r_;

  std::vector < VectorType*> tmp_;
  std::vector < VectorType*> zeros_;
  
  // dirichlet BC
  std::vector< std::vector< DataType> > fixed_vals_;
  std::vector< std::vector< DataType> > fixed_zeros_;
  std::vector< std::vector< int > > fixed_dofs_;
  
  //iterations of the smoother at the beginning and at the end of the scheme
  int pre_it_ = 1;
  int post_it_ = 1;
  GMGCycleType cycle_type_ = GMGCycleType::V;
  DataType omega_ = 1.;
  
  Timer timer_operator_;
  Timer timer_rhs_;
  Timer timer_respro_;
  Timer timer_res_;
  Timer timer_pro_;

  int nb_lvl_ = -1;
  DataType time_res_ = 0.;
  DataType time_pro_ = 0.;

private: 
  //Smoothers on each level
  LinearSolverState SolveImplLevel(int level, 
                                   const VectorType &b, 
                                   const VectorType &x, 
                                   VectorType *out);
 
  LinearSolverState NestedIteration(int level, const VectorType & b);
};

template <class Application, class LAD, int DIM >     
class GMGStandard : public GMGBase<Application, FeInterMapFullNodal< LAD, DIM >,
                                    FeInterMapFullNodal< LAD, DIM >, LAD, LAD, DIM > {
public:
  
  typedef typename LAD::MatrixType OperatorType;
  typedef typename LAD::VectorType VectorType;
  typedef typename LAD::DataType DataType;
  typedef GMGBase< Application, FeInterMapFullNodal< LAD, DIM >,
                   FeInterMapFullNodal< LAD, DIM >, LAD, LAD, DIM > GMGBASE;

  GMGStandard() 
  : GMGBASE()
  {
  }
  
  ~GMGStandard()
  {
    this->Clear();
  }

  void set_operators(const std::vector<OperatorType*>& A) 
  {    
    assert (A.size() == this->nb_lvl_);
    
    //flag
    this->op_manually_set_ = true;
    
    this->A_.resize(A.size(), nullptr);
    for (int i = 0; i< A.size(); ++i) 
    {
      assert(A[i] !=nullptr );
      this->A_[i] = A[i];
    }
    this->cA_ = A[0];

    this->SetModifiedOperator(true);
  }

  void set_restriction(std::vector <FeInterMapFullNodal< LAD, DIM > *>& R) 
  {
    LOG_ERROR("Invalid call for this GMG implementation");
    quit_program();
  }
  
  void set_prolongation(std::vector < FeInterMapFullNodal< LAD, DIM > *>& P) 
  {
    LOG_ERROR("Invalid call for this GMG implementation");
    quit_program();
  }
  
  void Clear() 
  {
    for (int i = 0; i < this->Res_.size(); ++ i) 
    {
      if (this->Pro_[i] != nullptr) 
      {  
        delete this->Pro_[i];
        this->Pro_[i] = nullptr;
      }
      if (this->Res_[i] != nullptr) 
      {  
        delete this->Res_[i];
        this->Res_[i] = nullptr;
      }
    }  
    GMGBASE::Clear();
  }

  void InitRestriction() override
  { 
    if (this->initialized_R_ && this->initialized_P_)
    {
      return;
    }
    
    LOG_INFO("GMG", "Init restriction");
    assert (this->spaces_.size() > 1);
    
    /// init Restriction and Prolongation Operators
    for (int i=0; i<this->Res_.size(); ++i)
    {
      if (this->Res_[i] != nullptr)
      {
        delete this->Res_[i];
      }
      if (this->Pro_[i] != nullptr)
      {
        delete this->Pro_[i];
      }
    }
    this->Res_.clear();
    this->Pro_.clear();
    
    this->Res_.resize(this->spaces_.size()-1, nullptr);
    this->Pro_.resize(this->spaces_.size()-1, nullptr);   

    std::vector<size_t> all_fe_inds;
    number_range<size_t>(0, 1, this->spaces_[0]->nb_fe(), all_fe_inds);

    for (int i= 0; i < Res_.size(); i++) 
    {
      this->Pro_[i] = new FeInterMapFullNodal<LAD, DIM > ();
      this->Pro_[i]->init(this->spaces_[i], this->spaces_[i+1], false, all_fe_inds, all_fe_inds);

      if (!this->use_transpose_prolongation_)
      {
        this->Res_[i] = new FeInterMapFullNodal<LAD, DIM > ();
        this->Res_[i]->init(this->spaces_[i+1], this->spaces_[i], false, all_fe_inds, all_fe_inds);
      }
    }
  
    GMGBASE::set_prolongation(Pro_, true);
    if (!this->use_transpose_prolongation_)
    {
      GMGBASE::set_restriction(Res_, true);
    }
    GMGBASE::InitRestriction();
  }

protected :

  std::vector < FeInterMapFullNodal< LAD, DIM >* >  Res_;
  std::vector < FeInterMapFullNodal< LAD, DIM >* >  Pro_;

};

template < class Application, class Res, class Pro, class LAD, class cLAD, int DIM >
void GMGBase<Application, Res, Pro, LAD, cLAD, DIM>::setup_gmg_vectors ( std::vector<VectorType*>& vec)
{
  const int num_vec = vec.size();
  for (int i=0; i<num_vec-1; ++i)
  {
    if (vec[i] != nullptr)
    {
      delete vec[i];
    }
  }
  vec.clear();
  vec.resize(this->nb_lvl_, nullptr);
  vec.resize(this->nb_lvl_, nullptr);
  
  for (int i=0; i<this->nb_lvl_; ++i)
  {
    assert (this->spaces_[i] != nullptr);
    
    vec[i] = new VectorType;
    this->ap_->prepare_rhs(i, this->spaces_[i], vec[i]); 
  }
}

template < class Application, class Res, class Pro, class LAD, class cLAD, int DIM >
void GMGBase<Application, Res, Pro, LAD, cLAD, DIM>::Clear() 
{
  LinearSolver<LAD, LAD>::Clear();
  
  for (int i = 0; i < this->nb_lvl_; ++i) 
  {
    if (this->S_[i] != nullptr) 
    {  
      delete this->S_[i];
      this->S_[i] = nullptr;
    }
    if (this->x_[i] != nullptr) 
    {  
      delete this->x_[i];
      this->x_[i] = nullptr;
    }
    if (this->e_[i] != nullptr) 
    {  
      delete this->e_[i];
      this->e_[i] = nullptr;
    }
    if (this->d_[i] != nullptr) 
    {  
      delete this->d_[i];
      this->d_[i] = nullptr;
    }
    if (this->r_[i] != nullptr) 
    {  
      delete this->r_[i];
      this->r_[i] = nullptr;
    }
    if (this->tmp_[i]!= nullptr) 
    {
      delete this->tmp_[i];
      this->tmp_[i] = nullptr;
    }
    if (this->zeros_[i] != nullptr) 
    {  
      delete this->zeros_[i];
      this->zeros_[i] = nullptr;
    }
    if(!op_manually_set_) 
    {
      if (this->A_[i] != nullptr) 
      {
        if (i > 0 && i < this->nb_lvl_ -1)
        {
          delete this->A_[i];
          this->A_[i] = nullptr;
        }
      }
      if (this->cA_ != nullptr)
      {
        delete this->cA_;
        this->cA_ = nullptr;
        this->A_[0] = nullptr;
      }
    }
    if (!rhs_manually_set_) 
    {
      if (this->b_[i] != nullptr) 
      {
        delete this->b_[i];
        this->b_[i] = nullptr;
      }
    }
  }
  this->initialized_LA_ = false;
  this->initialized_OP_ = false;
}

template < class Application, class Res, class Pro, class LAD, class cLAD, int DIM >
void GMGBase<Application, Res, Pro, LAD, cLAD, DIM>::BuildImpl(VectorType const *b, VectorType *x) 
{
  LOG_INFO("GMG", "Build");

  // space dependent setup
  this->InitLA(b,x);

  this->InitOperators();
  
  this->InitRestriction();
    
  // problem dependent setup
  this->UpdateFixedDofs();
  
  this->UpdateOperators();
  
  // pass operator to coarse solver
  assert(this->cA_ != nullptr);
  //this->cS_->SetPrintLevel(2);
  this->cS_->SetupOperator(*(this->cA_));
  this->cS_->Build(this->b_[0],this->x_[0]);
  
  // setup smoothers  
  for (int i = 1; i < this->nb_lvl_; ++i) 
  {
    assert (this->A_[i]!=nullptr);
    assert (this->preS_[i] != nullptr);
    assert (this->S_[i] != nullptr);
    assert (this->b_[i] != nullptr);
    assert (this->x_[i] != nullptr);
    
    //this->S_[i]->SetPrintLevel(2);
    this->S_[i]->SetupOperator(*(this->A_[i]));
    this->S_[i]->Build(this->b_[i], this->x_[i]);
  }
}

template < class Application, class Res, class Pro, class LAD, class cLAD, int DIM >
void GMGBase<Application, Res, Pro, LAD, cLAD, DIM>::InitStructure(VectorType const *b, VectorType *x) 
{
  assert (b != nullptr);
  assert (x != nullptr);
  
  this->InitLA(b,x);
  this->InitOperators();
  this->InitRestriction();
}

template < class Application, class Res, class Pro, class LAD, class cLAD, int DIM >
void GMGBase<Application, Res, Pro, LAD, cLAD, DIM>::InitRestriction()
{
  if (this->initialized_R_ && this->initialized_P_)
  {
    return;
  }

  LOG_INFO("GMG", "Init restriction");

  assert (this->use_transpose_prolongation_ || (this->R_.size() == this->spaces_.size() - 1));
  assert (this->P_.size() == this->spaces_.size() - 1);
  assert (this->spaces_.size() > 1);
  std::vector <size_t> fe_inds;
    
  for (int i = 0; i < this->spaces_[0]->nb_fe(); ++i) 
  { 
    fe_inds.push_back(i);	   
  }
    
  // note: we assume that restriction and prolongation operators are already set
  this->timer_respro_.reset();
  this->timer_respro_.start();
  for (int i= 0; i < R_.size(); i++) 
  {
    assert (this->P_[i] != nullptr);
    assert (this->spaces_[i] != nullptr);
    assert (this->spaces_[i+1] != nullptr);
    
    if (!this->initialized_P_)
    {
      LOG_INFO("GMG", "Init P");
      this->P_[i]->init(this->spaces_[i], this->spaces_[i+1], true, /**this->b_[i], *this->b_[i+1], */ fe_inds, fe_inds);
      this->initialized_P_ = true;
    }

    if (!this->use_transpose_prolongation_ && !this->initialized_R_)
    {
      LOG_INFO("GMG", "Init R");
      assert (this->R_[i] != nullptr);
      this->R_[i]->init(this->spaces_[i+1], this->spaces_[i], true, /**this->b_[i+1], *this->b_[i], */ fe_inds, fe_inds );
      this->initialized_R_ = true;
    }
  }
  this->timer_respro_.stop();
}

template < class Application, class Res, class Pro, class LAD, class cLAD, int DIM >
void GMGBase<Application, Res, Pro, LAD, cLAD, DIM>::InitLA(VectorType const *b, VectorType *x) 
{
  if (this->initialized_LA_)
  {
    return;
  }
  
  LOG_INFO("GMG", "init LA");
  
  // BC
  this->fixed_dofs_.clear();
  this->fixed_dofs_.resize(this->nb_lvl_);
  this->fixed_vals_.clear();
  this->fixed_vals_.resize(this->nb_lvl_);
  this->fixed_zeros_.clear();
  this->fixed_zeros_.resize(this->nb_lvl_);
    
  /// init matrices and vectors
  this->b_.resize(this->nb_lvl_);
          
  this->timer_rhs_.reset();
  this->timer_rhs_.start();
  
  this->setup_gmg_vectors(this->b_);
  this->setup_gmg_vectors(this->r_);
  this->setup_gmg_vectors(this->x_);
  this->setup_gmg_vectors(this->d_);
  this->setup_gmg_vectors(this->tmp_);
  this->setup_gmg_vectors(this->e_);
  this->setup_gmg_vectors(this->zeros_);  
  
  this->timer_rhs_.stop();
    
  // setup smoothers
  this->S_.resize(this->nb_lvl_, nullptr);
  for (int i=0; i<this->S_.size(); ++i)
  {
    if (this->S_[i] != nullptr)
    {
      delete this->S_[i];
    }
    if (i == 0)
      continue;
      
    assert (this->preS_[i] != nullptr);
   
    this->S_[i] = new Richardson<LAD>();
    std::string method = "Preconditioning";
    this->S_[i]->InitParameter(method, this->omega_, this->use_approx_defect_correction_, true);
    this->S_[i]->SetupPreconditioner(*(this->preS_[i]));
    //this->S_[i]->SetPrintLevel(2);
  }
  
  this->initialized_LA_ = true;
}

template < class Application, class Res, class Pro, class LAD, class cLAD, int DIM >
void GMGBase<Application, Res, Pro, LAD, cLAD, DIM>::InitOperators() 
{
  if (this->initialized_OP_ || this->op_manually_set_)
  {
    return;
  }
  
  LOG_INFO("GMG", "init OP");
       
  this->timer_operator_.reset();
  this->timer_operator_.start();
  this->A_.resize(this->nb_lvl_);
      
  assert(this->ap_ != nullptr);
  for (int i = 1; i < this->nb_lvl_-1; ++i) 
  {
    if (this->A_[i] != nullptr)
      delete this->A_[i];
      
    this->A_[i] = new OperatorType();
    
    assert(this->spaces_[i] != 0);
    
    // call application to setup matrix object
    this->ap_->prepare_operator(i, this->spaces_[i], this->A_[i]);    
  }

  if (this->cA_ != nullptr)
    delete this->cA_;
      
  this->cA_ = new CoarseOperatorType();
    
  assert(this->spaces_[0] != 0);
    
  // call application to setup matrix object
  this->ap_->prepare_operator(0, this->spaces_[0], this->cA_); 

  this->timer_operator_.stop();
  this->initialized_OP_ = true;
}

template < class Application, class Res, class Pro, class LAD, class cLAD, int DIM >
void GMGBase<Application, Res, Pro, LAD, cLAD, DIM>::UpdateOperators()
{
  if(this->op_manually_set_)
  {
    return;
  }
  
  assert (this->initialized_OP_);

  LOG_INFO("GMG", "update OP");
  for (int i = this->nb_lvl_-2; i >= 1; --i) 
  {   
    this->ap_->assemble_operator(i,
                                 this->spaces_[i], 
                                 this->fixed_dofs_[i],
                                 this->fixed_vals_[i],
                                 this->A_[i]); 
  }
  this->ap_->assemble_operator(0,
                               this->spaces_[0], 
                               this->fixed_dofs_[0],
                               this->fixed_vals_[0],
                               this->cA_); 

  this->A_[this->nb_lvl_-1] = this->op_;
}

template < class Application, class Res, class Pro, class LAD, class cLAD, int DIM >
void GMGBase<Application, Res, Pro, LAD, cLAD, DIM>::UpdateFixedDofs()
{
  LOG_INFO("GMG", "update fixed dofs");
  for (int i = 0; i < this->nb_lvl_; ++i) 
  {
    if (update_bc_ || (this->fixed_dofs_[i].size() == 0))
    {
      // call application to compute Dirichlet BC
      this->ap_->prepare_fixed_dofs_gmg(i,
                                        this->spaces_[i], 
                                        this->fixed_dofs_[i],
                                        this->fixed_vals_[i]);
                                
      this->fixed_zeros_[i].clear();
      this->fixed_zeros_[i].resize(this->fixed_dofs_[i].size(), 0.);
    }
  }
}

template < class Application, class Res, class Pro, class LAD, class cLAD, int DIM >
void GMGBase<Application, Res, Pro, LAD, cLAD, DIM>::ResetFixedDofs()
{
  LOG_INFO("GMG", "reset fixed dofs");
  for (int i = 0; i < this->nb_lvl_; ++i) 
  {
    this->fixed_dofs_[i].clear();
    this->fixed_vals_[i].clear();
    this->fixed_zeros_[i].clear();
  }
}

template < class Application, class Res, class Pro, class LAD, class cLAD, int DIM >
void GMGBase<Application, Res, Pro, LAD, cLAD, DIM>::ApplyFixedDofs(int level, bool zeros, VectorType *x) const
{
  if (!(this->fixed_dofs_[level].empty())) 
  {
    if (zeros)
    {
      x->SetValues(vec2ptr(this->fixed_dofs_[level]), 
                   this->fixed_dofs_[level].size(), 
                   vec2ptr(this->fixed_vals_[level]));
    }
    else
    {
      x->SetValues(vec2ptr(this->fixed_dofs_[level]), 
                   this->fixed_dofs_[level].size(), 
                   vec2ptr(this->fixed_zeros_[level]));
    }
  }
}

template < class Application, class Res, class Pro, class LAD, class cLAD, int DIM >
void GMGBase<Application, Res, Pro, LAD, cLAD, DIM>::Restrict(int level, const VectorType* in, VectorType* out)
{
  assert (in != nullptr);
  assert (out != nullptr);
  
  //assert (!in->ContainsNaN());

  this->timer_res_.start();
  if (!this->use_transpose_prolongation_)
  {
    assert (level < this->R_.size());
    assert (this->R_[level] != nullptr);
    assert (initialized_R_);
    this->R_[level]->interpolate(*in, *out);
  }
  else
  {
    assert (level < this->P_.size());
    assert (this->P_[level] != nullptr);
    assert (initialized_P_);
    this->P_[level]->interpolate_transpose(*in, *out);
  }

  //assert (!out->ContainsNaN());

  this->timer_res_.stop();
  this->time_res_ += this->timer_res_.get_duration();
  this->timer_res_.reset();
}

template < class Application, class Res, class Pro, class LAD, class cLAD, int DIM >
void GMGBase<Application, Res, Pro, LAD, cLAD, DIM>::Prolongate(int level, const VectorType* in, VectorType* out)
{
  assert (in != nullptr);
  assert (out != nullptr);
  assert (level < this->P_.size());
  
  //assert (!in->ContainsNaN());

  this->timer_pro_.start();
  assert (this->P_[level] != nullptr); 
  assert (initialized_P_);
  this->P_[level]->interpolate(*in, *out);

  //this->R_[level]->interpolate_transpose(*in, *out);
  //assert (!out->ContainsNaN());

  this->timer_pro_.stop();
  this->time_pro_ += this->timer_pro_.get_duration();
  this->timer_pro_.reset();
}

template < class Application, class Res, class Pro, class LAD, class cLAD, int DIM >
void GMGBase<Application, Res, Pro, LAD, cLAD, DIM>::BuildRhsLvl(const VectorType &b) 
{
  int num_levels = this->nb_lvl_;
  this->b_[num_levels-1]->CopyFrom(b);
  this->b_[num_levels-1]->Update(); // TODO: needed?

  for (int i = num_levels-2; i >=0; --i) 
  {
    if (this->interpolate_rhs_)
    {
      this->Restrict(i, this->b_[i+1], this->b_[i]);
    }
    else
    {
      this->ap_->assemble_rhs(i,
                              spaces_[i], 
                              this->fixed_dofs_[i],
                              this->fixed_vals_[i],
                              b_[i]);
    }
  }
}

template < class Application, class Res, class Pro, class LAD, class cLAD, int DIM >
LinearSolverState GMGBase<Application, Res, Pro, LAD, cLAD, DIM>::SolveImpl(const VectorType &b, VectorType *x) 
{
  IterateControl::State conv = IterateControl::kIterate;
    
  int num_levels = this->nb_lvl_;
  assert (num_levels >= 2);

  this->iter_ = 0;
  
  //assert (!b.ContainsNaN());

  this->BuildRhsLvl(b);
    
  if (this->iterative_initial_)
  {
    // fill x_ vector by computing x_0 = A_0^-1 b_0
    // and prolongation to higher levels
    LinearSolverState state = NestedIteration(num_levels-1, *(b_[num_levels-1]));
    assert (state == kSolverSuccess);
  }
  else
  {
    this->x_[num_levels-1]->CopyFrom(*x);
  }
    
  // r = b_L - A_L * x_L
  this->r_[num_levels-1]->CloneFrom(*b_[num_levels-1]);
  this->A_[num_levels-1]->VectorMultAdd(-1, *(x_[num_levels-1]), 1, this->r_[num_levels-1]);
    
  this->res_init_ = b.Norm2();
  this->res_ = this->r_[num_levels-1]->Norm2();
  this->res_rel_ = this->res_ / this->res_init_;    
  conv = this->control().Check(this->iter_, this->res_);
    
  if (this->print_level_ > 1) 
  {
    LOG_INFO(this->name_, "initial res norm   =  " << this->res_);
  }
      
  if (conv != IterateControl::kIterate) 
  {
    return kSolverSuccess;
  }
      
  while (conv == IterateControl::kIterate ) 
  {
    this->iter_ += 1;
 
    LinearSolverState state = SolveImplLevel(num_levels -1 , 
                                             *(b_[num_levels -1]), 
                                             *(x_[num_levels -1]), 
                                               x_[num_levels-1]);
      
    // r_L = b_L - A_L * x_L 
    this->r_[num_levels-1]->CopyFrom(*b_[num_levels-1]);
    this->A_[num_levels-1]->VectorMultAdd(-1, *(x_[num_levels-1]), 1, this->r_[num_levels-1]);    
      
    this->res_ = this->r_[num_levels-1]->Norm2();
    this->res_rel_ = this->res_ / this->res_init_;
    
    if (this->print_level_ > 2) 
    {
      LOG_INFO(this->name_, "residual (iteration "<< this->iter_ << "): " << this->res_);
    }
      
    conv = this->control().Check(this->iter_, this->res_);
    if (conv != IterateControl::kIterate) 
    {
      break;
    }
  }
  
  if (this->print_level_ > 1) 
  {
    LOG_INFO(this->name_, "final iterations   = " << this->iter_);
    LOG_INFO(this->name_, "final abs res norm = " << this->res_);
    LOG_INFO(this->name_, "final rel res norm = " << this->res_rel_)
  } 
  
  x->CloneFrom(*x_[num_levels - 1]);
  
  if (conv == IterateControl::kFailureDivergenceTol ||
      conv == IterateControl::kFailureMaxitsExceeded) 
  {
    return kSolverExceeded;
  }
    
  return kSolverSuccess;	  
}

template < class Application, class Res, class Pro, class LAD, class cLAD, int DIM >
LinearSolverState GMGBase<Application, Res, Pro, LAD, cLAD, DIM>::SolveImplLevel(int level, 
                                                                                 const VectorType &b, 
                                                                                 const VectorType &x, 
                                                                                 VectorType *out)  
{ 
  if (this->print_level_ > 2) 
  {  
    LOG_INFO("MGV", "solve on level " << level);
  }

  assert (out != nullptr);
  assert (!(level >= 1 && A_[level]==nullptr ));
  assert (cA_ != nullptr);
  assert (d_[level]!=nullptr);
  //assert (R_[level-1]!=nullptr);
  assert (d_[level-1]!=nullptr);
  assert (e_[level-1]!=nullptr);
  assert (tmp_[level-1]!=nullptr);
  assert (S_[level]!=nullptr);
  assert (cS_!=nullptr);
  //assert (P_[level-1]!=nullptr);
    
  //this->tmp_[level]->CloneFrom(x);
  //assert (!b.ContainsNaN());
  //assert (!x.ContainsNaN());

  this->tmp_[level]->CopyFrom(x);
  this->ApplyFixedDofs(level, false, this->tmp_[level]);  // set dirichlet BC 
  
  LinearSolverState state = kSolverSuccess;
    
  // pre_it steps of pre-smoothing via preconditioned Richardson iteration
  // tmp = S^preit * x
  if (this->pre_it_ > 0)
  {
    this->S_[level]->InitControl(this->pre_it_);
    state = this->S_[level]->Solve(b, this->tmp_[level]);
    //assert (!this->tmp_[level]->ContainsNaN());   
  }

  // r_l = b_l - A_l * tmp 
  this->r_[level]->CopyFrom(b);

  if (level >= 1)
  {
    this->A_[level]->VectorMultAdd(-1, *this->tmp_[level], 1., this->r_[level]);
  }
  else 
  {
    this->cA_->VectorMultAdd(-1, *this->tmp_[level], 1., this->r_[level]);
  }
  this->ApplyFixedDofs(level, true, this->r_[level]);  // set dirichlet BC zero
  
  if (this->print_level_ > 1)
  {
    DataType pre_smooth_res = this->r_[level]->Norm2();
    LOG_INFO("MGV", "residual after pre-smoothing by testing " << pre_smooth_res);
    LOG_INFO("MGV", "residual after pre-smoothing by solver  " << dynamic_cast<LinearSolver<LAD>*>(S_[level])->res());
    LOG_INFO("MGV", "iterations of  pre-smoothing by solver  " << dynamic_cast<LinearSolver<LAD>*>(S_[level])->iter());
  }

  this->Restrict(level-1, this->r_[level], this->d_[level-1]);
  this->ApplyFixedDofs(level-1, true, this->d_[level-1]);  // set dirichlet BC 
    
  if (level > 1) 
  {
    if( this->cycle_type_ == GMGCycleType::V) 
    {
      //assert (!this->e_[level-1]->ContainsNaN());
      //assert (!this->d_[level-1]->ContainsNaN());
      //assert (!this->zeros_[level-1]->ContainsNaN());
      state = SolveImplLevel(level - 1, *(this->d_[level-1]), *(this->zeros_[level-1]),this->e_[level-1]);
      //assert (!this->e_[level-1]->ContainsNaN());
    }
    else 
    {
      VectorType eel1;
      eel1.CloneFromWithoutContent(*this->e_[level-1]);
    
      state = SolveImplLevel(level - 1, *(this->d_[level-1]), *(this->zeros_[level-1]), &eel1);
        
      state = SolveImplLevel(level - 1, *(this->d_[level-1]), eel1, this->e_[level-1]);
    }
  }
  else 
  {
    //solve directly on coarsest level
    // e = A_0^{-1} dl1
    //assert (!this->d_[0]->ContainsNaN());
    //assert (!this->e_[0]->ContainsNaN());
    //this->r_[0]->CopyFrom(*(this->d_[0]));
    //this->A_[0]->VectorMultAdd(-1, *(this->e_[0]), 1., this->r_[0]);
    //assert (!this->r_[0]->ContainsNaN());

    state = this->cS_->Solve(*(this->d_[0]), this->e_[0]);
    //assert (!this->e_[0]->ContainsNaN());

    if (this->print_level_ > 1)
    {
      LOG_INFO("MGV(" << level << ")", "coarse solved with res " << this->cS_->res());
      LOG_INFO("MGV(" << level << ")", "coarse solved with iter " << this->cS_->iter());
    }
    this->e_[level-1]->Update();
  }
  //assert (!this->e_[level-1]->ContainsNaN());
  this->ApplyFixedDofs(level-1, false, this->e_[level-1]);  // set dirichlet BC 
    
  // out = tmp + P_l-1 * e_l-1
  //assert (!this->e_[level-1]->ContainsNaN());
  this->Prolongate(level-1, this->e_[level-1], out);
  
  out->Axpy(*this->tmp_[level], 1.);
  this->ApplyFixedDofs(level, false, out);  // set dirichlet BC 
    
  // out = S^postit * out
  if (this->post_it_ > 0)
  {     
    this->S_[level]->InitControl(this->post_it_);  
    state = this->S_[level]->Solve(b, out);
    //assert (!out->ContainsNaN());
  }
  this->ApplyFixedDofs(level, false, out);  // set dirichlet BC 
  //assert (!out->ContainsNaN());
  return state;
}

template < class Application, class Res, class Pro, class LAD, class cLAD, int DIM >
LinearSolverState GMGBase<Application, Res, Pro, LAD, cLAD, DIM>::NestedIteration(int level, const VectorType & b) 
{
  LinearSolverState state = kSolverSuccess;
    
  if (level == 0) 
  {
    //solve directly on coarsest mesh
    state = this->cS_->Solve(b, this->x_[0]);
      
    if (state != kSolverSuccess)
      return state;
        
    this->x_[0]->Update();    // necessary ??
  }
  else 
  {
    NestedIteration(level - 1, *(this->b_[level-1]));
    //assert (!this->x_[level - 1]->ContainsNaN());
    this->Prolongate(level-1, this->x_[level - 1], this->x_[level]);
    
    this->x_[level]->Update();// necessary??
      
    state = SolveImplLevel(level, b, *(this->x_[level]), this->x_[level]); 
    if (state != kSolverSuccess)
      return state;     
  }
  return state;
}


} //end namespace la
} //end namespace hiflow

#endif // HIFLOW_LINEARSOLVER_GMG_H_
