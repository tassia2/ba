#ifndef HIFLOW_LINEARSOLVER_GMG_H_
#define HIFLOW_LINEARSOLVER_GMG_H_

#include "common/timer.h"
#include "linear_algebra/vector.h"
#include "linear_algebra/coupled_vector.h"
#include "linear_solver/linear_solver.h"
#include "linear_solver/linear_solver_creator.h"
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

template < class Res, class Pro, class LAD, int DIM >
class GMGBase : public LinearSolver< LAD, LAD > {
public:
  typedef typename LAD::MatrixType OperatorType;
  typedef typename LAD::VectorType VectorType;
  typedef typename LAD::DataType DataType;
    
  GMGBase()
  : cycle_type_("V"), pre_it_(5), post_it_(5)
  {
  }

  
  virtual ~GMGBase() {
    this->Clear();
  }
  
  inline void set_restriction(std::vector <Res *> R) {	
    this->R_.resize(R.size());
    for (int i = 0; i< R.size(); ++i) {
      assert(R[i]!=nullptr);
      this->R_[i] = R[i];
    }
  }
  inline void set_prolongation(std::vector < Pro *> P) {
    this->P_.resize(P.size());
    for (int i = 0; i < P.size(); ++i) {
      assert(P[i]!=nullptr);
      this->P_[i] = P[i];
    }
  }
  inline void set_smoothers(std::vector <LinearSolver<LAD, LAD>* > S) {
    this->S_.resize(S.size());
    for (int i = 0; i< S.size(); ++i) {
      assert(S[i]!=nullptr);
      this->S_[i] = S[i];
    }
  }
  inline void set_operators(std::vector<OperatorType*> A) {    
	
    this->A_.resize(A.size());
    for (int i = 0; i< A.size(); ++i) {
      assert(A[i]!=nullptr);
      this->A_[i] = A[i];
    }
  }
  inline void set_rhs (std::vector<VectorType*> b) {    
    this->b_.resize(b.size());
    for (int i = 0; i< b.size(); ++i) {
      assert(b[i]!=nullptr);
      this->b_[i] = b[i];
    }
  }
  
  inline void set_smoother_iterations(int pre_it, int post_it) {
    this->pre_it_= pre_it;
    this->post_it_ = post_it;
  }
  
  inline void set_cycle_type(std::string type) {
    assert(type == "V" || type == "W" || type == "F");
    this->cycle_type_ = type;
  }
  


  
protected:

  virtual LinearSolverState SolveImpl(const VectorType &b, VectorType *x) {
	
    int num_levels = A_.size();
    IterateControl::State conv = IterateControl::kIterate;
    
    VectorType r;
    r.CloneFromWithoutContent(*b_[num_levels-1]);
    this->A_[num_levels-1]->VectorMult(*(x_[num_levels-1]), &r);
  
    r.ScaleAdd(*b_[num_levels-1], static_cast< DataType >(-1.));   
    DataType ressquared = r.Dot(r);
    this->res_ = std::sqrt(ressquared);
    this->res_init_ = this->res_;
    this->res_rel_ = 1.;
    conv = this->control().Check(this->iter_, this->res_);
    
    if (this->print_level_ > 1) {
      LOG_INFO(this->name_, "initial res norm   =  " << this->res_);
    }
    
    if (conv != IterateControl::kIterate) {
      return kSolverSuccess;
    }
    
    
    this->iter_ += 1;
    //std::cout << "Iteration: " << iteration << std::endl;
    LinearSolverState state = Init(num_levels-1, *(b_[num_levels-1]));
    this->A_[num_levels-1]->VectorMult(*(x_[num_levels-1]), &r);
    r.Axpy(*b_[num_levels-1], static_cast< DataType >(-1.));
    ressquared = r.Dot(r);
    this->res_ = sqrt(ressquared);
    this->res_rel_ = this->res_ / this->res_init_;
    
    conv = this->control().Check(this->iter_, this->res_);
    
    if (this->print_level_ > 2) {
      LOG_INFO(this->name_,
               "residual (iteration " << this->iter_ << "): " << this->res_);
    }
    
    //std::cout << "initialized starting values !"  << std::endl;
    
    while (conv == IterateControl::kIterate) {
      this->iter_ += 1;
      //std::cout << "Iteration: " << iteration << std::endl;
    
      state = SolveImplLevel(num_levels -1 , *(b_[num_levels -1]), *(x_[num_levels -1]), x_[num_levels-1]);					
      
      this->A_[num_levels-1]->VectorMult(*(x_[num_levels-1]), &r);
      r.Axpy(*b_[num_levels-1], static_cast< DataType >(-1.));
      ressquared = r.Dot(r);
      this->res_ = sqrt(ressquared);
      this->res_rel_ = this->res_ / this->res_init_;
    
      if (this->print_level_ > 2) {
        LOG_INFO(this->name_,
               "residual (iteration " << this->iter_ << "): " << this->res_);
      }
      
      conv = this->control().Check(this->iter_, this->res_);
      if (conv != IterateControl::kIterate) {
        break;
      }
    
    }
    
    if (this->print_level_ > 1) {
      LOG_INFO(this->name_, "final iterations   = " << this->iter_);
      LOG_INFO(this->name_, "final abs res norm = " << this->res_);
      LOG_INFO(this->name_, "final rel res norm = " << this->res_rel_)
    } 
    
    x->CloneFrom(*x_[num_levels - 1]);
    
    if (conv == IterateControl::kFailureDivergenceTol ||
      conv == IterateControl::kFailureMaxitsExceeded) {
      return kSolverExceeded;
    }
    
    return kSolverSuccess;	  
  }
  
  void Clear() {
	
    LinearSolver<LAD, LAD>::Clear();
  
    for (int i = 0; i < A_.size(); ++i) {
      
      if (this->x_[i] != nullptr) {  
        delete this->x_[i];
        this->x_[i] = nullptr;
      }
      
      if (this->e_[i] != nullptr) {  
        delete this->e_[i];
        this->e_[i] = nullptr;
      }
      
      if (this->d_[i] != nullptr) {  
        delete this->d_[i];
        this->d_[i] = nullptr;
      }
      
      if (this->tmp_[i]!= nullptr) {  
        delete this->tmp_[i];
        this->tmp_[i] = nullptr;
      }
      
      if (this->zeros_[i] != nullptr) {  
        delete this->zeros_[i];
        this->zeros_[i] = nullptr;
      }
    }
    
 
  }

  
  
  virtual void BuildImpl(VectorType const *b, VectorType *x) {
    
    this->x_.resize(A_.size());   
    this->d_.resize(A_.size());
    this->e_.resize(A_.size());
    this->tmp_.resize(A_.size());
    this->zeros_.resize(A_.size());

    assert(this->b_[0]!= nullptr);
    
    for (int i = 0; i < this->A_.size(); ++i) {
      
      this->x_[i] = new VectorType();
      this->d_[i] = new VectorType();
      this->e_[i] = new VectorType();

      this->tmp_[i] = new VectorType();
      zeros_[i] = new  VectorType();

      this->x_[i]->CloneFromWithoutContent(*(this->b_[i])); 
      this->x_[i]->Zeros();
      this->d_[i]->CloneFromWithoutContent(*(this->b_[i]));  
      this->d_[i]->Zeros();
      this->e_[i]->CloneFromWithoutContent(*(this->b_[i]));  
      this->e_[i]->Zeros();
      this->tmp_[i] ->CloneFromWithoutContent(*(b_[i]));
      tmp_[i]->Zeros();
      this->zeros_[i]->CloneFromWithoutContent(*(b_[i]));
      zeros_[i]->Zeros();
      
      assert(this->A_[i]!=nullptr);
      this->S_[i]->SetupOperator(*(this->A_[i]));

    }

    
  }


private: 
  //Smoothers on each level
  std::vector<LinearSolver<LAD, LAD> *> S_;
  //Restriction Operator between two successive levels
  std::vector<Res* > R_;
  //Prolongation Operator between two successive levels
  std::vector < Pro *> P_;

  //coefficient matrices of linear system on each level
  std::vector <OperatorType *>  A_;
  //right-hand side on each level
  std::vector <VectorType *>  b_;
  
  //defect on each level
  std::vector <VectorType *>  d_;
  //error on each level
  std::vector <VectorType *>  e_;
  //solution on each level
  std::vector <VectorType *> x_;

  std::vector< VectorType*> tmp_;
  std::vector< VectorType*> zeros_;
  
  //iterations of the smoother at the beginning and at the end of the scheme
  int pre_it_;
  int post_it_;
  
  //cycle type
  std::string cycle_type_;

  LinearSolverState SolveImplLevel(int level, const VectorType &b, VectorType &x, VectorType *out)  {
	   
    assert(A_[level]!=nullptr);
    assert(d_[level]!=nullptr);
    assert(R_[level-1]!=nullptr);
    assert(d_[level-1]!=nullptr);
    assert(e_[level-1]!=nullptr);
    assert(S_[level]!=nullptr);
    assert(S_[0]!=nullptr);
    assert(P_[level-1]!=nullptr);
    
    LinearSolverState state = kSolverSuccess;
    

    out->CopyFrom(x);
    
    //set other parameters very small (resp. large) such that the number of iterations is the relevant stopping criterion
    this->S_[level]->InitControl(this->pre_it_);
    state = this->S_[level]->Solve(b,out);     //TODO: Ist ja eig keine Lsg von LGS sondern Iterationsverfahren  based on A,b,x. res in x ueberschreiben ?
    this->A_[level]->VectorMult(*out, tmp_[level]);  

    this->d_[level]->CopyFrom(b);
    this->d_[level]->Axpy(*tmp_[level],-1);
    
    this->R_[level-1]->interpolate(*(this->d_[level]), *(this->d_[level-1]));
    
    if (level > 1) {
      if( this->cycle_type_ == "V") {
        state = SolveImplLevel(level - 1, *(this->d_[level-1]), *(this->zeros_[level-1]), this->e_[level-1]);
      }
      else {
        state = SolveImplLevel(level - 1, *(this->d_[level-1]), *(this->zeros_[level-1]), this->e_[level-1]);
        state = SolveImplLevel(level - 1, *(this->d_[level-1]), *(this->e_[level-1]), this->e_[level-1]);
      }
    
    
    }
    
    else {
    
      //solve directly on coarsest level
      state = this->S_[0]->Solve(*(this->d_[level-1]), this->e_[level-1]); 
      
 
    }
    this->P_[level-1]->interpolate(*(this->e_[level-1]), *(this->tmp_[level]));

    
    out->Axpy(*(this->tmp_[level]),1);
    
    //set other parameters very small (resp. large) such that the number of iterations is the relevant stopping criterion
    this->S_[level]->InitControl(this->post_it_);
    state = this->S_[level]->Solve(b, out);

    
    return state;
  }

  
  LinearSolverState Init(int level, const VectorType & b) {
	
    LinearSolverState state = kSolverSuccess;
    
    if (level == 0) {
      //solve directly on coarsest mesh
      state = this->S_[0]->Solve(b, this->x_[0]);

    }
    else {
      
      Init(level - 1, *(this->b_[level-1]));
      this->P_[level-1]->interpolate(*(this->x_[level - 1]), *(this->x_[level]));
      state = SolveImplLevel(level, b, *(this->x_[level]), this->x_[level]);      
      
    }

    /*std::vector<int> ids;
>>>>>>> 0c56853c275dcc109bd9ea23c989a317b52fe72c
    std::vector<DataType> values;
    x_[0]->GetAllDofsAndValues(ids, values);
    
    for (int i = 0; i < values.size(); ++i) {
      std::cout << values[i] << std::endl;
    }

    std::cout << "------------------------------------------------------------------------------------------------" << std::endl;
<<<<<<< HEAD
    x_[1]->GetAllDofsAndValues(ids, values);
    
    for (int i = 0; i < values.size(); ++i) {
      std::cout << values[i] << std::endl;
    }
=======
    
    x_[1]->GetAllDofsAndValues(ids, values);
    
    for (int i = 0; i < values.size(); ++i) {
      std::cout << values[i] << std::endl;
    }*/
    return state;
    
  }
  
	
};



template <class Application, class LAD, int DIM >     
class GMGStandard : public GMGBase< FeInterMapFullNodal< typename LAD::DataType, DIM >,
                                    FeInterMapFullNodal< typename LAD::DataType, DIM >, LAD, DIM > {
public:
  
  typedef typename LAD::MatrixType OperatorType;
  typedef typename LAD::VectorType VectorType;
  typedef typename LAD::DataType DataType;
  typedef GMGBase< FeInterMapFullNodal< typename LAD::DataType, DIM >,
                   FeInterMapFullNodal< typename LAD::DataType, DIM >, LAD, DIM > GMGBASE;
                   
             

	
  GMGStandard() 
  : ap_(nullptr), GMGBASE()
  {
  }
  
  ~GMGStandard()
  {
    this->Clear();
  }
  
  inline void set_application (Application * ap) {
    assert(ap!= nullptr);
    this->ap_= ap;
  }
  
  inline void set_spaces(std::vector < VectorSpace <DataType, DIM > * > spaces) {
    size_t nb_fe = spaces[0]->nb_fe();
    this->spaces_.resize(spaces.size());
    for (int i = 0; i< spaces.size(); ++i) {
      
      assert(spaces[i]!=nullptr);
      assert(nb_fe == spaces[i]->nb_fe());
      
      this->spaces_[i] = spaces[i];
    }
	
  }
  
  inline void set_smoothers (std::vector <LinearSolver<LAD, LAD>* > smoothers) {
    this->S_.resize(smoothers.size());
    
    for (int i = 0; i < smoothers.size(); ++i) {
      assert(smoothers[i]!=nullptr);
      this->S_[i] = smoothers[i];
    }
  }
	
  
  void Clear() {	  
    GMGBASE::Clear();
    
    for (int i = 0; i < Res_.size(); ++ i) {
      if (this->Pro_[i] != nullptr) {  
        delete this->Pro_[i];
        this->Pro_[i] = nullptr;
      }
      if (this->Res_[i] != nullptr) {  
        delete this->Res_[i];
        this->Res_[i] = nullptr;
      }
      
    }
	
    for (int i = 0; i < A_.size(); ++i) {
      if (this->A_[i] != nullptr) {  
        delete this->A_[i];
        this->A_[i] = nullptr;
      }
      if (this->b_[i] != nullptr) {  
        delete this->b_[i];
        this->b_[i] = nullptr;
      }
    }
 
  }
  
  
  
protected :

  Application * ap_;
  std::vector < VectorSpace <DataType , DIM > * > spaces_;
  std::vector <LinearSolver<LAD, LAD> *> S_;
  std::vector < FeInterMapFullNodal< typename LAD::DataType, DIM >* >  Res_;
  std::vector < FeInterMapFullNodal< typename LAD::DataType, DIM > *>  Pro_;

  std::vector < OperatorType *> A_;
  std::vector < VectorType *> b_; 

  virtual LinearSolverState SolveImpl(const VectorType &b, VectorType *x) {
    return GMGBASE::SolveImpl(b, x);
  }
 
 
  virtual void BuildImpl(VectorType const *b, VectorType *x) { 
	
    
    /// init Restriction and Prolongation Operators
    this->Res_.resize(this->spaces_.size()-1);
    this->Pro_.resize(this->spaces_.size()-1);   
    std::vector <size_t> fe_inds;
    
    for (int i = 0; i < spaces_[0]->nb_fe(); ++i) {		 
      fe_inds.push_back(i);	   
    }
   
    for (int i= 0; i < Res_.size(); ++i) {
      this->Res_[i] = new FeInterMapFullNodal<DataType, DIM > ();
      this->Res_[i]->init(spaces_[i+1], spaces_[i], fe_inds, fe_inds );
      this->Pro_[i] = new FeInterMapFullNodal<DataType, DIM > ();
      this->Pro_[i]->init(spaces_[i], spaces_[i+1], fe_inds, fe_inds);
    }
      
      
    /// init matrices and vectors
    this->A_.resize(this->S_.size());
    this->b_.resize(this->S_.size());
      
    assert(this->ap_ != nullptr);
    for (int i = 0; i < A_.size(); ++i) {
      
      this-> A_[i] = new OperatorType();
      
      assert(spaces_[i] != nullptr);
      this->ap_->prepare_operator(spaces_[i], A_[i]);    
      this-> b_[i] = new VectorType();
      this->ap_->assemble_rhs(spaces_[i], b_[i]);   
      
    }
  
    this->set_operators(A_);
    this->set_rhs(b_);
    this->set_prolongation(Pro_);
    this->set_restriction(Res_);
    GMGBASE::set_smoothers(S_);

    GMGBASE::BuildImpl(b, x);
  
    //std::cout << "successfully built the system!" <<std::endl; 
  }
	



};


	
} //end namespace la
} //end namespace hiflow

#endif // HIFLOW_LINEARSOLVER_GMG_H_
