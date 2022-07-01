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

#include "fem/fe_reference.h"
#include "fem/fe_transformation.h"
#include "fem/cell_trafo/cell_transformation.h"

namespace hiflow {
namespace doffem {

////////////////////////////////////////////////////////////////////////////////
////////////////// FETransformationStandard ////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

template < class DataType, int DIM >
void  FETransformationStandard<DataType, DIM>::inverse_map_shape_function_values (const CellTransformation<DataType, DIM>& cell_trafo,
                                                                                  const Coord& ref_pt,
                                                                                  size_t func_offset, size_t num_func, size_t nb_comp,
                                                                                  IndexFunction ind_func,
                                                                                  const std::vector<DataType> & mapped_vals,
                                                                                  std::vector<DataType>& shape_vals) const
{
  assert (mapped_vals.size() % nb_comp == 0);
  assert ((func_offset + num_func) * nb_comp <= mapped_vals.size() );
  assert (shape_vals.size() == mapped_vals.size());

  // loop over shape functions
  for (size_t j=func_offset; j < func_offset + num_func; ++j)
  {
    // phi_j[d] =  phi_hat_j[d]}
    for (size_t d=0; d<nb_comp; ++d)
    {
      //shape_vals[this->iv2ind(j,d)] = mapped_vals[this->iv2ind(j,d)];
      shape_vals[ind_func(j,d)] = mapped_vals[ind_func(j,d)];
    }
  }
}

template < class DataType, int DIM >
void FETransformationStandard<DataType, DIM>::map_shape_function_values (DataType const * detJ, 
                                                                         mat const * J,
                                                                         mat const * JinvT,
                                                                         size_t func_offset, size_t num_func, size_t nb_comp,
                                                                         const RefElement<DataType, DIM>& fe,
                                                                         const std::vector<DataType> & shape_vals, 
                                                                         std::vector<DataType>& mapped_vals) const
{
  assert (shape_vals.size() % nb_comp == 0);
  assert ((func_offset + num_func) * nb_comp <= shape_vals.size() );
  assert (shape_vals.size() == mapped_vals.size());

  // loop over shape functions
  for (size_t j=func_offset, e_j = func_offset + num_func; j != e_j; ++j)
  {
    // phi_j[d] =  phi_hat_j[d]}
    for (size_t d=0; d<nb_comp; ++d)
    {
      const int i = fe.iv2ind(j,d);
      mapped_vals[i] = shape_vals[i];
    }
  }
}

template < class DataType, int DIM >
void FETransformationStandard<DataType, DIM>::map_shape_function_values (const CellTransformation<DataType, DIM>& cell_trafo,
                                                                         const Coord& ref_pt,
                                                                         size_t func_offset, size_t num_func, size_t nb_comp,
                                                                         const RefElement<DataType, DIM>& fe,
                                                                         const std::vector<DataType> & shape_vals,
                                                                         std::vector<DataType>& mapped_vals) const
{
  this->map_shape_function_values (nullptr, nullptr, nullptr, func_offset, num_func, nb_comp, fe, shape_vals, mapped_vals);
}

template < class DataType, int DIM >
void FETransformationStandard<DataType, DIM>::map_shape_function_values (const CellTransformation<DataType, DIM>& cell_trafo,
                                                                         const Coord& ref_pt,
                                                                         size_t func_offset, size_t num_func, size_t nb_comp,
                                                                         IndexFunction ind_func,
                                                                         const std::vector<DataType> & shape_vals,
                                                                         std::vector<DataType>& mapped_vals) const
{
  assert (shape_vals.size() % nb_comp == 0);
  assert ((func_offset + num_func) * nb_comp <= shape_vals.size() );
  assert (shape_vals.size() == mapped_vals.size());

  // loop over shape functions
  for (size_t j=func_offset, e_j = func_offset + num_func; j != e_j; ++j)
  {
    // phi_j[d] =  phi_hat_j[d]}
    for (size_t d=0; d<nb_comp; ++d)
    {
      const int i = ind_func(j,d);
      mapped_vals[i] = shape_vals[i];
    }
  }
}

template < class DataType, int DIM >
void FETransformationStandard<DataType, DIM>::map_shape_function_gradients (DataType const * detJ, 
                                                                            Coord const * grad_inv_detJ, 
                                                                            mat const * J,
                                                                            mat const * Jinv,
                                                                            mat const * JinvT,
                                                                            std::vector< mat > const * H,
                                                                            size_t func_offset, size_t num_func, size_t nb_comp,
                                                                            const RefElement<DataType, DIM>& fe,
                                                                            const std::vector< DataType> & shape_vals, 
                                                                            const std::vector< Coord > & shape_grads,
                                                                            const std::vector< DataType> & mapped_vals,
                                                                            std::vector< Coord > & mapped_grads) const
{
  assert (shape_grads.size() % nb_comp == 0);
  assert ((func_offset + num_func) * nb_comp <= shape_grads.size() );
  assert (shape_grads.size() == mapped_grads.size());
  assert (JinvT != nullptr);
    
  // loop over shape functions
  for (size_t j=func_offset, e_j = func_offset + num_func; j != e_j; ++j)
  {
    // grad phi_j[d] = J^{-T} * phi_hat_j[d]
    for (size_t d=0; d<nb_comp; ++d)
    {
      const int i = fe.iv2ind(j,d);
      JinvT->VectorMult(shape_grads[i], mapped_grads[i]);
    }
  }
}

template < class DataType, int DIM >
void FETransformationStandard<DataType, DIM>::map_shape_function_gradients (const CellTransformation<DataType, DIM>& cell_trafo,
                                                                            const Coord& ref_pt,
                                                                            size_t func_offset, size_t num_func, size_t nb_comp,
                                                                            const RefElement<DataType, DIM>& fe,
                                                                            const std::vector< DataType> & shape_vals,
                                                                            const std::vector< Coord > & shape_grads,
                                                                            const std::vector< DataType> & mapped_vals,
                                                                            std::vector< Coord > & mapped_grads) const
{
  mat J, JinvT;
  cell_trafo.J(ref_pt, J);
  invTransp(J, JinvT);

  this->map_shape_function_gradients (nullptr, nullptr, nullptr, nullptr, &JinvT, nullptr,
                                      func_offset, num_func, nb_comp, fe,
                                      shape_vals, shape_grads, mapped_vals, mapped_grads);
}
                                                            
template < class DataType, int DIM >
void FETransformationStandard<DataType, DIM>::map_shape_function_hessians (mat const * JinvT,
                                                                           std::vector< mat > const * H,
                                                                           size_t func_offset, size_t num_func, size_t nb_comp,
                                                                           const RefElement<DataType, DIM>& fe,
                                                                           const std::vector< Coord > & shape_grads,
                                                                           const std::vector< mat > & shape_hessians,
                                                                           const std::vector< Coord > & mapped_grads, 
                                                                           std::vector< mat > & mapped_hessians) const
{
  assert (shape_hessians.size() % nb_comp == 0);
  assert ((func_offset + num_func) * nb_comp <= shape_hessians.size() );
  assert (shape_hessians.size() == mapped_hessians.size());
  assert (shape_grads.size() == shape_hessians.size());
  assert (mapped_grads.size() == shape_hessians.size());
  assert (JinvT != nullptr);
  assert (H != nullptr);
  assert (H->size() == DIM);

  mat Jinv;
  trans(*JinvT, Jinv);

  // loop over shape functions
  for (size_t i=func_offset, e_i = func_offset + num_func; i != e_i; ++i)
  {
    // loop over components of shape function
    for (size_t d=0; d<nb_comp; ++d)
    {
      // H_mapped_grad(i,d) = sum_c H[c] * grad_phi(i,d)[c];
      mat H_mapped_grad;
      const int id = fe.iv2ind(i,d);

      for (size_t c = 0; c < DIM; ++c) 
      {
        H_mapped_grad.Axpy(H->at(c), mapped_grads[id][c]);
      }

      // mapped_shape_function_hessians(i,d) = JinvT * (H_phi_hat_[q](i,d) - H_mapped_grad(i,d)) * Jinv;*/
      mat temp;
      for (size_t j = 0; j < DIM; ++j) 
      {
        for (size_t l = 0; l < DIM; ++l) 
        {
          const DataType H_phi_temp = shape_hessians[id](j, l) - H_mapped_grad(j, l);
          for (size_t k = 0; k < DIM; ++k) 
          {
            temp.add(j, k, H_phi_temp * Jinv(l, k));
          }
        }
      }
      for (size_t j = 0; j < DIM; ++j) 
      {
        for (size_t l = 0; l < DIM; ++l) 
        {
          // TODO_VECTORIZE
          for (size_t k = 0; k < DIM; ++k) 
          {
            mapped_hessians[id].add(j, k, JinvT->operator()(j, l) * temp(l, k));
          }
        }
      }
    }
  }
}

template < class DataType, int DIM >
void FETransformationStandard<DataType, DIM>::map_shape_function_hessians (const CellTransformation<DataType, DIM>& cell_trafo,
                                                                           const Coord& ref_pt, 
                                                                           size_t func_offset, size_t num_func, size_t nb_comp,
                                                                           const RefElement<DataType, DIM>& fe,
                                                                           const std::vector< Coord > & shape_grads, 
                                                                           const std::vector< mat > & shape_hessians,
                                                                           const std::vector< Coord > & mapped_grads,
                                                                           std::vector< mat > & mapped_hessians) const
{
  mat J, JinvT;
  cell_trafo.J(ref_pt, J);
  invTransp(J, JinvT);
  
  std::vector< mat > H (DIM);
  for (size_t d = 0; d < DIM; ++d)
  {
    cell_trafo.H(ref_pt, d, H[d]);
  }
    
  this->map_shape_function_hessians (&JinvT, &H, func_offset, num_func, nb_comp, fe,
                                     shape_grads, shape_hessians,  
                                     mapped_grads, mapped_hessians);
}

template class FETransformationStandard<float,1>;
template class FETransformationStandard<float,2>;
template class FETransformationStandard<float,3>;
template class FETransformationStandard<double,1>;
template class FETransformationStandard<double,2>;
template class FETransformationStandard<double,3>;

////////////////////////////////////////////////////////////////////////////////
////////////////// FETransformationContraPiola /////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

template < class DataType, int DIM >
void FETransformationContraPiola<DataType, DIM>::inverse_map_shape_function_values (const CellTransformation<DataType, DIM>& cell_trafo,
                                                                                    const Coord& ref_pt,
                                                                                    size_t func_offset, size_t num_func, size_t nb_comp,
                                                                                    IndexFunction ind_func,
                                                                                    const std::vector<DataType> & mapped_vals,
                                                                                    std::vector<DataType>& shape_vals) const
{
  assert (mapped_vals.size() % nb_comp == 0);
  assert ((func_offset + num_func) * nb_comp <= mapped_vals.size() );
  assert (shape_vals.size() == mapped_vals.size());
  assert (nb_comp == DIM);

  // compute J, det(J) and J^{-1} of cell transformatin at reference point
  mat J;
  mat Jinv;
  DataType detJ;
    
  cell_trafo.J_and_detJ (ref_pt, J, detJ);
  inv(J, Jinv);
    
  // loop over shape functions
  for (size_t j=func_offset, e_j = func_offset + num_func; j != e_j; ++j)
  {
    // apply inverse Piola transformation
    for (size_t d=0; d<nb_comp; ++d)
    {
      const int jd = ind_func(j,d);
      shape_vals[jd] = 0.;

      // TODO_VECTORIZE
      for (size_t k=0; k<DIM; ++k)
      {
        assert (ind_func(j,d) < shape_vals.size());
        assert (ind_func(j,k) < mapped_vals.size());
        
        shape_vals[jd] += Jinv(d,k) * mapped_vals[ind_func(j,k)];
      }
      shape_vals[jd] *= detJ;
    }
  }
}

template < class DataType, int DIM >
void FETransformationContraPiola<DataType, DIM>::map_shape_function_values (DataType const * detJ, 
                                                                            mat const * J,
                                                                            mat const * JinvT,
                                                                            size_t func_offset, size_t num_func, size_t nb_comp,
                                                                            const RefElement<DataType, DIM>& fe,
                                                                            const std::vector<DataType> & shape_vals, 
                                                                            std::vector<DataType>& mapped_vals) const
{
  assert (nb_comp == DIM);
  assert (shape_vals.size() % nb_comp == 0);
  assert ((func_offset + num_func) * nb_comp <= shape_vals.size() );
  assert (shape_vals.size() == mapped_vals.size());
  assert (detJ != nullptr);
  assert (*detJ != 0.);
  assert (J != nullptr);
    
  // loop over shape functions
  for (size_t j=func_offset, e_j = func_offset + num_func; j != e_j; ++j)
  {
    // phi_j[d] = 1. / detJ * sum_k {J(d,k) * phi_hat_j[k]}
    for (size_t d=0; d<nb_comp; ++d)
    {
      const int jd = fe.iv2ind(j,d);

      mapped_vals[jd] = 0.;

      // TODO_VECTORIZE
      for (size_t k=0; k<nb_comp; ++k)
      {
        mapped_vals[jd] += J->operator ()(d,k) * shape_vals[fe.iv2ind(j,k)];
      }
      mapped_vals[jd] /= (*detJ);
    }
  }
}

template < class DataType, int DIM >
void FETransformationContraPiola<DataType, DIM>::map_shape_function_values (const CellTransformation<DataType, DIM>& cell_trafo,
                                                                            const Coord& ref_pt,
                                                                            size_t func_offset, size_t num_func, size_t nb_comp,
                                                                            const RefElement<DataType, DIM>& fe,
                                                                            const std::vector<DataType> & shape_vals,
                                                                            std::vector<DataType>& mapped_vals) const
{
  mat J;
  DataType detJ;
  cell_trafo.J_and_detJ(ref_pt, J, detJ);
  
  this->map_shape_function_values (&detJ, &J, nullptr, func_offset, num_func, nb_comp, fe, shape_vals, mapped_vals);
}

template < class DataType, int DIM >
void FETransformationContraPiola<DataType, DIM>::map_shape_function_values (const CellTransformation<DataType, DIM>& cell_trafo,
                                                                            const Coord& ref_pt,
                                                                            size_t func_offset, size_t num_func, size_t nb_comp,
                                                                            IndexFunction ind_func,
                                                                            const std::vector<DataType> & shape_vals,
                                                                            std::vector<DataType>& mapped_vals) const
{
  mat J;
  DataType detJ;
  cell_trafo.J_and_detJ(ref_pt, J, detJ);
  
  assert (nb_comp == DIM);
  assert (shape_vals.size() % nb_comp == 0);
  assert ((func_offset + num_func) * nb_comp <= shape_vals.size() );
  assert (shape_vals.size() == mapped_vals.size());
  assert (detJ != 0.);
    
  // loop over shape functions
  for (size_t j=func_offset, e_j = func_offset + num_func; j != e_j; ++j)
  {
    // phi_j[d] = 1. / detJ * sum_k {J(d,k) * phi_hat_j[k]}
    for (size_t d=0; d<nb_comp; ++d)
    {
      const int jd = ind_func(j,d);

      mapped_vals[jd] = 0.;

      // TODO_VECTORIZE
      for (size_t k=0; k<nb_comp; ++k)
      {
        mapped_vals[jd] += J(d,k) * shape_vals[ind_func(j,k)];
      }
      mapped_vals[jd] /= detJ;
    }
  }
}
  
template < class DataType, int DIM >
void FETransformationContraPiola<DataType, DIM>::map_shape_function_gradients (DataType const * detJ, 
                                                                               Coord const * grad_inv_detJ, 
                                                                               mat const * J,
                                                                               mat const * Jinv,
                                                                               mat const * JinvT,
                                                                               std::vector< mat > const * H,
                                                                               size_t func_offset, size_t num_func, size_t nb_comp,
                                                                               const RefElement<DataType, DIM>& fe,
                                                                               const std::vector< DataType> & shape_vals, 
                                                                               const std::vector< Coord > & shape_grads,
                                                                               const std::vector< DataType> & mapped_vals,
                                                                               std::vector< Coord > & mapped_grads) const 
{
  assert (shape_grads.size() % nb_comp == 0);
  assert ((func_offset + num_func) * nb_comp <= shape_grads.size() );
  assert (detJ != nullptr);
  assert (*detJ != 0.);
  assert (J != nullptr);
  assert (Jinv != nullptr);
  assert (shape_grads.size() == mapped_grads.size());
  assert (nb_comp == DIM);
  
  // compute D_phi = 1/detJ * J * D_phi_hat * J^{-1}
  // with J = DF, F:K_hat -> K = cell trafo
  // DF_(ij) = d_j F_i
  // (D phi_hat)_{id} = d phi_hat_i / d x_d 
    
  // loop over shape functions
  for (size_t j=func_offset, e_j = func_offset + num_func; j != e_j; ++j)
  {
    // loop over components of current mapped function
    for (size_t k=0; k<nb_comp; ++k)
    {
      Coord tmp;
      
      // tmp = J(k,:) * D phi_hat
      for (size_t d=0; d<nb_comp; ++d)
      {
        // TODO_VECTORIZE
        for (size_t i=0; i<nb_comp; ++i)
        {
          tmp.add(d, J->operator ()(k,i) * shape_grads[fe.iv2ind(j,i)][d]);
        }
      }
      
      // grad_k = tmp * J^{-1}
      Coord grad_k;
      for (size_t d=0; d<nb_comp; ++d)
      {
        // TODO_VECTORIZE
        for (size_t i=0; i<nb_comp; ++i)
        {
          grad_k.add(d, tmp[i] * Jinv->operator ()(i,d));
        }
      }
      
      // grad_k /= det_J
      grad_k /= (*detJ);
      
      mapped_grads[fe.iv2ind(j,k)] = grad_k;
    }
  }
}
/*
 * OLD
    // phi_hat^(j)
    Coord phi;
    for (size_t d= 0; d<nb_comp; ++d)
    {
      phi.set(d, shape_vals[ind_func(j,d)]);
    }
    
    // J * phi_hat  
    Coord J_mult_phi;
    J->VectorMult(phi, J_mult_phi);
    
    // loop over components of current mapped function
    for (size_t k=0; k<nb_comp; ++k)
    {
      // tmp = H^(k) * phi_hat
      Coord tmp;
      H->at(k).VectorMult(phi, tmp);
      
      // tmp += (D phi_hat )^T * J^T(:,k)
      // (D phi_hat)_{id} = d phi_hat_i / d x_d 
      for (size_t d=0; d<nb_comp; ++d)
      {
        for (size_t i=0; i<nb_comp; ++i)
        {
          tmp[d] += shape_grads[ind_func(j,i)][d] * J->operator ()(k,d);
        }
      }
      
      // tmp /= det_J
      tmp *= (1. / (*detJ));
      
      // tmp += grad(inv_detJ) * (J_phi_hat)_k
      tmp.Axpy(*grad_inv_detJ, J_mult_phi[k]);
      
      // grad(phi_k) = J^{-T} * tmp     
      JinvT->VectorMult(tmp, mapped_grads[ind_func(j,k)]);
    }
*/

// TODO: anpassen
template < class DataType, int DIM >
void FETransformationContraPiola<DataType, DIM>::map_shape_function_gradients (const CellTransformation<DataType, DIM>& cell_trafo,
                                                                               const Coord& ref_pt,
                                                                               size_t func_offset, size_t num_func, size_t nb_comp,
                                                                               const RefElement<DataType, DIM>& fe,
                                                                               const std::vector< DataType> & shape_vals,
                                                                               const std::vector< Coord > & shape_grads,
                                                                               const std::vector< DataType> & mapped_vals,
                                                                               std::vector< Coord > & mapped_grads) const
{
  DataType detJ;
  mat J, Jinv;
  
  cell_trafo.J_and_detJ(ref_pt, J, detJ);
  inv(J, Jinv);
  
  this->map_shape_function_gradients (&detJ, nullptr, &J, &Jinv, nullptr, nullptr,
                                      func_offset, num_func, nb_comp, fe,
                                      shape_vals, shape_grads, mapped_vals, mapped_grads);
}
  
template < class DataType, int DIM >
void FETransformationContraPiola<DataType, DIM>::map_shape_function_hessians (mat const * JinvT,
                                                                              std::vector< mat > const * H,
                                                                              size_t func_offset, size_t num_func, size_t nb_comp,
                                                                              const RefElement<DataType, DIM>& fe,
                                                                              const std::vector< Coord > & shape_grads,
                                                                              const std::vector< mat > & shape_hessians,
                                                                              const std::vector< Coord > & mapped_grads, 
                                                                              std::vector< mat > & mapped_hessians) const
{
  assert (shape_hessians.size() % nb_comp == 0);
  assert ((func_offset + num_func) * nb_comp <= shape_hessians.size() );
  assert (shape_hessians.size() == mapped_hessians.size());
  assert (shape_grads.size() == shape_hessians.size());
  assert (mapped_grads.size() == shape_hessians.size());
  assert (JinvT != nullptr);
  assert (H != nullptr);
  assert (H->size() == DIM);

  NOT_YET_IMPLEMENTED;
}

template < class DataType, int DIM >
void FETransformationContraPiola<DataType, DIM>::map_shape_function_hessians (const CellTransformation<DataType, DIM>& cell_trafo,
                                                                              const Coord& ref_pt, 
                                                                              size_t func_offset, size_t num_func, size_t nb_comp,
                                                                              const RefElement<DataType, DIM>& fe,
                                                                              const std::vector< Coord > & shape_grads,
                                                                              const std::vector< mat > & shape_hessians,
                                                                              const std::vector< Coord > & mapped_grads,
                                                                              std::vector< mat > & mapped_hessians) const
{
  NOT_YET_IMPLEMENTED;
}

template class FETransformationContraPiola<float,1>;
template class FETransformationContraPiola<float,2>;
template class FETransformationContraPiola<float,3>;
template class FETransformationContraPiola<double,1>;
template class FETransformationContraPiola<double,2>;
template class FETransformationContraPiola<double,3>;

////////////////////////////////////////////////////////////////////////////////
////////////////// FETransformationCoPiola /////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

template < class DataType, int DIM >
void FETransformationCoPiola<DataType, DIM>::inverse_map_shape_function_values (const CellTransformation<DataType, DIM>& cell_trafo,
                                                                                const Coord& ref_pt,
                                                                                size_t func_offset, size_t num_func, size_t nb_comp,
                                                                                IndexFunction ind_func,
                                                                                const std::vector<DataType> & mapped_vals,
                                                                                std::vector<DataType>& shape_vals) const
{
  assert (mapped_vals.size() % nb_comp == 0);
  assert ((func_offset + num_func) * nb_comp <= mapped_vals.size() );
  assert (shape_vals.size() == mapped_vals.size());
  assert (nb_comp == DIM);

  // compute J, det(J) and J^{-1} of cell transformatin at reference point
  mat J;
  mat JT;
    
  cell_trafo.J (ref_pt, J);
  trans(J, JT);
    
  // loop over shape functions
  for (size_t j=func_offset, e_j = func_offset + num_func; j != e_j; ++j)
  {
    // apply inverse Piola transformation
    for (size_t d=0; d<nb_comp; ++d)
    {
      const int jd = ind_func(j,d);
      shape_vals[jd] = 0.;
      // TODO_VECTORIZE
      for (size_t k=0; k<DIM; ++k)
      {
        shape_vals[jd] += JT(d,k) * mapped_vals[ind_func(j,k)];
      }
    }
  }
}

template < class DataType, int DIM >
void FETransformationCoPiola<DataType, DIM>::map_shape_function_values (DataType const * detJ, 
                                                                        mat const * J,
                                                                        mat const * JinvT,
                                                                        size_t func_offset, size_t num_func, size_t nb_comp,
                                                                        const RefElement<DataType, DIM>& fe,
                                                                        const std::vector<DataType> & shape_vals, 
                                                                        std::vector<DataType>& mapped_vals) const
{
  assert (nb_comp == DIM);
  assert (shape_vals.size() % nb_comp == 0);
  assert ((func_offset + num_func) * nb_comp <= shape_vals.size() );
  assert (shape_vals.size() == mapped_vals.size());
  assert (JinvT != nullptr);
    
  // loop over shape functions
  for (size_t j=func_offset, e_j = func_offset + num_func; j != e_j; ++j)
  {
    // phi_j[d] = sum_k {J^{-T}(d,k) * phi_hat_j[k]}
    for (size_t d=0; d<nb_comp; ++d)
    {
      const int jd = fe.iv2ind(j,d);
      mapped_vals[jd] = 0.;
      // TODO_VECTORIZE
      for (size_t k=0; k<nb_comp; ++k)
      {
        mapped_vals[jd] += JinvT->operator ()(d,k) * shape_vals[fe.iv2ind(j,k)];
      }
    }
  }
}

template < class DataType, int DIM >
void FETransformationCoPiola<DataType, DIM>::map_shape_function_values (const CellTransformation<DataType, DIM>& cell_trafo,
                                                                        const Coord& ref_pt,
                                                                        size_t func_offset, size_t num_func, size_t nb_comp,
                                                                        const RefElement<DataType, DIM>& fe,
                                                                        const std::vector<DataType> & shape_vals,
                                                                        std::vector<DataType>& mapped_vals) const
{
  mat J, JinvT;
  cell_trafo.J(ref_pt, J);
  invTransp(J, JinvT);
  
  this->map_shape_function_values (nullptr, nullptr, &JinvT, func_offset, num_func, nb_comp, fe, shape_vals, mapped_vals);
}

template < class DataType, int DIM >
void FETransformationCoPiola<DataType, DIM>::map_shape_function_values (const CellTransformation<DataType, DIM>& cell_trafo,
                                                                        const Coord& ref_pt,
                                                                        size_t func_offset, size_t num_func, size_t nb_comp,
                                                                        IndexFunction ind_func,
                                                                        const std::vector<DataType> & shape_vals,
                                                                        std::vector<DataType>& mapped_vals) const
{
  mat J, JinvT;
  cell_trafo.J(ref_pt, J);
  invTransp(J, JinvT);
  
  assert (nb_comp == DIM);
  assert (shape_vals.size() % nb_comp == 0);
  assert ((func_offset + num_func) * nb_comp <= shape_vals.size() );
  assert (shape_vals.size() == mapped_vals.size());
    
  // loop over shape functions
  for (size_t j=func_offset, e_j = func_offset + num_func; j != e_j; ++j)
  {
    // phi_j[d] = sum_k {J^{-T}(d,k) * phi_hat_j[k]}
    for (size_t d=0; d<nb_comp; ++d)
    {
      const int jd = ind_func(j,d);
      mapped_vals[jd] = 0.;
      // TODO_VECTORIZE
      for (size_t k=0; k<nb_comp; ++k)
      {
        mapped_vals[jd] += JinvT(d,k) * shape_vals[ind_func(j,k)];
      }
    }
  }
}
  
// TODO: check whether correct
template < class DataType, int DIM >
void FETransformationCoPiola<DataType, DIM>::map_shape_function_gradients (DataType const * detJ, 
                                                                           Coord const * grad_inv_detJ, 
                                                                           mat const * J,
                                                                           mat const * Jinv,
                                                                           mat const * JinvT,
                                                                           std::vector< mat > const * H,
                                                                           size_t func_offset, size_t num_func, size_t nb_comp,
                                                                           const RefElement<DataType, DIM>& fe,
                                                                           const std::vector< DataType> & shape_vals, 
                                                                           const std::vector< Coord > & shape_grads,
                                                                           const std::vector< DataType> & mapped_vals,
                                                                           std::vector< Coord > & mapped_grads) const 
{
  assert (shape_grads.size() % nb_comp == 0);
  assert ((func_offset + num_func) * nb_comp <= shape_grads.size() );
  assert (Jinv != nullptr);
  assert (JinvT != nullptr);
  assert (H != nullptr);
  assert (H->size() == DIM);
  assert (shape_grads.size() == mapped_grads.size());
  assert (nb_comp == DIM);

  // loop over shape functions
  for (size_t j=func_offset, e_j = func_offset + num_func; j != e_j; ++j)
  {
    // phi := phi^(j)
    Coord phi;
    for (size_t d= 0; d<nb_comp; ++d)
    {
      phi.set(d, mapped_vals[fe.iv2ind(j,d)]);
    }
      
    // S = D(phi_hat)
    mat S;
    for (size_t d=0; d<nb_comp; ++d)
    {
      for (size_t i=0; i<nb_comp; ++i)
      {
        S.set(i,d, shape_grads[fe.iv2ind(j,i)][d]);
      }
    }
    
    // S -= sum_k H_k * phi_k
    for (size_t k=0; k<nb_comp; ++k)
    {
      for (size_t d=0; d<nb_comp; ++d)
      {
        for (size_t i=0; i<nb_comp; ++i)
        {
          S.add(i,d, -H->at(k)(i,d) * phi[k]);
        }
      }
    }
    
    // tmp = S * Jinv
    mat tmp = S * (*Jinv);
    
    // grad_phi_k = JinvT(k,:)*tmp
    for (size_t k=0; k<nb_comp; ++k)
    {
      const int jk = fe.iv2ind(j,k);
      for (size_t d=0; d<nb_comp; ++d)
      {
        mapped_grads[jk].set(d, 0.);
        for (size_t l=0; l<nb_comp; ++l)
        {
          mapped_grads[jk].add(d, JinvT->operator ()(k, l) * tmp(l, d)); 
        }
      }
    }
  }
}

template < class DataType, int DIM >
void FETransformationCoPiola<DataType, DIM>::map_shape_function_gradients (const CellTransformation<DataType, DIM>& cell_trafo,
                                                                           const Coord& ref_pt,
                                                                           size_t func_offset, size_t num_func, size_t nb_comp,
                                                                           const RefElement<DataType, DIM>& fe,
                                                                           const std::vector< DataType> & shape_vals,
                                                                           const std::vector< Coord > & shape_grads,
                                                                           const std::vector< DataType> & mapped_vals,
                                                                           std::vector< Coord > & mapped_grads) const
{
  mat J, Jinv, JinvT;
  std::vector< mat > H(DIM);
    
  cell_trafo.J (ref_pt, J);
  inv(J, Jinv);
  trans(Jinv, JinvT);
    
  for (size_t d = 0; d < DIM; ++d)
  {
    cell_trafo.H(ref_pt, d, H[d]);
  }
        
  this->map_shape_function_gradients (nullptr, nullptr, nullptr, &Jinv, &JinvT, &H,
                                      func_offset, num_func, nb_comp, fe,
                                      shape_vals, shape_grads, mapped_vals, mapped_grads);
}

template < class DataType, int DIM >
void FETransformationCoPiola<DataType, DIM>::map_shape_function_hessians (mat const * JinvT,
                                                                          std::vector< mat > const * H,
                                                                          size_t func_offset, size_t num_func, size_t nb_comp,
                                                                          const RefElement<DataType, DIM>& fe,
                                                                          const std::vector< Coord > & shape_grads,
                                                                          const std::vector< mat > & shape_hessians,
                                                                          const std::vector< Coord > & mapped_grads, 
                                                                          std::vector< mat > & mapped_hessians) const
{
  assert (shape_hessians.size() % nb_comp == 0);
  assert ((func_offset + num_func) * nb_comp <= shape_hessians.size() );
  assert (shape_hessians.size() == mapped_hessians.size());
  assert (shape_grads.size() == shape_hessians.size());
  assert (mapped_grads.size() == shape_hessians.size());
  assert (JinvT != nullptr);
  assert (H != nullptr);
  assert (H->size() == DIM);

  NOT_YET_IMPLEMENTED;
}

template < class DataType, int DIM >
void FETransformationCoPiola<DataType, DIM>::map_shape_function_hessians (const CellTransformation<DataType, DIM>& cell_trafo,
                                                                          const Coord& ref_pt, 
                                                                          size_t func_offset, size_t num_func, size_t nb_comp,
                                                                          const RefElement<DataType, DIM>& fe,
                                                                          const std::vector< Coord > & shape_grads,
                                                                          const std::vector< mat > & shape_hessians,
                                                                          const std::vector< Coord > & mapped_grads,
                                                                          std::vector< mat > & mapped_hessians) const
{
  NOT_YET_IMPLEMENTED;
}

template class FETransformationCoPiola<float,1>;
template class FETransformationCoPiola<float,2>;
template class FETransformationCoPiola<float,3>;
template class FETransformationCoPiola<double,1>;
template class FETransformationCoPiola<double,2>;
template class FETransformationCoPiola<double,3>;

}
}
