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

/// @author Dimitar Lukarski

#include "config.h"

#include "init_vec_mat.h"

#include "lvector_cpu.h"

#include "lmatrix_coo_cpu.h"
#include "lmatrix_csr_cpu.h"
#include "lmatrix_dense_cpu.h"

#include "cuda/GPUcublas2_CSR_lMatrix.h"
#include "cuda/GPUcublas2_lVector.h"
#include "cuda/lmatrix_coo_gpu.h"
#include "cuda/lmatrix_csr_gpu.h"
#include "cuda/lvector_gpu.h"

#include "opencl/lmatrix_csr_opencl.h"
#include "opencl/lvector_opencl.h"

#include "mkl/CPUmkl_coo_lmatrix.h"
#include "mkl/CPUmkl_csr_lmatrix.h"
#include "mkl/CPUmkl_lvector.h"

#include "common/macros.h"
#include "common/property_tree.h"
#include "lmp_log.h"

#include <iostream>
#include <stdlib.h>

namespace hiflow {
namespace la {

void read_LA_parameters(const PropertyTree &c,
                        PLATFORM& la_platform,
                        IMPLEMENTATION& la_impl,
                        MATRIX_FORMAT& la_matrix_format) 
{
  const std::string platform_str = c["Platform"].template get< std::string >("CPU");
  if (platform_str == "CPU") 
  {
    la_platform = CPU;
  } 
  else if (platform_str == "GPU") 
  {
    la_platform = GPU;
  }
  else if (platform_str == "OPENCL") 
  {
    la_platform = OPENCL;
  }  
  else 
  {
    LOG_ERROR("read_LA_parameters::params: No format of this name "
              "registered(platform).");
  } 

  const std::string impl_str = c["Implementation"].template get< std::string >("Naive");
    
  if (impl_str == "Naive") 
  {
    la_impl = NAIVE;
  } 
  else if (impl_str == "BLAS") 
  {
    la_impl = BLAS;
  } 
  else if (impl_str == "MKL") 
  {
    la_impl = MKL;
  } 
  else if (impl_str == "OPENMP") 
  {
    la_impl = OPENMP;
  } 
  else if (impl_str == "SCALAR") 
  {
    la_impl = SCALAR;
  } 
  else if (impl_str == "SCALAR_TEX") 
  {
    la_impl = SCALAR_TEX;
  }
  else if (impl_str == "HYPRE") 
  {
    la_impl = HYPRE;
  }
  else if (impl_str == "PETSC") 
  {
    la_impl = PETSC;
  }   
  else if (impl_str == "OPEN_CL") 
  {
    la_impl = OPEN_CL;
  }
  else if (impl_str == "CUBLAS2") 
  {
    la_impl = CUBLAS2;
  }      
  else 
  {
    LOG_ERROR("CoupledVectorCreator::params: No format of this name " "registered(implementation).");
  }

  const std::string matrix_str = c["MatrixFormat"].template get< std::string >("CSR");
  if (matrix_str == "CSR") 
  {
    la_matrix_format = CSR;
  } 
  else if (matrix_str == "COO") 
  {
    la_matrix_format = COO;
  }
  else if (matrix_str == "DENSE") 
  {
    la_matrix_format = DENSE;
  }
  else if (matrix_str == "ELL") 
  {
    la_matrix_format = ELL;
  }  
  else 
  {
    LOG_ERROR("CoupledVectorCreator::params: No format of this name " "registered(matrix format).");
  }
}

template < typename ValueType >
lVector< ValueType > *init_vector(const int size, const std::string name,
                                  const enum PLATFORM &platform,
                                  const enum IMPLEMENTATION &implementation) {
  if (platform == CPU) {

    switch (implementation) {

    case NAIVE:
      return new CPUsimple_lVector< ValueType >(size, name);
      break;

    case OPENMP:
      return new CPUopenmp_lVector< ValueType >(size, name);
      break;

    case MKL:
      return new CPUmkl_lVector< ValueType >(size, name);
      break;

    case BLAS:
      return new CPUcblas_lVector< ValueType >(size, name);
      break;

    default:;
    }
  }

#ifdef WITH_CUDA
  if (platform == GPU) {

    switch (implementation) {

    case BLAS:
      return new GPUblas_lVector< ValueType >(size, name);
      break;

    case CUBLAS2:
      return new GPUcublas2_lVector< ValueType >(size, name);
      break;

    default:;
    }
  }
#endif

  LOG_ERROR("init_vector() incompatibility PLATFORM/IMPLEMENTATION");
  LOG_ERROR(" Platform ID=" << platform
                            << " Implementation ID=" << implementation);
  quit_program();
  return NULL;
}

template < typename ValueType >
lVector< ValueType > *init_vector(const int size, const std::string name,
                                  const enum PLATFORM &platform,
                                  const enum IMPLEMENTATION &implementation,
                                  const struct SYSTEM &my_system) {
  if (platform == CPU) {

    switch (implementation) {

    case NAIVE:
      return new CPUsimple_lVector< ValueType >(size, name);
      break;

    case OPENMP:
      return new CPUopenmp_lVector< ValueType >(size, name);
      break;

    case MKL:
      return new CPUmkl_lVector< ValueType >(size, name);
      break;

    case BLAS:
      return new CPUcblas_lVector< ValueType >(size, name);
      break;

    default:;
    }
  }

#ifdef WITH_CUDA
  if (platform == GPU) {

    switch (implementation) {

    case BLAS:
      return new GPUblas_lVector< ValueType >(size, name);
      break;

    default:;
    }
  }
#endif

#ifdef WITH_OPENCL
  if (platform == OPENCL) {
    lVector< ValueType > *new_vector =
        new OPENCL_lVector< ValueType >(my_system.my_manager);
    new_vector->Init(size, name);
    return new_vector;
  }
#endif

  LOG_ERROR("init_vector() incompatibility PLATFORM/IMPLEMENTATION");
  LOG_ERROR(" Platform ID=" << platform
                            << " Implementation ID=" << implementation);
  quit_program();
  return NULL;
}

template < typename ValueType >
lMatrix< ValueType > *
init_matrix(const int init_nnz, const int init_num_row, const int init_num_col,
            const std::string init_name, const enum PLATFORM &platform,
            const enum IMPLEMENTATION &implementation,
            const enum MATRIX_FORMAT &matrix_format) {
  // DENSE
  if (matrix_format == DENSE) {
    if (platform == CPU) {
      switch (implementation) {
      case NAIVE:
        return new CPUsimple_DENSE_lMatrix< ValueType >(
            init_nnz, init_num_row, init_num_col, init_name);
        break;

      default:;
      }
    }
    // CSR
  } else if (matrix_format == CSR) {
    if (platform == CPU) {

      switch (implementation) {

      case NAIVE:
        return new CPUsimple_CSR_lMatrix< ValueType >(init_nnz, init_num_row,
                                                      init_num_col, init_name);
        break;

      case OPENMP:
        return new CPUopenmp_CSR_lMatrix< ValueType >(init_nnz, init_num_row,
                                                      init_num_col, init_name);
        break;

      case MKL:
        return new CPUmkl_CSR_lMatrix< ValueType >(init_nnz, init_num_row,
                                                   init_num_col, init_name);
        break;

      default:;
      }
    }

#ifdef WITH_CUDA
    if (platform == GPU) {

      switch (implementation) {

      case SCALAR:
        return new GPUscalar_CSR_lMatrix< ValueType >(init_nnz, init_num_row,
                                                      init_num_col, init_name);
        break;

      case SCALAR_TEX:
        return new GPUscalartex_CSR_lMatrix< ValueType >(
            init_nnz, init_num_row, init_num_col, init_name);
        break;

      case CUBLAS2:
        return new GPUcublas2_CSR_lMatrix< ValueType >(init_nnz, init_num_row,
                                                       init_num_col, init_name);
        break;

      default:;
      }
    }
#endif

    // COO
  } else if (matrix_format == COO) {
    if (platform == CPU) {

      switch (implementation) {

      case NAIVE:
        return new CPUsimple_COO_lMatrix< ValueType >(init_nnz, init_num_row,
                                                      init_num_col, init_name);
        break;

      case OPENMP:
        return new CPUopenmp_COO_lMatrix< ValueType >(init_nnz, init_num_row,
                                                      init_num_col, init_name);
        break;

      case MKL:
        return new CPUmkl_COO_lMatrix< ValueType >(init_nnz, init_num_row,
                                                   init_num_col, init_name);
        break;

      default:;
      }
    }

#ifdef WITH_CUDA
    if (platform == GPU) {
      // no COO on GPU
    }
#endif
  }

  LOG_ERROR("init_matrix() incompatibility PLATFORM/IMPLEMENTATION/FORMAT");
  LOG_ERROR("Platform ID=" << platform
                           << " Implementation ID=" << implementation
                           << " Matrix Format ID=" << matrix_format);
  quit_program();
  return NULL;
}

template < typename ValueType >
lMatrix< ValueType > *
init_matrix(const int init_nnz, const int init_num_row, const int init_num_col,
            const std::string init_name, const enum PLATFORM &platform,
            const enum IMPLEMENTATION &implementation,
            const enum MATRIX_FORMAT &matrix_format,
            const struct SYSTEM &my_system) {
  // DENSE
  if (matrix_format == DENSE) {
    if (platform == CPU) {
      switch (implementation) {
      case NAIVE:
        return new CPUsimple_DENSE_lMatrix< ValueType >(
            init_nnz, init_num_row, init_num_col, init_name);
        break;

      default:;
      }
    }
    // CSR
  } else if (matrix_format == CSR) {
    if (platform == CPU) {

      switch (implementation) {

      case NAIVE:
        return new CPUsimple_CSR_lMatrix< ValueType >(init_nnz, init_num_row,
                                                      init_num_col, init_name);
        break;

      case OPENMP:
        return new CPUopenmp_CSR_lMatrix< ValueType >(init_nnz, init_num_row,
                                                      init_num_col, init_name);
        break;

      case MKL:
        return new CPUmkl_CSR_lMatrix< ValueType >(init_nnz, init_num_row,
                                                   init_num_col, init_name);
        break;

      default:;
      }
    }

#ifdef WITH_CUDA
    if (platform == GPU) {

      switch (implementation) {

      case SCALAR:
        return new GPUscalar_CSR_lMatrix< ValueType >(init_nnz, init_num_row,
                                                      init_num_col, init_name);
        break;

      case SCALAR_TEX:
        return new GPUscalartex_CSR_lMatrix< ValueType >(
            init_nnz, init_num_row, init_num_col, init_name);
        break;

      case CUBLAS2:
        return new GPUcublas2_CSR_lMatrix< ValueType >(init_nnz, init_num_row,
                                                       init_num_col, init_name);
        break;

      default:;
      }
    }
#endif
  }

  // COO
  if (matrix_format == COO) {
    if (platform == CPU) {

      switch (implementation) {

      case NAIVE:
        return new CPUsimple_COO_lMatrix< ValueType >(init_nnz, init_num_row,
                                                      init_num_col, init_name);
        break;

      case OPENMP:
        return new CPUopenmp_COO_lMatrix< ValueType >(init_nnz, init_num_row,
                                                      init_num_col, init_name);
        break;

      case MKL:
        return new CPUmkl_COO_lMatrix< ValueType >(init_nnz, init_num_row,
                                                   init_num_col, init_name);
        break;

      default:;
      }
    }

#ifdef WITH_CUDA
    if (platform == GPU) {
      // no COO on GPU
    }
#endif
  }

#ifdef WITH_OPENCL
  if (matrix_format == CSR && platform == OPENCL) {
    lMatrix< ValueType > *new_matrix =
        new OPENCL_CSR_lMatrix< ValueType >(my_system.my_manager);
    new_matrix->Init(init_nnz, init_num_row, init_num_col, init_name);
    return new_matrix;
  }
#endif

  LOG_ERROR("init_matrix() incompatibility PLATFORM/IMPLEMENTATION/FORMAT");
  LOG_ERROR("Platform ID=" << platform
                           << " Implementation ID=" << implementation
                           << " Matrix Format ID=" << matrix_format);
  quit_program();
  return NULL;
}

template lVector< double > *
init_vector(const int size, const std::string name,
            const enum PLATFORM &platform,
            const enum IMPLEMENTATION &implementation);

template lVector< double > *init_vector(
    const int size, const std::string name, const enum PLATFORM &platform,
    const enum IMPLEMENTATION &implementation, const struct SYSTEM &my_system);

template lVector< float > *
init_vector(const int size, const std::string name,
            const enum PLATFORM &platform,
            const enum IMPLEMENTATION &implementation);

template lVector< float > *init_vector(
    const int size, const std::string name, const enum PLATFORM &platform,
    const enum IMPLEMENTATION &implementation, const struct SYSTEM &my_system);

template lMatrix< double > *
init_matrix(const int init_nnz, const int init_num_row, const int init_num_col,
            const std::string init_name, const enum PLATFORM &platform,
            const enum IMPLEMENTATION &implementation,
            const enum MATRIX_FORMAT &matrix_format);

template lMatrix< double > *
init_matrix(const int init_nnz, const int init_num_row, const int init_num_col,
            const std::string init_name, const enum PLATFORM &platform,
            const enum IMPLEMENTATION &implementation,
            const enum MATRIX_FORMAT &matrix_format,
            const struct SYSTEM &my_system);

template lMatrix< float > *
init_matrix(const int init_nnz, const int init_num_row, const int init_num_col,
            const std::string init_name, const enum PLATFORM &platform,
            const enum IMPLEMENTATION &implementation,
            const enum MATRIX_FORMAT &matrix_format);

template lMatrix< float > *
init_matrix(const int init_nnz, const int init_num_row, const int init_num_col,
            const std::string init_name, const enum PLATFORM &platform,
            const enum IMPLEMENTATION &implementation,
            const enum MATRIX_FORMAT &matrix_format,
            const struct SYSTEM &my_system);

} // namespace la
} // namespace hiflow
