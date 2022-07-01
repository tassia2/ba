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

#ifndef _ASSEMBLY_UTILS_H_
#define _ASSEMBLY_UTILS_H_

#include <vector>

/// \file assembly.h
/// \brief Assembly functions.
///
/// \author Staffan Ronnas, Simon Gawlok, Philipp Gerstner
///

namespace hiflow {
namespace la {
template < class DataType > class Vector;
struct SparsityStructure;
}

template < class DataType, int DIM > class VectorSpace;
template < class DataType, int DIM > class Element;
template < class DataType > class Quadrature;


  ///
  /// \brief Compute the matrix graph for the assembly.
  ///
  /// \details The matrix graph is a set of pairs (i,j) of
  /// global indices which are cannot be guaranteed to give a
  /// zero in the matrix assembly. Typically, these will
  /// correspond to all global basis functions with overlapping
  /// support.
  ///
  /// \param[in]  space        the VecSpace to assemble over
  /// \param[out] sparsity     the sparsity object containing
  ///                          arrays that describe the matrix graph
  /// \param[in] coupling_vars 2D array indicating the coupling of vars.
  ///                          Rows (first index) belong to test variables,
  ///                          columns (second index) belong to trial variables.
  ///                          If entry (i, j) is set to true, test variable i
  ///                          couples with trial variable j, and the
  ///                          corresponding block is contained in the sparsity
  ///                          structure. Otherwise, the block is skipped,
  ///                          resulting in a sparser structure. If this
  ///                          argument is not passed or is empty, full coupling
  ///                          of all variables is assumed. All rows and columns
  ///                          need to have the size of space.get_nb_var().

// new, all in one inmplementation
template < class DataType, int DIM >
void compute_sparsity_structure(const VectorSpace< DataType, DIM > &space, 
                                la::SparsityStructure &sparsity,
                                const std::vector< std::vector< bool > > & coupling_vars,
                                const bool use_interface_integrals);
                                
template < class DataType, int DIM >
void compute_sparsity_structure(const VectorSpace< DataType, DIM > &space, 
                                la::SparsityStructure &sparsity)
{
  std::vector< std::vector< bool > > coupling_vars;
  compute_sparsity_structure(space, sparsity, coupling_vars, true);
}

// old implementations, should be considered as deprecated, since compute_sparsity_structure combines these functions
template < class DataType, int DIM >
void compute_std_sparsity_structure(const VectorSpace< DataType, DIM > &space, 
                                    la::SparsityStructure &sparsity,
                                    const std::vector< std::vector< bool > > & coupling_vars);

template < class DataType, int DIM >
void compute_std_sparsity_structure(const VectorSpace< DataType, DIM > &space, 
                                   la::SparsityStructure &sparsity)
{
  std::vector< std::vector< bool > > coupling_vars;
  compute_std_sparsity_structure(space, sparsity, coupling_vars);
}

template < class DataType, int DIM >
void compute_hp_sparsity_structure(const VectorSpace< DataType, DIM > &space, 
                                   la::SparsityStructure &sparsity,
                                   const std::vector< std::vector< bool > > & coupling_vars);

template < class DataType, int DIM >
void compute_hp_sparsity_structure(const VectorSpace< DataType, DIM > &space, 
                                   la::SparsityStructure &sparsity)
{
  std::vector< std::vector< bool > > coupling_vars;
  compute_hp_sparsity_structure(space, sparsity, coupling_vars);
}

template < class DataType, int DIM >
void compute_dg_sparsity_structure(const VectorSpace< DataType, DIM > &space, 
                                   la::SparsityStructure &sparsity,
                                   const std::vector< std::vector< bool > > & coupling_vars);

template < class DataType, int DIM >
void compute_dg_sparsity_structure(const VectorSpace< DataType, DIM > &space, 
                                   la::SparsityStructure &sparsity)
{
  std::vector< std::vector< bool > > coupling_vars;
  compute_dg_sparsity_structure(space, sparsity, coupling_vars);
}
                                                                                                         
#if 0
// deprecated
template < class DataType, int DIM >
void InitStructure(const VectorSpace< DataType, DIM > &space,
                   std::vector< int > *rows_diag,
                   std::vector< int > *cols_diag,
                   std::vector< int > *rows_offdiag,
                   std::vector< int > *cols_offdiag,
                   std::vector< std::vector< bool > > *coupling_vars);
#endif 

template < class DataType, int DIM >
void init_master_quadrature(const Element< DataType, DIM > &slave_elem,
                            const Element< DataType, DIM > &master_elem,
                            const Quadrature< DataType > &slave_quad,
                            Quadrature< DataType > &master_quad);

} // namespace hiflow

#endif /* _ASSEMBLY_H_ */
