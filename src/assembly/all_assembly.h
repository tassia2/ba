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

/// \author Staffan Ronnas, Simon Gawlok

#ifndef _ALL_GLOBAL_ASSEMBLY_H_
#define _ALL_GLOBAL_ASSEMBLY_H_

#include <vector>
#include "assembly/assembly_types.h"
#include "assembly/assembly_utils.h"
#include "assembly/assembly_routines.h"
#include "assembly/generic_assembly_algorithm.h"
#include "common/pointers.h"
#include "common/array_tools.h"
#include "mesh/attributes.h"
#include "mesh/types.h"
#include "mesh/iterator.h"
#include "space/element.h"
#include "space/vector_space.h"

namespace hiflow {

//////////////// Implementation of AllGlobalAssembler ////////////////
template < class DataType, int DIM >
class AllGlobalAssembler 
{
public:

  typedef la::Matrix< DataType > GlobalMatrix;
  typedef la::Vector< DataType > GlobalVector;
  typedef la::SeqDenseMatrix< DataType > LocalMatrix;
  typedef std::vector< DataType > LocalVector;
  typedef VectorSpace< DataType, DIM > VecSpace;

  /// Generalized function type used for quadrature selection.
  typedef std::function< void (const Element< DataType, DIM > &,
                               Quadrature< DataType > &) >
    QuadratureSelectionFunction;

  typedef std::function< void (const Element< DataType, DIM > &,
                               Quadrature< DataType > &, const int) >
    FacetQuadratureSelectionFunction;

  // TODO: this could be optimized by splitting into separate cell/facet
  // selection functions.
  typedef std::function < void ( const Element< DataType, DIM > &, const Element< DataType, DIM > &, 
                                 int, int,
                                 Quadrature< DataType > &, Quadrature< DataType > &) >
    IFQuadratureSelectionFunction;

  AllGlobalAssembler()
  : q_select_(DefaultQuadratureSelection< DataType, DIM >()), 
    fq_select_(DefaultFacetQuadratureSelection< DataType, DIM >()), 
    if_q_select_(DefaultInterfaceQuadratureSelection< DataType, DIM >()),
    should_reset_assembly_target_(true) {}
  

  ///
  /// \brief Set the function used to determine which quadrature rule should be
  /// used.
  ///
  /// \details The choice of quadrature rules can be controlled
  /// by providing a QuadratureSelectionFunction. This function
  /// or functor will be called before the local assembly is
  /// performed on each element. By default, the
  /// DefaultQuadratureSelection function is used.
  ///
  /// \param[in] q_select   new quadrature selection function
  void set_quadrature_selection_function(QuadratureSelectionFunction q_select)
  {
    if (q_select != 0) 
    {
      q_select_ = q_select;
    } 
  }

  ///
  /// \brief Set the function used to determine which quadrature rule should be
  /// used. for interface quadrature.
  ///
  /// \details The choice of quadrature rules can be controlled
  /// by providing a IFQuadratureSelectionFunction. This function
  /// or functor will be called before the local assembly is
  /// performed on each element. By default, the
  /// DefaultInterfaceQuadratureSelection function is used.
  ///
  /// \param[in] q_select   new quadrature selection function
  void set_interface_quadrature_selection_fun(IFQuadratureSelectionFunction q_select)
  {
    if (q_select != 0) 
    {
      this->if_q_select_ = q_select;
    } 
  }

  ///
  /// \brief Change whether the assembled object should be
  /// reset prior to assembly.
  ///
  /// \details This function can be used to decide whether the
  /// target object of the assembly (scalar for scalar assembly,
  /// vector for vector assembly, etc) in the various assembly
  /// functions should be reset to zero before the
  /// assembly. Setting this value to false can be used to
  /// perform several assembly steps with the same object,
  /// e.g. for adding a boundary term to an assembly. By
  /// default, this is set to true.
  ///
  /// \param[in] should_reset   whether or not assembly object should be reset
  /// to zero.
  void should_reset_assembly_target(bool should_reset)
  {
    should_reset_assembly_target_ = should_reset;
  }

  ///
  /// \brief Assemble global matrix for a VecSpace.
  ///
  /// \details Assembles a matrix
  /// \f$A_{ij} = \int_{\Omega}{f(x,\varphi_i, \varphi_j)dx}\f$
  /// over the domain defined by the mesh
  /// associated to a VecSpace. The integrand is defined through
  /// a MatrixAssemblyFunction, which should return the locally
  /// assembled matrix for each element.
  ///
  /// \param[in] space           the VecSpace for which the assembly is
  /// performed \param[in] local_asm       function or functor that performs
  /// local matrix assembly \param[out] global_matrix  the assembled matrix
  /// \f$A_{ij}\f$ \see concept_assembly
  /// \param[in] traversal: assembler iterates through this list of cell entity numbers 
  
  template< class LocalAssembler>
  void assemble_matrix(const VecSpace &space,
                       LocalAssembler& local_asm,
                       GlobalMatrix &matrix) const;

  template< class LocalAssembler>
  void assemble_matrix(const VecSpace &space,
                       const std::vector<int>& traversal,
                       LocalAssembler& local_asm,
                       GlobalMatrix &matrix) const;

  /// \details this variant works similarly as the standard matrix assembly operation.
  /// However, the computed local matrices are not inserted into the Global Matrix.
  /// This function is used, if the local matrices are stored inside the user-defined 
  /// LocalAssembler, e.g. for further use in StencilOperator
  template< class LocalAssembler>
  void assemble_matrix(const VecSpace &space,
                       LocalAssembler& local_asm) const;

  template< class LocalAssembler>
  void assemble_matrix(const VecSpace &space,
                       const std::vector<int>& traversal,
                       LocalAssembler& local_asm) const;

  ///
  /// \brief Assemble global matrix for a VecSpace, defined
  /// by a boundary integral.
  ///
  /// \details Assembles a matrix
  /// \f$A_{ij} = \int_{\partial\Omega}{f(x,\varphi_i, \varphi_j)dx}\f$
  /// over the boundary of the domain defined by the mesh
  /// associated to a VecSpace. The integrand is defined through
  /// a BoundaryMatrixAssemblyFunction, which should return the locally
  /// assembled matrix for each boundary facet.
  ///
  /// \param[in] space           the VecSpace for which the assembly is
  /// performed \param[in] local_asm       function or functor that performs
  /// local matrix assembly \param[out] global_matrix  the assembled matrix
  /// \f$A_{ij}\f$ \see concept_assembly
  ///
  template< class LocalAssembler>
  void assemble_matrix_boundary(const VecSpace &space,
                                LocalAssembler& local_asm,
                                GlobalMatrix &matrix) const;

  ///
  /// \brief Assemble global matrix for a VecSpace.
  ///
  /// \details Assembles a matrix
  /// \f$A_{ij} = \sum_{K \in \mathcal T_h}\int_{\partial K}{f(x,\varphi_i,
  /// \varphi_j)dx}\f$ over the interfaces defined by the mesh associated to a
  /// VecSpace. The integrand is defined through a
  /// InterfaceMatrixAssemblyFun, which should return the locally assembled
  /// matrix for each element.
  ///
  /// \param[in] space           the VecSpace for which the assembly is
  /// performed \param[in] local_asm       function or functor that performs
  /// local matrix assembly \param[out] global_matrix  the assembled matrix
  /// \f$A_{ij}\f$ \see concept_assembly
  ///
  /// template Argument LocalAssembler should provide the following routine:
  /// void operator() (const Element< DataType, DIM > &master_elem,
  ///                  const Element< DataType, DIM > &slave_elem,
  ///                  const Quadrature< DataType > &master_quad,
  ///                  const Quadrature< DataType > &slave_quad,
  ///                  int master_facet_number, 
  ///                  int slave_facet_number,
  ///                  InterfaceSide if_side, 
  ///                  int slave_index, int num_slaves, 
  ///                  typename GlobalAssembler< DataType, DIM >::LocalMatrix &vals)
  
  
  template<class LocalAssembler>
  void assemble_interface_matrix(const VecSpace &space,
                                 LocalAssembler& local_asm,
                                 GlobalMatrix &matrix) const;

  template< class LocalAssembler>
  void assemble_interface_matrix(const VecSpace &space, 
                                 const mesh::InterfaceList& if_list,
                                 LocalAssembler& local_asm,
                                 GlobalMatrix &matrix) const;

  /// \details this variant works similarly as the interface matrix assembly operation.
  /// However, the computed local matrices are not inserted into the Global Matrix.
  /// This function is used, if the local matrices are stored inside the user-defined 
  /// LocalAssembler, e.g. for further use in StencilOperator
  template<class LocalAssembler>
  void assemble_interface_matrix(const VecSpace &space,
                                 LocalAssembler& local_asm) const;

  template< class LocalAssembler>
  void assemble_interface_matrix(const VecSpace &space, 
                                 const mesh::InterfaceList& if_list,
                                 LocalAssembler& local_asm) const;

  ///
  /// \brief Assemble global vector for a VecSpace.
  ///
  /// \details Assembles a vector
  /// \f$b_i = \int_{\Omega}{f(x, \varphi_i)dx}\f$
  /// over the domain defined by the mesh
  /// associated to a VecSpace. The integrand is defined through
  /// a VectorAssemblyFunction, which should return the locally
  /// assembled vector for each element.
  ///
  /// \param[in] space      the VecSpace for which the assembly is performed
  /// \param[in] local_asm  function or functor that performs local vector
  /// assembly \param[out] global_vector  the assembled vector \f$b_i\f$ \see
  /// concept_assembly
  ///
  template< class LocalAssembler>
  void assemble_vector(const VecSpace &space,
                       LocalAssembler& local_asm,
                       GlobalVector &vec) const;

  template< class LocalAssembler>
  void assemble_vector(const VecSpace &space,
                       const std::vector<int>& traversal,
                       LocalAssembler& local_asm,
                       GlobalVector &vec) const;

  ///
  /// \brief Assemble global vector for a VecSpace, defined
  /// by a boundary integral.
  ///
  /// \details Assembles a vector
  /// \f$b_i = \int_{\partial\Omega}{f(x, \varphi_i)ds}\f$
  /// over the boundary of the domain defined by the mesh
  /// associated to a VecSpace. The integrand is defined through
  /// a BoundaryVectorAssemblyFunction, which should return the locally
  /// assembled vector for each boundary facet.
  ///
  /// \param[in] space           the VecSpace for which the assembly is
  /// performed \param[in] local_asm       function or functor that performs
  /// local vector assembly \param[out] global_vector  the assembled vector
  /// \f$b_i\f$ \see concept_assembly
  ///
  template< class LocalAssembler>
  void assemble_vector_boundary(const VecSpace &space,
                                LocalAssembler& local_asm,
                                GlobalVector &vec) const;


  template<class LocalAssembler>
  void assemble_interface_vector (const VecSpace &space,
                                  LocalAssembler& local_asm,
                                  GlobalVector &vec) const;

  template< class LocalAssembler>
  void assemble_interface_vector(const VecSpace &space,
                                 const mesh::InterfaceList& if_list,
                                 LocalAssembler& local_asm,
                                 GlobalVector &vector) const;

  ///
  /// \brief Compute element-wise integral for a VecSpace.
  ///
  /// \details Computes an integral \f$\int_{K}{f(x)dx}\f$
  /// over the cells \f$K\f$ in the mesh associated to a
  /// VecSpace. The integrand is defined through the
  /// ScalarAssemblyFunction, which should return the value of
  /// the integral for each element.
  ///
  /// \param[in] space      the VecSpace to integrate over
  /// \param[in] local_asm  function or functor that performs local integration
  /// \param[out] vec       the value of the integral for each cell in the mesh
  /// \see concept_assembly
  ///
  template< class LocalAssembler>
  void assemble_scalar(const VecSpace &space,
                       LocalAssembler& local_asm,
                       std::vector< DataType > &vec) const;


  ///
  /// \brief Compute integral for a VecSpace.
  ///
  /// \details Computes an integral \f$\int_{\Omega}{f(x)dx}\f$
  /// over the domain defined by the mesh associated to a
  /// VecSpace. The integrand is defined through the
  /// ScalarAssemblyFunction, which should return the value of
  /// the integral for each element.
  ///
  /// \param[in] space      the VecSpace to integrate over
  /// \param[in] local_asm  function or functor that performs local integration
  /// \param[out] value     the value of the integral
  /// \see concept_assembly
  ///
  template< class LocalAssembler>
  void integrate_scalar(const VecSpace &space,
                        LocalAssembler& local_asm,
                        DataType &integral) const;

  ///
  /// \brief Compute facet-wise boundary integral for a VecSpace.
  ///
  /// \details Computes a boundary integral \f$\int_{\partial K}{f(x)dx}\f$
  /// over the cells in the mesh associated to a
  /// VecSpace. The integrand is defined through the
  /// BoundaryScalarAssemblyFunction, which should return the value of
  /// the integral for each boundary facet.
  ///
  /// \param[in] space      the VecSpace to integrate over
  /// \param[in] local_asm  function or functor that performs local integration
  /// \param[out] vec       the value of the integral for each facet on the
  /// boundary \see concept_assembly
  ///
  template< class LocalAssembler>
  void assemble_scalar_boundary(const VecSpace &space,
                                LocalAssembler& local_asm,
                                std::vector< DataType > &vec) const;

  ///
  /// \brief Compute boundary integral for a VecSpace.
  ///
  /// \details Computes a boundary integral
  /// \f$\int_{\partial\Omega}{f(x)dx}\f$ over the boundary of
  /// the domain defined by the mesh associated to a
  /// VecSpace. The integrand is defined through the
  /// BoundaryScalarAssemblyFunction, which should return the value of
  /// the integral for each element.
  ///
  /// \param[in] space      the VecSpace to integrate over
  /// \param[in] local_asm  function or functor that performs local integration
  /// \param[out] value     the value of the integral
  /// \see concept_assembly
  ///
  template< class LocalAssembler>
  void integrate_scalar_boundary(const VecSpace &space,
                                 LocalAssembler& local_asm,
                                 DataType &integral) const;

  ///
  /// \brief Compute boundary maximum for a VecSpace.
  ///
  /// \details Determines a maximum value
  /// \f$\max_{\partial\Omega}{f(x)}\f$ at the boundary of
  /// the domain defined by the mesh associated to a
  /// VecSpace. The cell maxima are defined through the
  /// BoundaryScalarAssemblyFunction, which should return the maximal value
  /// for each element.
  ///
  /// \param[in] space      the VecSpace to maximize over
  /// \param[in] local_asm  function or functor that performs local maximization
  /// \param[out] maximum   the maximal value
  /// \see concept_assembly
  ///
  template< class LocalAssembler>
  void maximize_scalar_boundary(const VecSpace &space,
                                LocalAssembler& local_asm,
                                DataType &maximum) const;

  template< class LocalAssembler>
  void assemble_interface_scalar(const VecSpace &space, 
                                 LocalAssembler& local_asm, 
                                 std::vector< DataType > &values) const;

  template< class LocalAssembler>
  void assemble_interface_scalar_cells(const VecSpace &space, 
                                       LocalAssembler& local_asm, 
                                       std::vector< DataType > &values) const;

  ///
  /// \brief Compute integral for a VecSpace.
  ///
  /// \details Computes an integral \f$\int_{\Omega}{f(x)dx}\f$
  /// over the domain defined by the mesh associated to a
  /// VecSpace. The vector valued integrand is defined through the
  /// MultipleScalarAssemblyFunction, which should return the value of
  /// the integral for each element.
  ///
  /// \param[in] space      the VecSpace to integrate over
  /// \param[in] local_asm  function or functor that performs local integration
  /// \param[in] num_scalars dimension of vector valued integrand
  /// \param[out] value     the value of the integral
  /// \see concept_assembly
  ///
  template< class LocalAssembler>
  void integrate_multiple_scalar(const VecSpace &space,
                                 LocalAssembler& local_asm,
                                 int num_scalars,
                                 std::vector< DataType > &integral) const;

  ///
  /// \brief Compute element-wise integral for a VecSpace.
  ///
  /// \details Computes an integral \f$\int_{K}{f(x)dx}\f$
  /// over the cells \f$K\f$ in the mesh associated to a
  /// VecSpace. The vector-valued integrand is defined through the
  /// MultipleScalarAssemblyFunction, which should return the values of
  /// the integral for each element.
  ///
  /// \param[in] space      the VecSpace to integrate over
  /// \param[in] local_asm  function or functor that performs local integration
  /// \param[in] num_scalars dimension of vector valued integrand
  /// \param[out] vec       the values of the integral for each cell in the mesh
  /// \see concept_assembly
  ///
  template< class LocalAssembler>
  void assemble_multiple_scalar(const VecSpace &space,
                                LocalAssembler& local_asm,
                                int num_scalars,
                                std::vector< std::vector< DataType > > &vec) const;

  template< class LocalAssembler>
  void integrate_multiple_scalar_boundary(const VecSpace &space,
                                          LocalAssembler& local_asm,
                                          int num_scalars,
                                          std::vector<DataType> &integral) const;

  template< class LocalAssembler>                    
  void assemble_multiple_scalar_boundary(const VecSpace &space,
                                         LocalAssembler& local_asm,
                                         int num_scalars,
                                         std::vector< std::vector< DataType > > &vec) const;

  template< class LocalAssembler>
  void assemble_interface_multiple_scalar_cells(const VecSpace &space,
                                                LocalAssembler& local_asm, 
                                                const int num_scalars,
                                                std::vector< LocalVector > &values) const;          



private:
  void set_constraint_rows_to_identity (const VecSpace &space, GlobalMatrix &matrix) const;

  IFQuadratureSelectionFunction if_q_select_;  
  QuadratureSelectionFunction q_select_;
  FacetQuadratureSelectionFunction fq_select_;
  bool should_reset_assembly_target_;                               
};

template < class DataType, int DIM >
void AllGlobalAssembler< DataType, DIM >::set_constraint_rows_to_identity(const VecSpace &space, 
                                                                          GlobalMatrix &matrix) const 
{
  // Set rows of constrained dofs to identity to obtain non-singular matrix
  SortedArray< int > constrained_dofs;
  const DofInterpolation<DataType> &interp = space.dof().dof_interpolation();
  
  for (auto it = interp.begin(), end = interp.end(); it != end; ++it) 
  {
    if (space.dof().is_dof_on_subdom(it->first)) 
    {
      constrained_dofs.find_insert(it->first);
    }
  }
  
  if (!constrained_dofs.empty()) 
  {
    matrix.diagonalize_rows(&constrained_dofs.front(), constrained_dofs.size(), 1.);
  }
}

template < class DataType, int DIM >
template< class LocalAssembler>
void AllGlobalAssembler< DataType, DIM >::assemble_matrix(const VecSpace &space, 
                                                          LocalAssembler& local_asm,
                                                          GlobalMatrix &matrix) const 
{
  if (should_reset_assembly_target_) 
  {
    matrix.Zeros();
  }
  CellMatrixAssembly< InteriorAssemblyAlgorithm, DataType, DIM > assembly(space, matrix);
  assembly.assemble(local_asm, this->q_select_);
  this->set_constraint_rows_to_identity(space, matrix);
}

template < class DataType, int DIM >
template< class LocalAssembler>
void AllGlobalAssembler< DataType, DIM >::assemble_matrix(const VecSpace &space, 
                                                          const std::vector<int>& traversal,
                                                          LocalAssembler& local_asm,
                                                          GlobalMatrix &matrix) const 
{
  if (should_reset_assembly_target_) 
  {
    matrix.Zeros();
  }
  CellMatrixAssembly< InteriorAssemblyAlgorithm, DataType, DIM > assembly(space, traversal, matrix);
  assembly.assemble(local_asm, this->q_select_);
  this->set_constraint_rows_to_identity(space, matrix);
}

template < class DataType, int DIM >
template< class LocalAssembler>
void AllGlobalAssembler< DataType, DIM >::assemble_matrix(const VecSpace &space, 
                                                          LocalAssembler& local_asm) const 
{
  CellMatrixAssembly< InteriorAssemblyAlgorithm, DataType, DIM > assembly(space);
  assembly.assemble(local_asm, this->q_select_);
}

template < class DataType, int DIM >
template< class LocalAssembler>
void AllGlobalAssembler< DataType, DIM >::assemble_matrix(const VecSpace &space, 
                                                          const std::vector<int>& traversal,
                                                          LocalAssembler& local_asm) const 
{
  CellMatrixAssembly< InteriorAssemblyAlgorithm, DataType, DIM > assembly(space, traversal);
  assembly.assemble(local_asm, this->q_select_);
}

template < class DataType, int DIM >
template< class LocalAssembler>
void AllGlobalAssembler< DataType, DIM >::assemble_interface_matrix(const VecSpace &space, 
                                                                    LocalAssembler& local_asm,
                                                                    GlobalMatrix &matrix) const 
{
  // Create interface list from mesh
  mesh::ConstMeshPtr mesh = &space.mesh(); // OK since pointer is intrusive
  mesh::InterfaceList if_list = mesh::InterfaceList::create(mesh);

  this->assemble_interface_matrix(space, if_list, local_asm, matrix);
}

template < class DataType, int DIM >
template< class LocalAssembler>
void AllGlobalAssembler< DataType, DIM >::assemble_interface_matrix(const VecSpace &space, 
                                                                    const mesh::InterfaceList& if_list,
                                                                    LocalAssembler& local_asm,
                                                                    GlobalMatrix &matrix) const 
{
  if (this->should_reset_assembly_target_) 
  {
    matrix.Zeros();
  }
  
  InterfaceMatrixAssembly<DataType,DIM> assem;
  assem.assemble (space, if_list, this->if_q_select_, local_asm, &matrix);
  this->set_constraint_rows_to_identity(space, matrix);
}

template < class DataType, int DIM >
template< class LocalAssembler>
void AllGlobalAssembler< DataType, DIM >::assemble_interface_matrix(const VecSpace &space, 
                                                                    LocalAssembler& local_asm) const 
{ 
  // Create interface list from mesh
  mesh::ConstMeshPtr mesh = &space.mesh(); // OK since pointer is intrusive
  mesh::InterfaceList if_list = mesh::InterfaceList::create(mesh);

  this->assemble_interface_matrix(space, if_list, local_asm);
}

template < class DataType, int DIM >
template< class LocalAssembler>
void AllGlobalAssembler< DataType, DIM >::assemble_interface_matrix(const VecSpace &space,
                                                                    const mesh::InterfaceList& if_list, 
                                                                    LocalAssembler& local_asm) const 
{ 
  InterfaceMatrixAssembly<DataType,DIM> assem;
  assem.assemble (space, if_list, this->if_q_select_, local_asm, nullptr);
}

template < class DataType, int DIM >
template< class LocalAssembler>
void AllGlobalAssembler< DataType, DIM >::assemble_matrix_boundary(const VecSpace &space,
                                                                   LocalAssembler& local_asm,
                                                                   GlobalMatrix &matrix) const 
{
  if (should_reset_assembly_target_) 
  {
    matrix.Zeros();
  }

  CellMatrixAssembly< BoundaryAssemblyAlgorithm, DataType, DIM > assembly(space, matrix);
  assembly.assemble(local_asm, this->fq_select_);
  this->set_constraint_rows_to_identity(space, matrix);
}


template < class DataType, int DIM >
template< class LocalAssembler>
void AllGlobalAssembler< DataType, DIM >::assemble_vector(const VecSpace &space, 
                                                          LocalAssembler& local_asm,
                                                          GlobalVector &vec) const 
{
  if (should_reset_assembly_target_) 
  {
    vec.Zeros();
  }

  CellVectorAssembly< InteriorAssemblyAlgorithm, DataType, DIM > assembly(space, vec);
  assembly.assemble(local_asm, this->q_select_);
}

template < class DataType, int DIM >
template< class LocalAssembler>
void AllGlobalAssembler< DataType, DIM >::assemble_vector(const VecSpace &space, 
                                                          const std::vector<int>& traversal,
                                                          LocalAssembler& local_asm,
                                                          GlobalVector &vec) const 
{
  if (should_reset_assembly_target_) 
  {
    vec.Zeros();
  }

  CellVectorAssembly< InteriorAssemblyAlgorithm, DataType, DIM > assembly(space, traversal, vec);
  assembly.assemble(local_asm, this->q_select_);
}

template < class DataType, int DIM >
template< class LocalAssembler>
void AllGlobalAssembler< DataType, DIM >::assemble_interface_vector(const VecSpace &space, 
                                                                    LocalAssembler& local_asm,
                                                                    GlobalVector &vector) const 
{ 
  // Create interface list from mesh
  mesh::ConstMeshPtr mesh = &space.mesh(); // OK since pointer is intrusive
  mesh::InterfaceList if_list = mesh::InterfaceList::create(mesh);

  this->assemble_interface_vector(space, if_list, local_asm, vector);
}

template < class DataType, int DIM >
template< class LocalAssembler>
void AllGlobalAssembler< DataType, DIM >::assemble_interface_vector(const VecSpace &space,
                                                                    const mesh::InterfaceList& if_list,
                                                                    LocalAssembler& local_asm,
                                                                    GlobalVector &vector) const 
{
  if (this->should_reset_assembly_target_) 
  {
    vector.Zeros();
  }

  InterfaceVectorAssembly<DataType,DIM> assem;
  assem.assemble (space, if_list, this->if_q_select_, local_asm, vector);
}

template < class DataType, int DIM >
template< class LocalAssembler>
void AllGlobalAssembler< DataType, DIM >::assemble_vector_boundary(const VecSpace &space,
                                                                   LocalAssembler& local_asm,
                                                                   GlobalVector &vec) const 
{
  if (should_reset_assembly_target_) 
  {
    vec.Zeros();
  }

  CellVectorAssembly< BoundaryAssemblyAlgorithm, DataType, DIM > assembly(space, vec);
  assembly.assemble(local_asm, this->fq_select_);
}

template < class DataType, int DIM >
template< class LocalAssembler>
void AllGlobalAssembler< DataType, DIM >::integrate_scalar(const VecSpace &space, 
                                                           LocalAssembler& local_asm,
                                                           DataType &integral) const 
{
  const size_t num_cells = space.mesh().num_entities(space.mesh().tdim());
  std::vector< DataType > cell_values(num_cells, 0.);

  assemble_scalar(space, local_asm, cell_values);
  integral = std::accumulate(cell_values.begin(), cell_values.end(), 0.);
}

template < class DataType, int DIM >
template< class LocalAssembler>
void AllGlobalAssembler< DataType, DIM >::assemble_scalar(const VecSpace &space, 
                                                          LocalAssembler& local_asm,
                                                          std::vector< DataType > &vec) const 
{
  if (should_reset_assembly_target_) 
  {
    vec.clear();
    const size_t num_cells = space.mesh().num_entities(space.mesh().tdim());
    vec.resize(num_cells, 0.);
  }

  StandardScalarAssembly< InteriorAssemblyAlgorithm, DataType, DIM > assembly(space, vec);
  assembly.assemble(local_asm, this->q_select_);
}

template < class DataType, int DIM >
template< class LocalAssembler>
void AllGlobalAssembler< DataType, DIM >::assemble_interface_scalar(const VecSpace &space, 
                                                                    LocalAssembler& local_asm, 
                                                                    std::vector< DataType > &values) const 
{
  // Create interface list from mesh
  mesh::ConstMeshPtr mesh = &space.mesh(); // OK since pointer is intrusive
  mesh::InterfaceList if_list = mesh::InterfaceList::create(mesh);

  // Clear and create values data structure
  values.clear();
  values.resize(if_list.size(), 0.);

  InterfaceScalarAssembly<DataType,DIM> assem;
  assem.assemble (space, if_list, this->if_q_select_, local_asm, values);
}

template < class DataType, int DIM >
template< class LocalAssembler>
void AllGlobalAssembler< DataType, DIM >::assemble_interface_scalar_cells(const VecSpace &space, 
                                                                          LocalAssembler& local_asm, 
                                                                          std::vector< DataType > &values) const 
{
  // Create interface list from mesh
  mesh::ConstMeshPtr mesh = &space.mesh(); // OK since pointer is intrusive
  mesh::InterfaceList if_list = mesh::InterfaceList::create(mesh);

  // Clear and create values data structure
  values.clear();
  values.resize(mesh->num_entities(mesh->tdim()), 0.);

  InterfaceCellScalarAssembly<DataType,DIM> assem;
  assem.assemble (space, if_list, this->if_q_select_, local_asm, values);
}

template < class DataType, int DIM >
template< class LocalAssembler>
void AllGlobalAssembler< DataType, DIM >::integrate_scalar_boundary(const VecSpace &space,
                                                                    LocalAssembler& local_asm,
                                                                    DataType &integral) const 
{
  // TODO: actually the number of boundary facets is needed
  const size_t num_facets = space.mesh().num_entities(space.mesh().tdim() - 1);
  std::vector< DataType > cell_values(num_facets, 0.);

  assemble_scalar_boundary(space, local_asm, cell_values);
  integral = std::accumulate(cell_values.begin(), cell_values.end(), 0.);
}

template < class DataType, int DIM >
template< class LocalAssembler>
void AllGlobalAssembler< DataType, DIM >::assemble_multiple_scalar(const VecSpace &space,
                                                                   LocalAssembler& local_asm,
                                                                   const int num_scalars,
                                                                   std::vector< std::vector< DataType > > &vec) const 
{
  if (should_reset_assembly_target_) 
  {
    vec.clear();
    const size_t num_cells = space.mesh().num_entities(space.mesh().tdim());
    vec.resize(num_cells);
    for (size_t l = 0; l < num_cells; ++l) 
    {
      vec[l].resize(num_scalars, 0.);
    }
  }

  StandardMultipleScalarAssembly< InteriorAssemblyAlgorithm, DataType, DIM > assembly(space, vec, num_scalars);
  assembly.assemble(local_asm, this->q_select_);
}

template < class DataType, int DIM >
template< class LocalAssembler>
void AllGlobalAssembler< DataType, DIM >::integrate_multiple_scalar_boundary(const VecSpace &space,
                                                                             LocalAssembler& local_asm,
                                                                             const int num_scalars,
                                                                             std::vector< DataType > &integral) const 
{
  const size_t num_facets = space.mesh().num_entities(space.mesh().tdim() - 1);
  std::vector< std::vector<DataType> > cell_values(num_facets);
  
  for (size_t l = 0; l < num_facets; ++l) 
  {
    cell_values[l].resize(num_scalars, 0.);
  }

  assemble_multiple_scalar_boundary(space, local_asm, num_scalars, cell_values);
  
  integral.clear();
  integral.resize(num_scalars, 0.);

  for (size_t i = 0; i < num_facets; ++i) 
  {
    for (size_t l = 0; l < num_scalars; ++l) 
    {
      integral[l] += cell_values[i][l];
    }
  }
}

template < class DataType, int DIM >
template< class LocalAssembler>
void AllGlobalAssembler< DataType, DIM >::maximize_scalar_boundary(const VecSpace &space,
                                                                   LocalAssembler& local_asm,
                                                                   DataType &maximum) const 
{
  // TODO: actually the number of boundary facets is needed
  const size_t num_facets = space.mesh().num_entities(space.mesh().tdim() - 1);
  std::vector< DataType > cell_values(num_facets, 0.);

  assemble_scalar_boundary(space, local_asm, cell_values);
  maximum = *std::max_element(cell_values.begin(), cell_values.end());
}

template < class DataType, int DIM >
template< class LocalAssembler>
void AllGlobalAssembler< DataType, DIM >::assemble_scalar_boundary(const VecSpace &space,
                                                                   LocalAssembler& local_asm,
                                                                   std::vector< DataType > &vec) const 
{
  if (should_reset_assembly_target_) 
  {
    vec.clear();
    const size_t num_cells = space.mesh().num_entities(space.mesh().tdim() - 1);
    vec.resize(num_cells, 0.);
  }

  // TODO: how should vec be defined -> all facets or only boundary facets?
  // If the latter, then what ordering should be used?
  StandardBoundaryScalarAssembly< BoundaryAssemblyAlgorithm, DataType, DIM > assembly(space, vec);
  assembly.assemble(local_asm, this->fq_select_);
}

template < class DataType, int DIM >
template< class LocalAssembler>
void AllGlobalAssembler< DataType, DIM >::assemble_multiple_scalar_boundary(const VecSpace &space,
                                                                            LocalAssembler& local_asm,
                                                                            const int num_scalars,
                                                                            std::vector< std::vector< DataType > > &vec) const 
{
  if (should_reset_assembly_target_) 
  {
    vec.clear();
    const size_t num_facets = space.mesh().num_entities(space.mesh().tdim() - 1);
    vec.resize(num_facets);
    for (size_t l = 0; l < num_facets; ++l) 
    {
      vec[l].resize(num_scalars, 0.);
    }
  }

  // TODO: how should vec be defined -> all facets or only boundary facets?
  // If the latter, then what ordering should be used?
  StandardBoundaryMultipleScalarAssembly< BoundaryAssemblyAlgorithm, DataType, DIM > assembly(space, vec, num_scalars);
  assembly.assemble(local_asm, this->fq_select_);
}

template < class DataType, int DIM >
template< class LocalAssembler>
void AllGlobalAssembler< DataType, DIM >::integrate_multiple_scalar(const VecSpace &space,
                                                                    LocalAssembler& local_asm, 
                                                                    const int num_scalars,
                                                                    std::vector< DataType > &integral) const 
{
  const size_t num_cells = space.mesh().num_entities(space.mesh().tdim());
  std::vector< std::vector< DataType > > cell_values(num_cells);
  for (size_t l = 0; l < num_cells; ++l) {
    cell_values[l].resize(num_scalars, 0.);
  }

  assemble_multiple_scalar(space, local_asm, num_scalars, cell_values);
  integral.clear();
  integral.resize(num_scalars, 0.);

  for (size_t i = 0; i < num_cells; ++i) 
  {
    for (size_t l = 0; l < num_scalars; ++l) 
    {
      integral[l] += cell_values[i][l];
    }
  }
}

template < class DataType, int DIM >
template< class LocalAssembler>
void AllGlobalAssembler< DataType, DIM >::assemble_interface_multiple_scalar_cells(const VecSpace &space,
                                                                                   LocalAssembler& local_asm, 
                                                                                   const int num_scalars,
                                                                                   std::vector< LocalVector > &values) const 
{
  // Create interface list from mesh
  mesh::ConstMeshPtr mesh = &space.mesh(); // OK since pointer is intrusive
  mesh::InterfaceList if_list = mesh::InterfaceList::create(mesh);

  // Clear and create values data structure
  values.clear();
  values.resize(mesh->num_entities(mesh->tdim()));
  for (size_t j = 0; j < values.size(); ++j) 
  {
    values[j].resize(num_scalars, 0.);
  }
  
  InterfaceCellMultipleScalarAssembly<DataType,DIM> assem;
  assem.assemble (space, if_list, this->if_q_select_, local_asm, values);
}


} // namespace hiflow
#endif
