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

#ifndef HIFLOW_ASSEMBLY_DG_ASSEMBLY_H
#define HIFLOW_ASSEMBLY_DG_ASSEMBLY_H

#include "assembly/assembly_routines.h"
#include "assembly/assembly_routines_deprecated.h"
#include "assembly/global_assembler_deprecated.h"
#include "assembly/standard_assembly_deprecated.h"
#include "assembly/quadrature_selection.h"
#include "mesh/interface.h"
#include "quadrature/quadrature.h"

#include "common/log.h"
#include "common/array_tools.h"
#include "common/sorted_array.h"
#include "mesh/mesh.h"
#include <boost/function.hpp>
#include <functional>

/// \author Staffan Ronnas, Jonathan Schwegler, Simon Gawlok, Philipp Gerstner

namespace hiflow {

/// \author Staffan Ronnas, Jonathan Schwegler, Simon Gawlok

/// \brief Class for performing global assembly over interfaces for e.g.
/// Discontinuous Galerkin-Methods.

template < class DataType, int DIM >
class DGGlobalAssembler : public StandardGlobalAssembler< DataType, DIM > {
public:
  
  // TODO: this could be optimized by splitting into separate cell/facet
  // selection functions.
  typedef std::function < void ( const Element< DataType, DIM > &, const Element< DataType, DIM > &, 
                                 int, int,
                                 Quadrature< DataType > &, Quadrature< DataType > &) >
                                 IFQuadratureSelectionFun;
  
  typedef GlobalAssembler< DataType, DIM > GlobalAsm;
  typedef VectorSpace< DataType, DIM > VecSpace;

  DGGlobalAssembler();

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
                                 typename GlobalAsm::GlobalMatrix &matrix) const;

  ///
  /// \brief Assemble global vector for a VecSpace.
  ///
  /// \details Assembles a vector
  /// \f$b_i = \sum_{K \in \mathcal T_h} \int_{\partial K}{f(x, \varphi_i)dx}\f$
  /// over the interfaces defined by the mesh
  /// associated to a VecSpace. The integrand is defined through
  /// a InterfaceVectorAssemblyFun, which should return the locally
  /// assembled vector for each element.
  ///
  /// \param[in] space      the VecSpace for which the assembly is performed
  /// \param[in] local_asm  function or functor that performs local vector
  /// assembly \param[out] global_vector  the assembled vector \f$b_i\f$ \see
  /// concept_assembly
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
  ///                  typename GlobalAssembler< DataType, DIM >::LocalVector &vals)
  
  template<class LocalAssembler>
  void assemble_interface_vector(const VecSpace &space,
                                 LocalAssembler& local_asm,
                                 typename GlobalAsm::GlobalVector &vec) const;

  ///
  /// \brief Assemble global value for a VecSpace.
  ///
  /// \details Assembles a scalar
  /// \f$v_K = \int_{\partial K}{f(x, \varphi_i)dx}, \ K \in \mathcal T_h\f$
  /// over the interfaces defined by the mesh
  /// associated to a VecSpace. The integrand is defined through
  /// a InterfaceScalarAssemblyFun, which should return the locally
  /// assembled value for each element.
  ///
  /// \param[in] space      the VecSpace for which the assembly is performed
  /// \param[in] local_asm  function or functor that performs local vector
  /// assembly \param[out] values  the assembled values \f$v_K\f$ \see
  /// concept_assembly
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
  ///                  DataType &value)
                                   
  template<class LocalAssembler>
  void assemble_interface_scalar(const VecSpace &space,
                                 LocalAssembler& local_asm,
                                 std::vector< DataType > &values) const;

  ///
  /// \brief Assemble global value for a VecSpace.
  ///
  /// \details Assembles a scalar
  /// \f_K = \int_{\partial K}{f(x)dx}, \ K \in \mathcal T_h\f$
  /// over the interfaces defined by the mesh
  /// associated to a VecSpace. The integrand is defined through
  /// a InterfaceScalarAssemblyFun, which should return the locally
  /// assembled value for each interface.
  ///
  /// \param[in] space      the VecSpace for which the assembly is performed
  /// \param[in] local_asm  function or functor that performs local vector
  /// assembly \param[out] values  the assembled values \f$v_K\f$ correctly
  /// distributed to cells \see concept_assembly
  ///
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
  ///                  DataType &value)
                                   
  template<class LocalAssembler>
  void assemble_interface_scalar_cells(const VecSpace &space,
                                       LocalAssembler& local_asm,
                                       std::vector< DataType > &values) const;

  ///
  /// \brief Assemble global value for a VecSpace.
  ///
  /// \details Assembles multiple scalars
  /// \f_K = \int_{\partial K}{f(x)dx}, \ K \in \mathcal T_h\f$
  /// over the interfaces defined by the mesh
  /// associated to a VecSpace. The integrand is defined through
  /// a InterfaceMultipleScalarAssemblyFun, which should return the locally
  /// assembled values for each interface.
  ///
  /// \param[in] space      the VecSpace for which the assembly is performed
  /// \param[in] local_asm  function or functor that performs local vector
  /// assembly \param[in] num_scalars dimension of integrand \param[out] values
  /// the assembled values \f_K\f$ correctly distributed to cells \see
  /// concept_assembly
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
  ///                  std::vector<DataType> &vals)
                                   
  template<class LocalAssembler>
  void assemble_interface_multiple_scalar_cells(const VecSpace &space,
                                                LocalAssembler& local_asm, 
                                                const int num_scalars,
                                                std::vector< typename GlobalAsm::LocalVector > &values) const;

  ///
  /// \brief Add contributions of an interface scalar assembly to a vector of
  /// cell values in a naive way. This functionality is, e.g., needed in error
  /// estimators
  void distribute_interface_to_cell_values_naive(
      const VecSpace &space,
      std::vector< DataType > &cell_values,
      const std::vector< DataType > &interface_values) const;

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
  void
  set_interface_quadrature_selection_fun(IFQuadratureSelectionFun q_select);

private:


  IFQuadratureSelectionFun if_q_select_;
};

template < class DataType, int DIM >
DGGlobalAssembler< DataType, DIM >::DGGlobalAssembler()
    : if_q_select_(DefaultInterfaceQuadratureSelection< DataType, DIM >()) 
{
  // By default, we don't reset the target.
  this->should_reset_assembly_target(false);
}


template < class DataType, int DIM >
template< class LocalAssembler>
void DGGlobalAssembler< DataType, DIM >::assemble_interface_matrix( const VecSpace &space, 
                                                                    LocalAssembler& local_asm,
                                                                    typename GlobalAsm::GlobalMatrix &matrix) const 
{
  if (this->should_reset_assembly_target_) 
  {
    matrix.Zeros();
  }
  
  // Create interface list from mesh
  mesh::ConstMeshPtr mesh = &space.mesh(); // OK since pointer is intrusive
  mesh::InterfaceList if_list = mesh::InterfaceList::create(mesh);

  InterfaceMatrixAssembly<DataType,DIM> assem;
  assem.assemble (space, if_list, if_q_select_, local_asm, &matrix);
}

template < class DataType, int DIM >
template< class LocalAssembler>
void DGGlobalAssembler< DataType, DIM >::assemble_interface_vector(const VecSpace &space, 
                                                                   LocalAssembler& local_asm, 
                                                                   typename GlobalAsm::GlobalVector &vec) const 
{
  if (this->should_reset_assembly_target_) 
  {
    vec.Zeros();
  }

    // Create interface list from mesh
  mesh::ConstMeshPtr mesh = &space.mesh(); // OK since pointer is intrusive
  mesh::InterfaceList if_list = mesh::InterfaceList::create(mesh);

  InterfaceVectorAssembly<DataType,DIM> assem;
  assem.assemble (space, if_list, if_q_select_, local_asm, vec);
}

template < class DataType, int DIM >
template< class LocalAssembler>
void DGGlobalAssembler< DataType, DIM >::assemble_interface_scalar(const VecSpace &space, 
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
  assem.assemble (space, if_list, if_q_select_, local_asm, values);
}

template < class DataType, int DIM >
template< class LocalAssembler>
void DGGlobalAssembler< DataType, DIM >::assemble_interface_scalar_cells(const VecSpace &space, 
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
  assem.assemble (space, if_list, if_q_select_, local_asm, values);
}

template < class DataType, int DIM >
template< class LocalAssembler>
void DGGlobalAssembler< DataType, DIM >::assemble_interface_multiple_scalar_cells(const VecSpace &space,
                                                                                  LocalAssembler& local_asm, 
                                                                                  const int num_scalars,
                                                                                  std::vector< typename GlobalAsm::LocalVector > &values) const 
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
  assem.assemble (space, if_list, if_q_select_, local_asm, values);
}

template < class DataType, int DIM >
void DGGlobalAssembler< DataType, DIM >::distribute_interface_to_cell_values_naive(const VecSpace &space, 
                                                                                   std::vector< DataType > &cell_values,
                                                                                   const std::vector< DataType > &interface_values) const 
{
  // Create interface list from mesh
  mesh::ConstMeshPtr mesh = &space.mesh(); // OK since pointer is intrusive

  // Check compatibility of mesh and cell_values vector
  assert(mesh->num_entities(mesh->tdim()) == cell_values.size());

  mesh::InterfaceList if_list = mesh::InterfaceList::create(mesh);

  // Check compatibility of interface list and interface_values vector
  assert(if_list.size() == interface_values.size());

  // Loop over interfaces
  int i = 0;
  for (mesh::InterfaceList::const_iterator it = if_list.begin(),
                                     end_it = if_list.end();
       it != end_it; ++it) {
    int remote_index_master = -10;
    mesh->get_attribute_value("_remote_index_", mesh->tdim(),
                              it->master_index(), &remote_index_master);

    const int num_slaves = it->num_slaves();
    if (remote_index_master == -1) {
      if (num_slaves > 0) {
        // Master only gets half of the contribution of interface value
        cell_values[it->master_index()] += 0.5 * interface_values[i];
      } else {
        // boundary facet
        // Master only gets contribution of interface value
        cell_values[it->master_index()] += interface_values[i];
      }
    }

    // weight per slave
    DataType weight_slave = 0.5;
    if (num_slaves > 0) {
      weight_slave /= num_slaves;
    }
    // Loop over slaves
    for (int s = 0; s < num_slaves; ++s) {
      int remote_index_slave = -10;
      mesh->get_attribute_value("_remote_index_", mesh->tdim(),
                                it->slave_index(s), &remote_index_slave);

      if (remote_index_slave == -1) {
        cell_values[it->slave_index(s)] += weight_slave * interface_values[i];
      }
    }
    ++i;
  }
}

template < class DataType, int DIM >
void DGGlobalAssembler< DataType, DIM >::set_interface_quadrature_selection_fun( IFQuadratureSelectionFun q_select) 
{
  if_q_select_ = q_select;
}

} // namespace hiflow

#endif
