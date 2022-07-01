// Copyright (C) 2011-2020 Vincent Heuveline
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
#ifndef _HP_ASSEMBLY_H_
#define _HP_ASSEMBLY_H_

#include "assembly/standard_assembly_deprecated.h"
#include "common/log.h"
#include "common/pointers.h"
#include "common/sorted_array.h"
#include "dof/dof_interpolation.h"
#include "mesh/types.h"
#include "space/element.h"
#include "space/vector_space.h"
#include <utility>
#include <vector>

//#define OCTAVE_OUTPUT

namespace hiflow {


// using doffem::DofInterpolation;
using namespace doffem;

/// hp-FEM Global Assembly strategy
template < class DataType, int DIM >
class HpFemAssembler : public StandardGlobalAssembler< DataType, DIM > {

  typedef GlobalAssembler< DataType, DIM > GlobalAsm;
  typedef VectorSpace< DataType, DIM > VecSpace;

  virtual void assemble_vector_impl(
      const VecSpace &space,
      typename GlobalAsm::VectorAssemblyFunction local_asm,
      typename GlobalAsm::GlobalVector &vec,
      typename GlobalAsm::QuadratureSelectionFunction q_select) const;

  virtual void assemble_matrix_impl(
      const VecSpace &space,
      typename GlobalAsm::MatrixAssemblyFunction local_asm,
      typename GlobalAsm::GlobalMatrix &mat,
      typename GlobalAsm::QuadratureSelectionFunction q_select) const;

};

//////////////////////////////////////////////////////////////////////////////////
//////////////// Implementation of HpFemAssembler
///////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////

template < class DataType, int DIM >
void HpFemAssembler< DataType, DIM >::assemble_vector_impl(
    const VecSpace &space,
    typename GlobalAsm::VectorAssemblyFunction local_asm,
    typename GlobalAsm::GlobalVector &vec,
    typename GlobalAsm::QuadratureSelectionFunction q_select)
    const {
  vec.begin_update();
  HpVectorAssembly< DataType, DIM > assembly(space, vec);
  assembly.assemble(local_asm, q_select);
  vec.end_update();
}

template < class DataType, int DIM >
void HpFemAssembler< DataType, DIM >::assemble_matrix_impl(
    const VecSpace &space,
    typename GlobalAsm::MatrixAssemblyFunction local_asm,
    typename GlobalAsm::GlobalMatrix &mat,
    typename GlobalAsm::QuadratureSelectionFunction q_select)
    const {
  mat.begin_update();
  HpMatrixAssembly< DataType, DIM > assembly(space, mat);
  assembly.assemble(local_asm, q_select);
  mat.end_update();
}

} // hiflow namespace
//////////////// OLD CODE ////////////////

#if 0

///
/// \brief Assemble global vector for a VecSpace.
///
/// \details Assembles a vector
/// \f$b_i = \int_{\Omega}{f(x, \varphi_i)dx}\f$
/// over the domain defined by the mesh
/// associated to a VecSpace. The integrand is defined through
/// the LocalIntegrator object, whose assemble_local_vector(const
/// Element&, LocalVector&) function should return the locally
/// assembled vector for each element.
///
/// \param[in] space      the VecSpace for which the assembly is performed
/// \param[in] local_int  functor that performs local vector assembly
/// \param[out] global_vector  the assembled vector \f$b_i\f$
/// \see concept_assembly
///

template<class LocalAssembler, class VectorType>
void assemble_vector ( const VecSpace& space,
                       LocalAssembler& local_asm,
                       VectorType& global_vector )
{
    std::ofstream octave ( "check_assembly.m", std::ios_base::app );
    octave.precision ( 16 );

    LOG_INFO ( "assembly", "\n=> Start vector assembly" );
    typedef VecSpace::MeshEntityIterator CellIterator;
    using hiflow::la::MatrixEntry;
    using hiflow::la::MatrixEntryList;

    global_vector.Zeros ( );
    std::vector<int> dof;
    LocalVector lv;

    std::vector<int> traversal_order;
    sort_elements ( space, traversal_order );
    const int num_elements = traversal_order.size ( );

    const FEType* prev_fe_type = 0;
    Quadrature<double> quadrature;

    const DofInterpolation& interp = space.dof ( ).dof_interpolation ( );

    for ( int e = 0; e < num_elements; ++e )
    {
        const int elem_index = traversal_order[e];
        Element elem ( space, elem_index );
        const FEType* fe_type = elem.get_fe_type ( 0 );

        if ( prev_fe_type == 0 || fe_type->get_my_id ( ) != prev_fe_type->get_my_id ( ) )
        {
            // TODO: Now chooses quadrature based on fe type of first variable -- improve this
            choose_quadrature ( *fe_type, quadrature );
            prev_fe_type = fe_type;
        }

        // get global dof indices
        elem.get_dof_indices ( dof );
        const int num_dofs = dof.size ( );

        // assemble locally
        lv.clear ( );
        lv.resize ( num_dofs, 0. );
        local_asm.initialize_for_element ( elem, quadrature );
        local_asm.assemble_local_vector ( elem, lv );

        octave << "% Element " << elem_index << "\n";
        octave << "dof = [" << string_from_range ( dof.begin ( ), dof.end ( ) ) << "] + 1;\n"
                << "b_local = [" << precise_string_from_range ( lv.begin ( ), lv.end ( ) ) << "]';\n"
                << "b(dof) += b_local;\n";

        LOG_INFO ( "assembly", "Element " << elem_index << "\n"
                   << "Dofs = " << string_from_range ( dof.begin ( ), dof.end ( ) ) << "\n"
                   << "lv = " << string_from_range ( lv.begin ( ), lv.end ( ) ) << "\n\n" );

        // assemble into global system
        for ( int i = 0; i < num_dofs; ++i )
        {
            DofInterpolation::const_iterator it = interp.find ( dof[i] );
            if ( it != interp.end ( ) )
            {
                // dof[i] is constrained -> add contributions to dependent dofs
                for ( ConstraintIterator c_it = it->second.begin ( ),
                      c_end = it->second.end ( ); c_it != c_end; ++c_it )
                {
                    if ( space.dof ( ).is_dof_on_subdom ( c_it->first ) )
                    {
                        global_vector.Add ( c_it->first, c_it->second * lv[i] );
                    }
                }
            }
            else
            {
                // dof[i] is unconstrained -> add contribution to this dof
                if ( space.dof ( ).is_dof_on_subdom ( dof[i] ) )
                {
                    global_vector.Add ( dof[i], lv[i] );
                }
            }
        }
    }
    LOG_INFO ( "assembly", "\n=> End vector assembly" );
    octave << "\n\n";
    octave.close ( );
}

///
/// \brief Assemble global matrix for a VecSpace.
///
/// \details Assembles a matrix
/// \f$A_{ij} = \int_{\Omega}{f(x,\varphi_i, \varphi_j)dx}\f$
/// over the domain defined by the mesh
/// associated to a VecSpace. The integrand is defined through
/// the LocalIntegrator object, whose assemble_local_matrix(const
/// Element&, LocalVector&) function should return the locally
/// assembled vector for each element.
///
/// \param[in] space      the VecSpace for which the assembly is performed
/// \param[in] local_int  functor that performs local vector assembly
/// \param[out] global_matrix  the assembled matrix \f$A_{ij}\f$
/// \see concept_assembly
///

template<class LocalAssembler, class MatrixType>
void assemble_matrix ( const VecSpace& space,
                       LocalAssembler& local_asm,
                       MatrixType& global_matrix )
{
    std::ofstream octave ( "check_assembly.m", std::ios_base::app );
    octave.precision ( 16 );
    typedef VecSpace::MeshEntityIterator CellIterator;
    const int dim = space.get_dim ( );

    const DofInterpolation& interp = space.dof ( ).dof_interpolation ( );

    global_matrix.Zeros ( );

    octave << "% ==== Global matrix assembly ====\n";
    octave << "A = zeros(" << global_matrix.nrows_global ( ) << ");\n";

    std::vector<int> dof;
    LocalMatrix lm;

    std::vector<int> traversal_order;
    sort_elements ( space, traversal_order );

    //     std::cout << "Element order = " << string_from_range(traversal_order.begin(), traversal_order.end()) << "\n";

    const int num_elements = traversal_order.size ( );

    // TODO: It would be nice to be able to iterate like this...
    // for (ElementIterator it = space.begin(); it != space.end(); ++it) {
    const FEType* prev_fe_type = 0;
    Quadrature<double> quadrature;

    for ( int e = 0; e < num_elements; ++e )
    {
        const int elem_index = traversal_order[e];
        Element elem ( space, elem_index );

        // TODO: Now chooses quadrature based on fe type of first variable -- improve this
        const FEType* fe_type = elem.get_fe_type ( 0 );

        if ( prev_fe_type == 0 || fe_type->get_my_id ( ) != prev_fe_type->get_my_id ( ) )
        {
            choose_quadrature ( *fe_type, quadrature );
            prev_fe_type = fe_type;
        }

        // get global dof indices
        elem.get_dof_indices ( dof );
        const int num_dofs = dof.size ( );

        // assemble locally
        lm.Resize ( num_dofs, num_dofs );
        lm.Zeros ( );
        local_asm.initialize_for_element ( elem, quadrature );
        local_asm.assemble_local_matrix ( elem, lm );

        octave << "% Element " << elem_index << "\n";
        octave << "dof = [" << string_from_range ( dof.begin ( ), dof.end ( ) ) << "] + 1;\n"
                << "A_local = " << lm << ";\n"
                << "A(dof, dof) += A_local;\n";

        LOG_INFO ( "assembly", "Element " << elem_index << "\n"
                   << "Dofs = " << string_from_range ( dof.begin ( ), dof.end ( ) ) << "\n"
                   << "lm = " << lm << "\n\n" );

        // Assemble into global system.  Only add entries to
        // unconstrained rows and columns, and only if the dof
        // corresponding to the row belongs to the local subdomain.
        for ( int i = 0; i < num_dofs; ++i )
        {
            DofInterpolation::const_iterator it_i = interp.find ( dof[i] );

            if ( it_i != interp.end ( ) )
            {
                // dof[i] is constrained -> add contributions to dependent rows
                for ( ConstraintIterator c_it = it_i->second.begin ( ),
                      c_end = it_i->second.end ( ); c_it != c_end; ++c_it )
                {
                    if ( space.dof ( ).is_dof_on_subdom ( c_it->first ) )
                    {

                        for ( int j = 0; j < num_dofs; ++j )
                        {
                            DofInterpolation::const_iterator it_j = interp.find ( dof[j] );

                            if ( it_j != interp.end ( ) )
                            {
                                // dof[j] is constrained -> add attributions to dependent columns
                                // TODO: are these not cleared at the end anyway?
                                for ( ConstraintIterator c2_it = it_j->second.begin ( ),
                                      c2_end = it_j->second.end ( ); c2_it != c2_end; ++c2_it )
                                {
                                    global_matrix.Add ( c_it->first,
                                                        c2_it->first,
                                                        c_it->second * c2_it->second * lm ( i, j ) );
                                }
                            }
                            else
                            {
                                // dof[j] unconstrained -> add contribution to dof[j] column
                                global_matrix.Add ( c_it->first,
                                                    dof[j],
                                                    c_it->second * lm ( i, j ) );
                            }
                        }
                    }
                }
            }
            else
            {
                // dof[i] is unconstrained
                if ( space.dof ( ).is_dof_on_subdom ( dof[i] ) )
                {
                    for ( int j = 0; j < num_dofs; ++j )
                    {
                        DofInterpolation::const_iterator it_j = interp.find ( dof[j] );
                        if ( it_j != interp.end ( ) )
                        {
                            for ( ConstraintIterator c_it = it_j->second.begin ( ),
                                  c_end = it_j->second.end ( ); c_it != c_end; ++c_it )
                            {
                                // dof[j] is constrained -> add attributions to dependent columns
                                global_matrix.Add ( dof[i],
                                                    c_it->first,
                                                    c_it->second * lm ( i, j ) );
                            }
                        }
                        else
                        {
                            // dof[j] unconstrained - assemble normally
                            global_matrix.Add ( dof[i], dof[j], lm ( i, j ) );
                        }
                    }
                }
            }
        }
    }

    // Set rows of constrained dofs to identity to obtain non-singular
    // matrix
    std::vector<int> constrained_dofs;
    for ( DofInterpolation::const_iterator it = interp.begin ( ), end = interp.end ( );
          it != end; ++it )
    {

        if ( space.dof ( ).is_dof_on_subdom ( it->first ) )
        {
            constrained_dofs.push_back ( it->first );
        }
    }

    {
        const int DEBUG_LEVEL = 3;
        LOG_DEBUG ( 3, "Constrained dofs in assemble_matrix() =\n"
                    << string_from_range ( constrained_dofs.begin ( ), constrained_dofs.end ( ) ); )
    }

    if ( !constrained_dofs.empty ( ) )
    {
        global_matrix.ZeroRows ( &constrained_dofs.front ( ), constrained_dofs.size ( ), 1. );
    }

    octave << "\n\n";
    octave.close ( );
}
#endif
#endif
