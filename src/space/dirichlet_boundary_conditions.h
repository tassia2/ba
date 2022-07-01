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

#ifndef HIFLOW_DIRICHLET_BOUNDARY_CONDITIONS_H
#define HIFLOW_DIRICHLET_BOUNDARY_CONDITIONS_H

#include <mpi.h>

#include <cassert>
#include <string>
#include <vector>

#include "common/vector_algebra_descriptor.h"
#include "dof/dof_fem_types.h"
#include "dof/dof_partition.h"
#include "dof/dof_impl/dof_container.h"
#include "fem/fe_mapping.h"
#include "mesh/iterator.h"
#include "mesh/mesh.h"
#include "mesh/types.h"
#include "mesh/writer.h"
#include "space/vector_space.h"
#include "space/element.h"

/// @author Eva Ketelaer, Staffan Ronnas, Philipp Gerstner

namespace hiflow {

namespace mesh{
class Entity;
};

/// \brief Finds the dofs and the coords on a given (boundary) facet.

/// \details Serves as a helper function for the function
/// compute_dirichlet_dofs_and_values.

/// \param [in] boundary_face     iterator on the boundary facets.
/// \param [in] space             space object containing the mesh and the FE
///                               approximation.
/// \param [in] var               variable for which the dirichlet dofs should
///                               be set. The function can be called several
///                               times for vector-valued problems.
/// \param [out] dof_ids          the id:s of the dofs on the facet.
/// \param [out] dof_points       the coordinates of the dof id:s.
template < class DataType, int DIM >
void find_dofs_on_face( mesh::Id boundary_id, 
                        const VectorSpace< DataType, DIM > &space, 
                        size_t fe_ind,
                        std::vector< doffem::DofID > &dof_ids,
                        int& local_face_number) 
{
  using namespace mesh;
  using namespace doffem;

  const Mesh &mesh = space.mesh();
  const TDim tdim = mesh.tdim();

  // check if the boundary face exists and get the location where the entity
  // number should be stored
  int face_number;
  const bool check = mesh.find_entity(tdim - 1, boundary_id, &face_number);
  assert(check);

  // Get the face to be able to access to the data associated with the face
  Entity face = mesh.get_entity(tdim - 1, face_number);

#ifndef NDEBUG
  // Get the cell associated with the cell and check if only one cell was
  // found
  IncidentEntityIterator dbg_cell = mesh.begin_incident(face, tdim);
  assert(dbg_cell != mesh.end_incident(face, tdim));
  assert(++dbg_cell == mesh.end_incident(face, tdim));
#endif

  // reset the cell because it was changed in the assert above
  IncidentEntityIterator cell = face.begin_incident(tdim);

  // loop over all faces of the cell to get the local face index for
  // identifying the dofs
  local_face_number = 0;
  for (IncidentEntityIterator global_face = cell->begin_incident(tdim - 1);
       global_face != cell->end_incident(tdim - 1); ++global_face) 
  {
    // if the global face id equals the boundary id the local face index is
    // found
    if (global_face->id() == boundary_id) 
    {
      break;
    }
    local_face_number++;
  }

  assert(local_face_number >= 0 &&
         local_face_number < cell->cell_type().num_regular_entities(tdim - 1));

  // Get dofs and coords
  const DofPartition< DataType, DIM > &dof = space.dof();

  dof.get_dofs_on_subentity(fe_ind, cell->index(), tdim - 1, local_face_number, dof_ids);
}

template < class DataType, int DIM >
void find_dofs_on_face( const mesh::EntityIterator &boundary_face,
                        const VectorSpace< DataType, DIM > &space, 
                        size_t fe_ind,
                        std::vector< doffem::DofID > &dof_ids) 
{
  int local_face_number = 0;
  find_dofs_on_face<DataType, DIM>(boundary_face->id(), space, fe_ind, dof_ids, local_face_number);
}


/// \brief Container for fitting a user-defined Dirichlet functor into 
/// the framework required by MappingPhys2Ref
template < class DataType, int DIM, class DirichletEvaluator >
class DirichletContainer 
{
public:
  using Coord = typename StaticLA<DIM, DIM, DataType>::ColVectorType;

  DirichletContainer (DirichletEvaluator* dirichlet_eval, size_t nb_comp)
  : dirichlet_eval_(dirichlet_eval),
  nb_comp_ (nb_comp)
  {
  }
  
  ~DirichletContainer()
  {
  } 
  
  void evaluate (const mesh::Entity& face, const Coord& pt, std::vector<DataType>& vals) const
  {
    assert (this->dirichlet_eval_ != nullptr);
    this->dirichlet_eval_->evaluate(face, pt, vals);
    assert (vals.size() == 0 || vals.size() == this->nb_comp());
  }
  
  size_t nb_comp() const 
  {
    return this->nb_comp_;
  }

  size_t nb_func() const 
  {
    return 1;  
  }
  
  size_t iv2ind(size_t j, size_t v) const 
  {
    assert (v < this->nb_comp_);
    return v;
  }
  
  size_t weight_size() const 
  {
    return this->nb_func() * this->nb_comp();
  }
  
private:
  DirichletEvaluator * dirichlet_eval_;
  size_t nb_comp_;

};

/// \brief Locate and evaluate Dirichlet boundary conditions.

/// \details The function loops over all boundary facets and calls a
/// user-provided function object (a functor) to obtain the values for
/// the Dirichlet dofs on that boundary facet. This functor has to
/// provide at least one function called evaluate, with the facet
/// entity and a vector with the coordinates on that facet, that
/// returns a std::vector<double> with the Dirichlet values for the
/// dofs on this boundary facet. If the dofs of the boundary should
/// not be constrained, an empty vector should be returned.
/// \param [in] dirichlet_eval    user-provided function object which computes
///                               the dirichlet values.
/// \param [in] space             space object containing the mesh and the FE
///                               approximation.
/// \param [in] var               variable for which the dirichlet dofs should
///                               be set. The function can be called several
///                               times for vector-valued problems.
/// \param [out] dirichlet_dofs   vector to which the indices of the dirichlet
///                               dofs are appended.
/// \param [out] dirichlet_values vector which the values of the dirichlet dofs
///                               are appended.

template < class DirichletEvaluator, class DataType, int DIM >
void compute_dirichlet_dofs_and_values( DirichletEvaluator &dirichlet_eval, 
                                        const VectorSpace< DataType, DIM > &space,
                                        size_t fe_ind, 
                                        std::vector< doffem::DofID > &dirichlet_dofs,
                                        std::vector< DataType > &dirichlet_values) 
{
  using namespace mesh;

  const Mesh &mesh = space.mesh();
  const TDim tdim = mesh.tdim();

  // extract boundary of mesh
  MeshPtr boundary_mesh = mesh.extract_boundary_mesh();

  int rank = -1, size = -1;
  MPI_Comm_rank(space.get_mpi_comm(), &rank);
  MPI_Comm_size(space.get_mpi_comm(), &size);

  const bool is_sequential = (size == 1);
  if (!is_sequential) 
  {
    assert(mesh.has_attribute("_sub_domain_", tdim));
  }
  
  size_t nb_comp = space.fe_2_var(fe_ind).size();
  
  DirichletContainer<DataType, DIM, DirichletEvaluator> dirichlet_func (&dirichlet_eval, nb_comp);
  
  // Loop over all faces which belong to the boundary
  for (EntityIterator it_boundary = boundary_mesh->begin(tdim - 1);
       it_boundary != boundary_mesh->end(tdim - 1); ++it_boundary) 
  {
    // get id of boundary face
    const Id boundary_id = it_boundary->id();

    // check if the boundary face exists and get the location
    // where the entity number should be stored
    int face_number;
    const bool check = mesh.find_entity(tdim - 1, boundary_id, &face_number);
    assert(check);

    // Get the face to be able to access to the data associated with the face
    Entity face = mesh.get_entity(tdim - 1, face_number);

    // reset the cell because it was changed in the assert above
    IncidentEntityIterator cell = face.begin_incident(tdim);

    std::vector< doffem::DofID > gl_dofs_on_face;
    int local_face_number;
    
    // Get the global dofs indices attached to the face
    find_dofs_on_face<DataType, DIM>(boundary_id, space, fe_ind, gl_dofs_on_face, local_face_number);

    // get reference element
    auto ref_fe = space.fe_manager().get_fe(cell->index(), fe_ind);
    
    // Get dof container
    auto dofs = ref_fe->dof_container();
    
    // get the local dof ids attached to the face
    const std::vector< doffem::DofID > loc_dofs_on_face = dofs->get_dof_on_subentity(tdim - 1, local_face_number ); 
    
    assert (loc_dofs_on_face.size() == gl_dofs_on_face.size());
    
    // create object that maps the user-defined BC evaluator dirichlet_eval to 
    // a function defined on the reference cell. This object is needed for evaluating
    // the dof functionals
    doffem::MappingPhys2Ref < DataType, DIM, DirichletContainer<DataType, DIM, DirichletEvaluator> > * ref_cell_eval
      = new doffem::MappingPhys2Ref < DataType, DIM, DirichletContainer<DataType, DIM, DirichletEvaluator> > (&dirichlet_func, 
                                                                                                              &face, 
                                                                                                              ref_fe->fe_trafo(),
                                                                                                              space.get_cell_transformation(cell->index()));

    // evaluate dof functional attached to face for dirichlet_eval as input
    std::vector< std::vector<DataType> > values_on_face;
    dofs->evaluate (ref_cell_eval, loc_dofs_on_face, values_on_face);
    
    // get dof factors -> should not be necessary, since no slave cell for boundary facets
    Element<DataType, DIM> elem (space, cell->index());
    std::vector< DataType > dof_factors;
    space.dof().get_dof_factors_on_cell(cell->index(), dof_factors);
  
    const size_t start_dof = elem.dof_offset(fe_ind);
    
    //const std::vector< DataType > values_on_face = dirichlet_eval.evaluate(*it_boundary, coords_on_face);
    if (!values_on_face.empty()) 
    {
      assert (values_on_face.size() == loc_dofs_on_face.size());
      
      // If non-empty vector was returned, insert into output vectors
      // vectors to filter out only dofs that belong to our subdomain
      std::vector< doffem::DofID > dofs_on_face_checked;
      std::vector< DataType > values_on_face_checked;

      dofs_on_face_checked.reserve(gl_dofs_on_face.size());
      values_on_face_checked.reserve(gl_dofs_on_face.size());

      int k = 0;
      for (std::vector< doffem::DofID >::iterator dof_it = gl_dofs_on_face.begin();
           dof_it != gl_dofs_on_face.end(); ++dof_it) 
      {
        if (space.dof().owner_of_dof(*dof_it) == rank) 
        {
          assert (values_on_face[k].size() == 1);
          DataType cur_dof_value = values_on_face[k][0];
          
          // sort out non-Dirichlet evaluations
          if (!std::isnan(cur_dof_value))
          {
            cur_dof_value /= dof_factors[start_dof + loc_dofs_on_face[k]];
            dofs_on_face_checked.push_back(*dof_it);
            values_on_face_checked.push_back(cur_dof_value);
          }
        }
        ++k;
      }

      dirichlet_dofs.insert(dirichlet_dofs.end(), 
                            dofs_on_face_checked.begin(),
                            dofs_on_face_checked.end());
      dirichlet_values.insert(dirichlet_values.end(),
                              values_on_face_checked.begin(),
                              values_on_face_checked.end());
    }
  }
}

template < class DataType, int DIM >
void compute_dirichlet_dofs_1D( const VectorSpace< DataType, DIM > &space,
                                size_t fe_ind, 
                                std::vector< doffem::DofID > &dirichlet_dofs)
{
  assert (DIM == 1);
  
  // Loop over all cells.
  for (mesh::EntityIterator facet_it = space.mesh().begin(DIM - 1),
                      facet_end = space.mesh().end(DIM - 1);
                      facet_it != facet_end; ++facet_it) 
  {
    // Returns the number of neighbors for each cell, to check if it is on the facet.
    const mesh::EntityCount num_cell_neighbors = facet_it->num_incident_entities(DIM);

    // If it lies on the facet, the corresponding DOF is a Dirichlet DOF and
    // is added to dirichlet_dofs_.
    if (num_cell_neighbors == 1) 
    {
      std::vector< int > dof_number_;
      space.dof().get_dofs_on_subentity( fe_ind, facet_it->begin_incident(DIM)->index(), 0, facet_it->index(), dof_number_);
      dirichlet_dofs.push_back(dof_number_[0]);
    }
  }
}
} // namespace hiflow

#endif
