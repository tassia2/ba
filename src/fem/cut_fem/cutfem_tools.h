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

/// \author Philipp Gerstner

#ifndef HIFLOW_CUTFEM_TOOLS_H_
#define HIFLOW_CUTFEM_TOOLS_H_

#include "common/bbox.h"
#include "common/log.h"
#include "common/vector_algebra_descriptor.h"
#include "common/array_tools.h"
#include "common/parcom.h"
#include "fem/cell_trafo/cell_transformation.h"
#include "fem/cut_fem/cell_cut.h"
#include "mesh/cell_type.h"
#include "mesh/entity.h"
#include "mesh/geometric_tools.h"
#include "mesh/types.h"
#include "mesh/iterator.h"
#include "space/vector_space.h"

#include <map>
#include <cmath>
#include <vector>
#include <numeric>
#include <set>

namespace hiflow {

namespace doffem {

template <class LAD, int DIM, class DomainFunction>
class CutFeEvaluator
{
  typedef typename LAD::VectorType VectorType;
  typedef typename LAD::DataType DataType;
  using Coord = typename StaticLA<DIM, DIM, DataType>::ColVectorType;
  
public:
  CutFeEvaluator()
  : space_(nullptr), Xi_(nullptr), cell_domain_(nullptr), initialized_(false), nb_comp_(0)
  {
  }

  void init (const VectorSpace< DataType, DIM > &space, 
             const VectorType& coeff,
             const std::vector<int>& cell_domain,
             const DomainFunction& xi,
             const DataType dist_thresh,
             const int fe_ind_for_lower_xi,
             const int fe_ind_for_higher_xi,
             const int cell_type_for_lower_xi,
             const int cell_type_for_higher_xi
             )
  {
    assert (fe_ind_for_lower_xi != fe_ind_for_higher_xi);

    space_ = &(space);
    cell_domain_ = &(cell_domain);
    Xi_ = &(xi);
    dist_thresh_ = dist_thresh;
    
    cell_type_lower_ = cell_type_for_lower_xi;
    cell_type_higher_ = cell_type_for_higher_xi;

    std::vector<size_t> vars_lower = space.fe_2_var(fe_ind_for_lower_xi);
    std::vector<size_t> vars_higher = space.fe_2_var(fe_ind_for_higher_xi);
    assert (vars_lower.size() == vars_higher.size());
    nb_comp_ = vars_lower.size();
    
    fe_eval_lower_.reset();
    fe_eval_higher_.reset();
    fe_eval_lower_ = std::shared_ptr< FeEvalCell<DataType, DIM> >(new FeEvalCell<DataType, DIM>(space, coeff, fe_ind_for_lower_xi));
    fe_eval_higher_ = std::shared_ptr< FeEvalCell<DataType, DIM> >(new FeEvalCell<DataType, DIM>(space, coeff, fe_ind_for_higher_xi));

    initialized_ = true;
  }
  
  bool is_initialized() const 
  {
    return this->initialized_;
  }
  
  void evaluate(const Entity &cell, 
                const Coord &pt,
                std::vector< DataType > &values) const 
  {
    const int cell_index = cell.index();
    assert (cell_index < space_->fe_manager().nb_cell_trafos());
    CCellTrafoSPtr<DataType, DIM> c_trafo = space_->get_cell_transformation(cell_index);
    
    Coord ref_pt;
    c_trafo->inverse(pt, ref_pt);
    this->r_evaluate(cell, ref_pt, values);
  }

  void r_evaluate(const Entity &cell, 
                  const Coord &ref_pt,
                  std::vector< DataType > &values) const 
  {
    values.resize(nb_comp_, 0.);
    const int gdim = cell.gdim();
    const int cell_index = cell.index();
    const int cell_type = this->cell_domain_->at(cell_index);
    assert (cell_index < space_->fe_manager().nb_cell_trafos());
    CCellTrafoSPtr<DataType, DIM> c_trafo = space_->get_cell_transformation(cell_index);
    
    if (cell_type == cell_type_lower_)
    {
      // \{xi < thresh\}
      fe_eval_lower_->r_evaluate(cell, ref_pt, values);
      return;
    }
    else if (cell_type == cell_type_higher_)
    {
      // \{xi > thresh\}
      fe_eval_higher_->r_evaluate(cell, ref_pt, values);
      return;
    }
    
    assert (cell_type == 0);
    // interface cell
    
    Coord pt;
    c_trafo->transform(ref_pt, pt);
    DataType chi = this->Xi_->evaluate(cell, pt);
    
    if (chi < dist_thresh_)
    {
      fe_eval_lower_->r_evaluate(cell, ref_pt, values);
    }
    else
    {
      fe_eval_higher_->r_evaluate(cell, ref_pt, values);
    }
  }
  
  size_t nb_comp() const 
  {
    return nb_comp_;
  }

  const VectorSpace< DataType, DIM >* space_;
  const std::vector<int>* cell_domain_;
  const DomainFunction* Xi_;
  int cell_type_lower_;
  int cell_type_higher_;   
  std::shared_ptr< FeEvalCell<DataType, DIM> > fe_eval_lower_;
  std::shared_ptr< FeEvalCell<DataType, DIM> > fe_eval_higher_;
  int nb_comp_;
  DataType dist_thresh_;
  bool initialized_;
};

// interface_dofs[i]        = indices of dofs located at domain i + interface cells
// overlap_dofs[i]          = indices of dofs located at domain i + interface cells + overlap cells i
// dof_iface_ext            = union{i=1,2} [indices of dofs located outside (domain i + interface cells) ]
// dof_overlap_ext          = union{i=1,2} [indices of dofs located outside (domain i + interface cells + overlap cells i ) ]
// dof_overlap_ext_with_bdy = union{i=1,2} [indices of dofs located outside (domain i + interface cells + overlap cells i ) ]
//                          + union{i=1,2} [indices of dofs located at interface between domain 1-i and overlap cells i ]

template <class DataType, int DIM>
void determine_cutfem_dof_domain (const VectorSpace<DataType, DIM>& space,
                                  const std::vector<int>& ext_cell_domain,
                                  const std::vector< std::vector< int > >& fe_ind,
                                  const std::map<int, std::vector<int> >& celltype_2_fixed,
                                  std::vector<SortedArray<int> >& interface_dofs,
                                  std::vector<SortedArray<int> >& overlap_dofs,
                                  SortedArray< int >& dofs_iface_ext,
                                  SortedArray< int >& dofs_iface_ext_with_bdy,
                                  SortedArray< int >& dofs_overlap_ext,
                                  SortedArray< int >& dofs_overlap_ext_with_bdy)
{
  const int num_domain = fe_ind.size();
  const int num_fe = fe_ind[0].size();
  for (int i=0; i!=num_domain; ++i)
  {
    assert(fe_ind[i].size() == num_fe);
  }

  dofs_iface_ext.clear();
  dofs_iface_ext_with_bdy.clear();
  dofs_overlap_ext.clear();
  dofs_overlap_ext_with_bdy.clear();
  
  interface_dofs.clear();
  interface_dofs.resize(num_domain);
  overlap_dofs.clear();
  overlap_dofs.resize(num_domain);
  
  const int num_cell = space.meshPtr()->num_entities(space.meshPtr()->tdim());  
  assert (ext_cell_domain.size() == num_cell);
    
  int num_interface_cells = 0;
  int num_overlap_cells = 0;
  for (int c=0; c<num_cell; ++c)
  {
    // determine type of cell
    const int cell_type = ext_cell_domain[c];
    
    // collect dofs on interface cells
    if (cell_type == 0)
    {
      num_interface_cells++;
    }
    if ((cell_type == 0) || (cell_type == 2) || (cell_type == -2))
    {
      num_overlap_cells++;
    }
  }
  const auto num_dofs_per_cell = space.nb_dof_on_cell(0);
  for (int i=0; i!=num_domain; ++i)
  {
    interface_dofs[i].reserve(num_dofs_per_cell * num_interface_cells);
    overlap_dofs[i].reserve(num_dofs_per_cell * num_overlap_cells);
  }

  dofs_iface_ext.reserve(space.nb_dofs_local());
  dofs_iface_ext_with_bdy.reserve(space.nb_dofs_local());
  dofs_overlap_ext.reserve(space.nb_dofs_local());
  dofs_overlap_ext_with_bdy.reserve(space.nb_dofs_local());
   
  // projection on solid domain / interface and incident cells
  // loop through all cells
  std::vector<gDofId> dof_ind; 
  ParCom parcom(space.get_mpi_comm());
    
  for (int c=0; c<num_cell; ++c)
  {
    // determine type of cell
    const int cell_type = ext_cell_domain[c];
    
    Entity cell = space.meshPtr()->get_entity(DIM, c);

    bool print = false;
          
    // collect dofs on overlap cells
    if ((cell_type == 0) || (cell_type == 2) || (cell_type == -2))
    {
      // interface
      for (int i=0; i<num_domain; ++i)
      {
        for (int fe = 0; fe != num_fe; ++fe)
        {
          dof_ind.clear();
          space.get_dof_indices(fe_ind[i][fe], c, dof_ind);
          const int num_dof = dof_ind.size();
          
          if (print)
          {
            std::cout << " rank " << parcom.rank() << " dofs " << string_from_range(dof_ind.begin(), dof_ind.end()) << std::endl;
            for (int l=0; l!=num_dof; ++l)
            {
              std::cout << " rank " << parcom.rank() << " dof id " << dof_ind[l] << " owner " << space.dof().owner_of_dof(dof_ind[l]) << std::endl;
            }
          }
          
          for (int l=0; l!=num_dof; ++l)
          {
            overlap_dofs[i].find_insert(dof_ind[l]);
          }
          if (cell_type == 0)
          {
            for (int l=0; l!=num_dof; ++l)
            {
              interface_dofs[i].find_insert(dof_ind[l]);
            }
          }
        }
      }
    }  
  }
    
  // exchange dof indices with neighboring sub domains
  std::vector< std::vector< std::vector<gDofId> > > neighbor_interface_dofs(num_domain);
  std::vector< std::vector< std::vector<gDofId> > > neighbor_overlap_dofs(num_domain);
  
  int num_proc = parcom.size();
  for (int i=0; i!=num_domain; ++i)
  {
    neighbor_interface_dofs[i].resize(num_proc);
    neighbor_overlap_dofs[i].resize(num_proc);
  }

/*
  int max_dof = -1;
  int min_dof = 1e9;
  for (int c=0; c<num_cell; ++c)
  {
    // interface
    for (int i=0; i<2; ++i)
    {
      for (int fe = 0; fe != num_fe; ++fe)
      {
        dof_ind.clear();
        space.get_dof_indices(fe_ind[i][fe], c, &dof_ind);
          
        const int num_dof = dof_ind.size();
        for (int l=0; l!=num_dof; ++l)
        {
          if (dof_ind[l] > max_dof)
          {
            max_dof = dof_ind[l];
          }
          if (dof_ind[l] < min_dof)
          {
            min_dof = dof_ind[l];
          }
        }
      }
    }
  }  
  std::cout << parcom.rank() << " : dof range " << min_dof << " - " << max_dof << std::endl;
*/

  for (int i=0; i!=num_domain; ++i)
  {
    //std::cout << " rank " << parcom.rank() << " nb dofs overlap " << i << " before comm " << overlap_dofs[i].size() << std::endl;
    
    parcom.broadcast_data_to_neighbors<gDofId> (space.meshPtr(),
                                             interface_dofs[i].data(),
                                             neighbor_interface_dofs[i]);
                                             
    parcom.broadcast_data_to_neighbors<gDofId> (space.meshPtr(),
                                             overlap_dofs[i].data(),
                                             neighbor_overlap_dofs[i]);
                                             
    for (int p=0; p!=num_proc; ++p)
    {
      for (int l=0, e_l = neighbor_interface_dofs[i][p].size(); l != e_l; ++l)
      {
        interface_dofs[i].find_insert(neighbor_interface_dofs[i][p][l]);
      }
      for (int l=0, e_l = neighbor_overlap_dofs[i][p].size(); l != e_l; ++l)
      {
        overlap_dofs[i].find_insert(neighbor_overlap_dofs[i][p][l]);
      }
    }
    //std::cout << " rank " << parcom.rank() << " nb dofs overlap " << i << " after comm " << overlap_dofs[i].size() << std::endl;
  }
    
  // determine dofs to be set to zero
  std::vector<int> fixed_domain;
  
  for (int c=0; c<num_cell; ++c)
  {
    // determine type of cell
    const int cell_type = ext_cell_domain[c];
    
    Entity cell = space.meshPtr()->get_entity(DIM, c);
    const bool print = false;
    
    bool on_overlap = false;
    switch(cell_type)
    {
      case 1:
        // domain 1 -> set solution for variable 2 to zero
        // i = 1;
        on_overlap = false;
        break;
      case 2:
        // domain 1 -> set solution for variable 2 to zero
        // i = 1;
        on_overlap = true;
        break;
      case -1:
        // domain 2 -> set solution for variable 1 to zero
        // i = 0;
        on_overlap = false;
        break;
      case -2:
        // domain 2 -> set solution for variable 1 to zero
        // i = 0;
        on_overlap = true;
        break;
      case 0:
        continue;
        break;
      default:
        assert(false);
        break;
    }
    
    assert (cell_type != 0);

    auto search = celltype_2_fixed.find(cell_type);
    if (search == celltype_2_fixed.end())
    {
      continue;
    }
    if (search->second.size() == 0)
    {
      continue;
    }

    for (auto i : search->second)
    {
      assert (i >= 0);
      assert (i < num_domain);
    
      for (int fe = 0; fe != num_fe; ++fe)
      {
        dof_ind.clear();
        space.get_dof_indices(fe_ind[i][fe], c, dof_ind);
    
        const int num_dof = dof_ind.size();
        for (int l=0; l!=num_dof; ++l)
        {
          int pos;
          const auto gl_dof = dof_ind[l];

          if (print)
          {
            std::cout << " rank " << parcom.rank() << " cell " << c 
                      << " cell type " << cell_type 
                      << " domain " << i << " / " << num_domain 
                      << " fe " << fe_ind[i][fe] 
                      << " dof " << gl_dof 
                      << " is_local " << space.dof().is_dof_on_subdom(gl_dof) 
                      << " on overlap " << on_overlap 
                      << " found in overlap " << overlap_dofs[i].find(gl_dof, &pos)
                      << std::endl;
          }
          if (!space.dof().is_dof_on_subdom(gl_dof)) 
          {
            // skip non-local dofs
            continue;
          }           

          if (!on_overlap)
          {
            if (!(overlap_dofs[i].find(gl_dof, &pos)))
            {
              // dof does not belong to overlap cell
              dofs_overlap_ext.find_insert(gl_dof);
            }
            dofs_overlap_ext_with_bdy.find_insert(gl_dof);
          }
          if (!(interface_dofs[i].find(gl_dof, &pos)))
          {
            // dof does not belong to interface cell
            dofs_iface_ext.find_insert(gl_dof);
          }
          dofs_iface_ext_with_bdy.find_insert(gl_dof);

          //std::cout << dofs_overlap_ext.size() << " " << dofs_overlap_ext_with_bdy.size() 
          //          << " " << dofs_iface_ext.size() << " " << dofs_iface_ext_with_bdy.size() << std::endl;
        }
      }
    }
  }
}

template <class DataType, int DIM>
void determine_cutfem_dof_domain (const VectorSpace<DataType, DIM>& space,
                                  const std::vector<int>& ext_cell_domain,
                                  const std::vector< std::vector< int > >& fe_ind,
                                  std::vector<SortedArray<int> >& interface_dofs,
                                  std::vector<SortedArray<int> >& overlap_dofs,
                                  SortedArray< int >& dofs_iface_ext,
                                  SortedArray< int >& dofs_iface_ext_with_bdy,
                                  SortedArray< int >& dofs_overlap_ext,
                                  SortedArray< int >& dofs_overlap_ext_with_bdy)
{
  std::map<int, std::vector<int> > celltype_2_fixed;
  std::vector<int> domain_0(1,0);
  std::vector<int> domain_1(1,1);
  celltype_2_fixed[1] = domain_1;
  celltype_2_fixed[2] = domain_1;
  celltype_2_fixed[-1] = domain_0;
  celltype_2_fixed[-2] = domain_0;

  determine_cutfem_dof_domain(space, ext_cell_domain, fe_ind, celltype_2_fixed,
                              interface_dofs, overlap_dofs, dofs_iface_ext,
                              dofs_iface_ext_with_bdy, dofs_overlap_ext, dofs_overlap_ext_with_bdy);
}

} // namespace mesh
} // namespace hiflow

#endif
