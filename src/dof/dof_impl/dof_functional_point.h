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

#ifndef _DOF_DOF_FUNCTIONAL_POINT_H_
#define _DOF_DOF_FUNCTIONAL_POINT_H_

#include <map>
#include <vector>

#include "common/vector_algebra_descriptor.h"
#include "dof/dof_impl/dof_functional.h"
#include "fem/function_space.h"
#include "mesh/geometric_tools.h"

namespace hiflow {

namespace doffem {

template <class DataType, int DIM> class DofContainerLagrange;

/// Class for Dof functional imlpementing any kind of point evaluation
/// \author Philipp Gerstner

template < class DataType, int DIM > 
class DofPointEvaluation : public virtual DofFunctional <DataType, DIM>
{
public:
  using Coord = typename StaticLA<DIM, DIM, DataType>::ColVectorType;

  /// Constructor
  DofPointEvaluation()
  {
    this->coord_set_ = false;
    this->type_ = DofFunctionalType::POINT_EVAL;
  }

  /// Destructor
  virtual ~DofPointEvaluation()
  {}
  
  inline Coord get_point () const
  {
    return this->dof_coord_;
  }

  inline size_t comp() const 
  {
    return this->comp_;
  }
  
private:
  void init (Coord ref_pt, size_t comp, CRefCellSPtr<DataType, DIM> ref_cell)
  {
    assert (ref_cell != nullptr);
    this->dof_coord_ = ref_pt;
    this->coord_set_ = true;
    this->ref_cell_ = ref_cell;
    this->comp_ = comp;
    mesh::find_subentities_containing_point<DataType, DIM> (ref_pt, ref_cell->topo_cell(), ref_cell->get_coords(), this->attached_to_subentity_);
  }
  
  void evaluate (FunctionSpace<DataType, DIM> const * space,
                 std::vector<DataType>& dof_values ) const
  {
    assert (this->coord_set_);
    assert (space->nb_comp() > 0);
    
    assert (dof_values.size() == space->dim());
    
    if (space->nb_comp() == 1)
    {
      assert(this->comp_ == 0);
      space->N(this->dof_coord_, dof_values);
    }
    else
    {
      std::vector<DataType> weights(space->weight_size(), 0.);
      space->N(this->dof_coord_, weights);
    
      for (size_t j=0; j<space->dim(); ++j)
      {
        assert (space->iv2ind(j, this->comp_) < weights.size());
        dof_values[j] = weights[space->iv2ind(j, this->comp_)];
      }
    }
  }

  void evaluate (RefCellFunction<DataType, DIM> const * func, 
                 size_t offset, 
                 std::vector<DataType> &dof_values) const
  {
    assert (func != nullptr);
    assert (this->coord_set_);
    const size_t nb_comp = func->nb_comp();
    const size_t nb_func = func->nb_func();
    assert (offset + nb_func <= dof_values.size());
    
    std::vector<DataType> values(func->weight_size(), 0.);
    
    //std::cout << "func->evaluate " << this->comp_ << std::endl;
    func->evaluate(this->dof_coord_, values);
    
    assert ( values.size() == func->weight_size() );
    
    for (size_t i = 0; i < nb_func; ++i)
    {
      //std::cout << "point " << this->comp_ << std::endl;
      assert (func->iv2ind(i,this->comp_) < values.size() );
      dof_values[offset+i] = values[func->iv2ind(i,this->comp_)];
    }
  }

  // point at which functions are evaluated
  Coord dof_coord_;

  // component that should be evaluated
  size_t comp_;
  
  bool coord_set_;
   
  friend class DofContainerLagrange<DataType, DIM>;
};

// TODO: check whether test spaces are equal
/*
template <class DataType, int DIM>
bool equal_dofs (const DofPointEvaluation<DataType, DIM> &dof_a,
                 const DofPointEvaluation<DataType, DIM> &dof_b,
                 const CellTransformation<DataType, DIM> &c_trafo_a,
                 const CellTransformation<DataType, DIM> &c_trafo_b) 
{
  if (dof_a.comp() != dof_b.comp())
  {
    return false;
  }

  Coord ref_pt_a = dof_a.get_point();
  Coord ref_pt_b = dof_b.get_point();
    
  Coord pt_a;
  Coord pt_b;
    
  c_trafo_a.transform ( ref_pt_a, pt_a );
  c_trafo_b.transform ( ref_pt_b, pt_b );
    
  if (pt_a == pt_b)
  {
    return true;
  }
  return false;
}
*/
} // namespace doffem
} // namespace hiflow
#endif
