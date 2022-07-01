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

#ifndef HIFLOW_SPACE_FE_INTERPOLATION_GLOBAL
#define HIFLOW_SPACE_FE_INTERPOLATION_GLOBAL

/// \author Philipp Gerstner

#include <vector>

#include "common/parcom.h"
#include "linear_algebra/vector.h"
#include "mesh/entity.h"
#include "space/vector_space.h"
#include "space/fe_interpolation_cell.h"

namespace hiflow {

template < class DataType, int DIM, class CellInterpolator > 
class FeInterBase
{
  typedef la::Vector<DataType> Vector;

public:

protected:
  FeInterBase(const VectorSpace< DataType, DIM > &space)
  : space_(space)
  {
    this->parcom_ = new ParCom(space.get_mpi_comm());
  }

  ~FeInterBase()
  {
    if (this->parcom_ != nullptr)
    {
      delete this->parcom_;
    }
  }
  
  virtual void interpolate (CellInterpolator const * cell_inter, 
                            size_t fe_ind, 
                            Vector& sol) const;
                            
  const VectorSpace< DataType, DIM>& space_;

  ParCom* parcom_;
};

/// \brief Global interpolator for nodal cell interpolation
/// For convenience regarding the ugly template argument "CellInterpolator" 
template < class DataType, int DIM, class Functor> 
class FeInterNodal : public FeInterBase<DataType, DIM, FeInterCellNodal<DataType, DIM, Functor > >
{
public:
  typedef la::Vector<DataType> Vector;
  typedef FeInterCellNodal<DataType, DIM, Functor > NodalCellInterpolator;
  
  FeInterNodal(const VectorSpace< DataType, DIM > &space, 
               Functor const * func,
               size_t fe_ind)
  : FeInterBase<DataType, DIM, NodalCellInterpolator >(space),
  func_(func),
  fe_ind_(fe_ind)
  {
    assert (func != nullptr);
    assert (fe_ind < space.nb_fe());
    this->cell_inter_ = new NodalCellInterpolator(space);
    this->cell_inter_->set_function(this->func_);
  }

  ~FeInterNodal()
  {
    if (this->cell_inter_ != nullptr)
    {
      delete this->cell_inter_; 
    }
  }
  
  void interpolate (Vector& sol) const
  {  
    FeInterBase<DataType, DIM, NodalCellInterpolator >::interpolate(this->cell_inter_, this->fe_ind_, sol);
  }
    
private:
  Functor const * func_;
  size_t fe_ind_;
  
  mutable NodalCellInterpolator const * cell_inter_;
  
};

////////////////////////////////////////////////////
///////////// FeInterGlobal /////////////////////////
////////////////////////////////////////////////////

template < class DataType, int DIM, class CellInterpolator >
void FeInterBase<DataType, DIM, CellInterpolator>::interpolate (CellInterpolator const * cell_inter, 
                                                                size_t fe_ind, 
                                                                Vector& sol) const
{
  assert (cell_inter != nullptr);
    
  assert (fe_ind < this->space_.nb_fe());
  std::vector< std::vector<DataType> > cell_coeff;
  std::vector<DataType> dof_values;
      
  // loop over all cells
  for (mesh::EntityIterator cell = this->space_.meshPtr()->begin(DIM); 
       cell != this->space_.meshPtr()->end(DIM); ++cell) 
  {  
    // evaluate interpolation on current cell
    cell_inter->compute_fe_coeff (&(*cell), fe_ind, cell_coeff);

    assert (cell_coeff.size() == this->space_.nb_dof_on_cell(fe_ind, cell->index()) );
    assert (cell_coeff[0].size() == 1);
    
    // insert values into vector sol
    dof_values.resize(cell_coeff.size(), 0.);
    for (size_t i=0; i<dof_values.size(); ++i)
    {
      dof_values[i] = cell_coeff[i][0];
    }
    this->space_.insert_dof_values(fe_ind, cell->index(), sol, dof_values);
    
    //log_2d_array(cell_coeff, std::cout, 2);
  }
  sol.Update();
}


} // namespace hiflow
#endif
