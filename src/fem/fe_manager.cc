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

#include "common/array_tools.h"
#include "fem/fe_manager.h"
#include "fem/cell_trafo/cell_transformation.h"
#include "fem/fe_reference.h"
#include "mesh/mesh.h"
#include "mesh/iterator.h"
#include "mesh/periodicity_tools.h"

namespace hiflow {
namespace doffem {

template < class DataType, int DIM >
FEManager< DataType, DIM >::FEManager()
    : tdim_(0), nb_fe_(0), nb_var_(0), num_entities_(-1), mesh_(nullptr), fe_tank_initialized_(false) 
{
  initialized_.clear();

  fe_tank_.clear();
  fe_inst_.clear();
  this->cell_transformation_.clear();
  
  is_dg_.clear();
  fe_conformity_.clear();
  var_2_fe_.clear();
  var_2_comp_.clear();

}

template < class DataType, int DIM >
FEManager< DataType, DIM >::~FEManager() 
{
  this->clear_cell_transformation();
  this->fe_inst_.clear();
}

template < class DataType, int DIM >
void FEManager< DataType, DIM >::set_mesh(const mesh::Mesh &mesh) 
{
  mesh_ = &mesh;
  this->tdim_ = mesh.tdim();
  num_entities_ = mesh_->num_entities(tdim_);
  fe_tank_.resize(num_entities_);
}

template < class DataType, int DIM >
FEType FEManager< DataType, DIM >::fe_type_for_var(int cell_index, size_t var) const 
{
  assert (var < this->nb_var_);
  return this->get_fe_for_var(cell_index, var)->type();
}

template < class DataType, int DIM >
FEType FEManager< DataType, DIM >::fe_type(int cell_index, size_t fe_ind) const 
{
  assert (fe_ind < this->nb_fe_);
  return this->get_fe(cell_index, fe_ind)->type();
}

template < class DataType, int DIM >
int FEManager< DataType, DIM >::get_fe_index(CRefElementSPtr< DataType, DIM >& ref_fe, int cell_index)  const
{
  assert (cell_index >= 0);
  assert (cell_index < this->fe_tank_.size());
  int ind = -1;
  for (size_t fe_ind = 0; fe_ind < this->fe_tank_[cell_index].size(); ++fe_ind)
  {
    if ((*this->fe_tank_[cell_index][fe_ind]) == (*ref_fe))
    {
      ind = fe_ind;
      break;
    }
  }
  return ind;
}

template < class DataType, int DIM >
size_t FEManager< DataType, DIM >::nb_dof_on_cell (int cell_index, size_t fe_ind) const
{
  assert (this->fe_tank_initialized_);
  assert (cell_index >= 0);
  assert (cell_index < this->fe_tank_.size());
  assert (fe_ind < this->fe_tank_[cell_index].size());
  
  return this->fe_tank_[cell_index][fe_ind]->nb_dof_on_cell();
}

template < class DataType, int DIM >
size_t FEManager< DataType, DIM >::nb_dof_on_cell (int cell_index) const
{
  size_t nb_dofs = 0;
  assert (this->fe_tank_initialized_);
  assert (cell_index >= 0);
  assert (cell_index < this->fe_tank_.size());
  assert (this->fe_tank_[cell_index].size() == this->nb_fe_);
  
  for (size_t l = 0; l < this->fe_tank_[cell_index].size(); ++l) 
  {
    nb_dofs += this->fe_tank_[cell_index][l]->nb_dof_on_cell();
  }
  
  return nb_dofs;
}

template < class DataType, int DIM >
void FEManager< DataType, DIM >::init( const std::vector<FEType> &fe_types, 
                                       const std::vector<bool> & is_dg,
                                       const std::vector< std::vector< std::vector< int > > > &param) 
{
  assert(param.size() == fe_types.size());
  assert(is_dg.size() == fe_types.size());

  this->tdim_ = this->mesh_->tdim();
   
  this->nb_fe_ = fe_types.size();

  this->var_2_fe_.clear();
  this->var_2_comp_.clear();
  
  this->initialized_.clear();
  this->initialized_.resize(nb_fe_, false);
  
  this->fe_tank_.clear();
  this->fe_tank_.resize(this->mesh_->num_entities(tdim_));
  this->fe_inst_.clear();
  
  this->clear_cell_transformation();
  cell_transformation_.resize(num_entities_);
  
  std::vector< mesh::MasterSlave > period = mesh_->get_period();
  RefCellType ref_cell_type = RefCellType::NOT_SET;

  this->fe_2_var_.clear();
  this->fe_2_var_.resize(nb_fe_);
  this->is_dg_ = is_dg;
  
  // loop over all components in Finite element vector space
  this->fe_conformity_.clear();
  this->fe_conformity_.resize(nb_fe_, FEConformity::NONE);
  
  this->same_fe_on_all_cells_ = true;

  for (size_t fe_ind = 0; fe_ind < nb_fe_; ++fe_ind) 
  {
    assert (param[fe_ind].size() == 1 || param[fe_ind].size() == mesh_->num_entities(tdim_));
     
    // loop over all mesh cells
    int c = 0;
    int prev_fe_id = -1;
    for (mesh::EntityIterator it = mesh_->begin(tdim_), e_it = mesh_->end(tdim_); it != e_it; ++it) 
    {
      std::vector<int> fe_params;
      if (param[fe_ind].size() == 1)
      {
        fe_params = param[fe_ind][0];
      }
      else
      {
        fe_params = param[fe_ind][c];
      }
      mesh::CellType::Tag topo_cell_type = it->cell_type().tag();
      mesh::AlignNumber align_number = it->get_align_number();
            
      // initialize FE on current mesh cell and store correpsonding pointer in fe_tank
      int old_size = this->fe_inst_.nb_fe_inst();
      
      const size_t fe_id = this->fe_inst_.add_fe (fe_types[fe_ind], topo_cell_type, fe_params);
      
      if (prev_fe_id < 0)
      {
        prev_fe_id = fe_id;
      }
      else
      {
        if (fe_id != prev_fe_id)
        {
          this->same_fe_on_all_cells_ = false;
        }
        prev_fe_id = fe_id;
      }


      int new_size = this->fe_inst_.nb_fe_inst();
      
      if (!is_dg[fe_ind])
      {
        // we do not allow mixed conformity within a single FE
        assert (this->fe_conformity_[fe_ind] == FEConformity::NONE 
                || this->fe_conformity_[fe_ind] == this->fe_inst_.max_fe_conformity(fe_id));
               
        this->fe_conformity_[fe_ind] = this->fe_inst_.max_fe_conformity(fe_id);
      }
      else
      {
        this->fe_conformity_[fe_ind] = FEConformity::L2;
      }

      if (old_size != new_size)
      {
          LOG_INFO("added FE", "index = " << fe_ind << ", type = " << as_integer(fe_types[fe_ind]) 
                    << ", max possible conformity = " << as_integer(this->fe_inst_.max_fe_conformity(fe_id))); 
      }
      
      this->fe_tank_[it->index()].push_back(this->fe_inst_.get_fe(fe_id));
      assert (this->fe_tank_[it->index()].size() == fe_ind+1);

//    this->cell_transformation_[it->index()].push_back(cell_trafo);
      if (fe_ind == 0)
      {
        // create and initialize cell transformation for current cell
        std::vector<DataType> coord_vtx;
        it->get_coordinates(coord_vtx);

        // TODO: I (Philipp G) think there is a bug...
        if (!period.empty()) 
        {
          std::vector<DataType> tmp_coord = coord_vtx;
          coord_vtx = unperiodify(tmp_coord, DIM, period);
        }
        
        CellTrafoSPtr<DataType, DIM> cell_trafo 
            = this->fe_inst_.create_cell_trafo(fe_id, coord_vtx, *it,  period);
        //  = this->fe_inst_.create_cell_trafo(fe_id, align_number);

        this->cell_transformation_[it->index()] = cell_trafo;
      }
      c++;
    }

    auto example_fe = this->fe_tank_[0][fe_ind];
    
    // ensure that all reference elements live on the same reference cell
    if (ref_cell_type == RefCellType::NOT_SET)
    {
      assert (example_fe->ref_cell_type() != RefCellType::NOT_SET);
      ref_cell_type = example_fe->ref_cell_type();
    }
    else
    {
      assert ( ref_cell_type == example_fe->ref_cell_type());
    }
    
    // setup dof variable <-> physical variable mappings 
    size_t nb_comp = example_fe->nb_comp();
    for (size_t v=0; v<nb_comp; ++v) 
    {
      this->var_2_fe_.push_back(fe_ind);
      this->var_2_comp_.push_back(v);
      this->fe_2_var_[fe_ind].push_back(nb_var_+v);
    }
    
    this->nb_var_ += nb_comp;
  }
  this->fe_tank_initialized_ = true;
  LOG_INFO("# different FEs", this->fe_inst_.nb_fe_inst());
  //LOG_INFO("FE conformity", string_from_range(this->fe_conformity_.begin(), this->fe_conformity_.end()));
}


template < class DataType, int DIM >
void FEManager< DataType, DIM >::get_status() const 
{
  std::cout << "DIM:   " << tdim_ << std::endl;
  std::cout << "nb_fe: " << nb_fe_ << std::endl;
  std::cout << "nb_var: " << nb_var_ << std::endl;

  std::cout << "tank size: " << fe_tank_.size() << std::endl;
  for (size_t i = 0; i < fe_tank_.size(); ++i) 
  {
    for (size_t l=0; l< fe_tank_[i].size(); ++l)
    {
      std::cout << "\t" << i << ", " << l << "\t" << fe_tank_[i][l]->name() << std::endl;
    }
  }
}

template < class DataType, int DIM >
void FEManager< DataType, DIM >::clear_cell_transformation()
{
  // no need to delete, since we are using a shared pointer
  /*
  for (size_t c=0; c < this->cell_transformation_.size(); ++c)
  {
      delete this->cell_transformation_[c];
  }
  * */
  this->cell_transformation_.clear();
}

template class FEManager< float, 3 >;
template class FEManager< float, 2 >;
template class FEManager< float, 1 >;

template class FEManager< double, 3 >;
template class FEManager< double, 2 >;
template class FEManager< double, 1 >;

} // namespace doffem
} // namespace hiflow
