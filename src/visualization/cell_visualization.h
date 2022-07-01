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

#ifndef HIFLOW_VISU_CELL_VISUALIZATION
#define HIFLOW_VISU_CELL_VISUALIZATION

/// \author Staffan Ronnas, Martin Baumann, Teresa Beck, Simon Gawlok, Jonas
/// Kratzke, Philipp Gerstner
///
/// \brief Visualization of finite element functions.
///
/// Using this class a Vtk (http://www.vtk.org/) unstructured grid visualization
/// file can be created. Please find detailed information about Vtk's file
/// formats at http://www.vtk.org/VTK/img/file-formats.pdf.
/// This type of visualization writes out every cell and with function values
/// provided by a user-defined evaluation function.
///
/// Please note for simulations with multiple visualization calls, that this
/// class is NOT ment to be initialized once for several visualization calls.
/// Please construct a new instantiation of the CellVisualization every single
/// time you want to visualize your data.
///


#include <map>
#include <string>
#include <vector>
#include <sstream>
#include <boost/function.hpp>

#include "visualization/visualization_data.h"
#include "common/log.h"
#include "common/bbox.h"
#include "common/grid.h"
#include "mesh/iterator.h"
#include "fem/cell_trafo/cell_transformation.h"

#define nSKIP_GHOST

namespace hiflow {

namespace la {
template < class DataType > class Vector;
}

using namespace mesh;

template < class DataType, int DIM > class VectorSpace;

/// \brief Description of a square 2d grid or cubic 3d grid.
template < class DataType, int DIM > 
class DummyTransform {
public:
  using Coord = typename StaticLA<DIM, DIM, DataType>::ColVectorType;
  DummyTransform()
  {}
  
  ~DummyTransform()
  {}
  
  void transform (Coord& pt) const 
  {}
  
};

template < class DataType, int DIM > 
class CellVisualizationGrids {
public:
  using Coord = typename StaticLA<DIM, DIM, DataType>::ColVectorType;

  CellVisualizationGrids(const Mesh *mesh, int num_intervals, DataType origin, DataType side_length);

  int num_visu_points() const;
  int num_visu_cells() const;
  int num_points(CellType::Tag cell_tag) const;
  int num_cells(CellType::Tag cell_tag) const;

  const std::vector< int > &vertices_of_cell(CellType::Tag cell_tag, int i) const;
  const std::vector< Coord > &coords(CellType::Tag cell_tag) const;

private:
  typename ScopedPtr< Grid< DataType, DIM > >::Type grids_[CellType::NUM_CELL_TYPES];
  const int tdim_;
  int num_visu_points_;
  int num_visu_cells_;
  
};

/// \brief Visualization of finite element solutions.

template < class DataType, int DIM > 
class CellVisualization : public virtual VisualizationData<DataType, DIM>
{
public:
  using Coord = typename StaticLA<DIM, DIM, DataType>::ColVectorType;
  using Grad = typename StaticLA<DIM, DIM, DataType>::ColVectorType;
  typedef boost::function1<void, Coord&> CoordTrafoFunction;


  explicit CellVisualization(const VectorSpace< DataType, DIM > &space);
  
  explicit CellVisualization(const VectorSpace< DataType, DIM > &space, const int num_intervals);

  ~CellVisualization () {
    this->clear();
  }
  
  /// setup data structures defining the mesh in vtk format
  void visualize_mesh ()
  {
    DummyTransform<DataType, DIM> trafo;
    this->visualize_mesh<DummyTransform<DataType, DIM> >(&trafo);
  }
  
  template <class CoordTrafo>
  void visualize_mesh (CoordTrafo * trafo);
  
  void visualize_cell_data(const std::vector< DataType > &cell_data, 
                           const std::string name);

  template <class OtherDataType >
  void visualize_cell_data(const std::vector< OtherDataType > &cell_data, 
                           const std::string name);
                               
  void visualize(const la::Vector<DataType>& coeff,
                 const size_t var, 
                 const std::string name);

  void visualize(const la::Vector<DataType>& coeff,
                 const std::vector<size_t>& vars, 
                 const std::vector<std::string> &names);
                 
  template <class CellWiseEvaluator>
  void visualize(const CellWiseEvaluator &fun, 
                 const std::vector<std::string> &names );

  template <class CellWiseEvaluator>
  void visualize(const CellWiseEvaluator &fun, 
                 const std::string name );
                 
  template <class CellWiseEvaluator>
  void visualize_grad (const CellWiseEvaluator &fun, 
                       const std::vector<std::string> &name);

  void visualize_grad(const la::Vector<DataType>& coeff,
                      const size_t var, 
                      const std::string name);
  
  void visualize_grad(const la::Vector<DataType>& coeff,
                      const std::vector<size_t>& vars, 
                      const std::vector<std::string> &names);

protected:
  bool parallel_visualization_;
  
  template <class CellWiseEvaluator>
  void visualize_grid(const CellWiseEvaluator &fun, 
                      const std::vector<std::string> &names );
                 
  template <class CellWiseEvaluator>
  void visualize_facet(const CellWiseEvaluator &fun, 
                       const std::vector<std::string> &names );
                 
  template <class CellWiseEvaluator>
  void visualize_grad_grid (const CellWiseEvaluator &fun, 
                            const std::vector<std::string> &names);
                     
  template <class CellWiseEvaluator>
  void visualize_grad_facet (const CellWiseEvaluator &fun, 
                             const std::vector<std::string> &names);
        
  const VectorSpace< DataType, DIM > &space_;
  CellVisualizationGrids< DataType, DIM > grids_;
  int num_intervals_;
  
};

///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////

template <class DataType, int DIM>
template <class OtherDataType >
void CellVisualization<DataType, DIM>::visualize_cell_data(const std::vector< OtherDataType > &cell_data, 
                                                           const std::string name)
{
  const size_t num_cells = cell_data.size();
  std::vector<DataType> tmp_data (num_cells,0.);
  
  for (size_t c=0; c<num_cells; ++c)
  {
    tmp_data[c] = static_cast<DataType> (cell_data[c]);
  }
  this->visualize_cell_data(tmp_data, name);
}

template <class DataType, int DIM>
template <class CellWiseEvaluator>
void CellVisualization<DataType, DIM>::visualize(const CellWiseEvaluator &fun, 
                                                 const std::string name )
{
  assert (fun.nb_comp() == 1);
  std::vector< std::string > names(1, name);
  this->visualize(fun, names);
}

template <class DataType, int DIM>
template <class CellWiseEvaluator>
void CellVisualization<DataType, DIM>::visualize(const CellWiseEvaluator &fun, 
                                                 const std::vector<std::string> &names )
{
  if (this->num_intervals_ != 1)
  { 
    visualize_grid(fun, names);
  }
  else
  {
    visualize_grid(fun, names);
    //visualize_facet(fun, names);
  }
}


template <class DataType, int DIM>
template <class CellWiseEvaluator>
void CellVisualization<DataType, DIM>::visualize_grid(const CellWiseEvaluator &fun, 
                                                      const std::vector<std::string> &names )
{
  const mesh::Mesh &mesh = space_.mesh();
  const mesh::TDim tdim = mesh.tdim();

  if (this->mapped_pts_.size() == 0) 
  {
    this->visualize_mesh();
  }

  const size_t nb_var = fun.nb_comp();
  assert (nb_var == names.size());

  std::vector< std::vector< DataType > > values(nb_var); 

  for (size_t d=0; d<nb_var; ++d)
  {
    values[d].reserve( grids_.num_visu_points());
  }
  
  std::vector<  DataType > cell_values(nb_var); 
  
  for (mesh::EntityIterator it = mesh.begin(tdim), end_it = mesh.end(tdim);
       it != end_it; ++it) 
  {
    if (parallel_visualization_) 
    {
      int rem_ind = -100;
      it->get<int>("_remote_index_", &rem_ind);
      if (rem_ind != -1) 
      {
#ifdef SKIP_GHOST
        continue;
#endif
      }
    }

    std::vector< Coord > ref_pts = this->grids_.coords(it->cell_type().tag());
    const size_t num_pt = ref_pts.size();
    
    /// loop over points in cell
    for (size_t i=0; i<num_pt; ++i)
    {
      fun.r_evaluate(*it, ref_pts[i], cell_values); 
      assert (cell_values.size() == nb_var);

      for (size_t d=0; d<nb_var; ++d)
      {
        values[d].push_back(cell_values[d]);
      }
    }
  }
  for (size_t d=0; d<nb_var; ++d)
  {
    this->functions_.insert(std::make_pair(names[d], values[d]));  
  }
}

template <class DataType, int DIM>
template <class CellWiseEvaluator>
void CellVisualization<DataType, DIM>::visualize_facet(const CellWiseEvaluator &fun, 
                                                       const std::vector<std::string> &names )
{
  const mesh::Mesh &mesh = space_.mesh();
  const mesh::TDim tdim = mesh.tdim();
  const mesh::TDim tdim_f = mesh.tdim()-1;

  if (this->mapped_pts_.size() == 0) 
  {
    this->visualize_mesh();
  }
  const size_t nb_var = fun.nb_comp();
  
  std::vector< std::vector <DataType >> values(nb_var);
  std::vector<DataType> cell_values(nb_var);
 
  for (size_t d=0; d<nb_var; ++d)
  {
    values[d].reserve(grids_.num_visu_points());
  }

  for (mesh::EntityIterator it = mesh.begin(tdim_f), end_it = mesh.end(tdim_f);
       it != end_it; ++it) 
  {

   // if (it->get_material_number() != this->mat_num_) {
   //   continue;
    //}

    if (parallel_visualization_) 
    {
      std::vector<int> rem_ind(0);

      for (mesh::IncidentEntityIterator it_cell = it->begin_incident(tdim),
           end_it_cell = it->end_incident(tdim);
           it_cell != end_it_cell; ++it_cell) 
      {
        int ind;
        it_cell->get<int>("_remote_index_", &ind);
        rem_ind.push_back(ind);
      }

      if (std::count(rem_ind.begin(), rem_ind.end(), -1) == 0) 
      {
#ifdef SKIP_GHOST
        continue;
#endif
      }
    }
    cell_values.clear();
    //cell_values.resize(it->num_vertices(), 1.e32);

    std::vector<DataType> coords;
    it->get_coordinates(coords);

    mesh::IncidentEntityIterator it_cell = it->begin_incident(tdim);
    
    const doffem::CellTransformation<DataType, DIM> &cell_trans =
      space_.get_cell_transformation(it_cell->index());

    std::vector<DataType> ref_coords;
    ref_coords.resize(coords.size());

    for (int i = 0; i != it->num_vertices(); ++i) 
    {
      Coord phys_coords;
      Coord ref_coords;
      
      for (int d = 0; d < DIM; d++) 
      {
        phys_coords[d] = coords[i * DIM + d];
      }
      
      cell_trans.inverse(phys_coords, ref_coords);
      
      //evaluate values on each cell, and insert to values
      fun.r_evaluate(*it_cell, ref_coords, cell_values);
      
      for (int d = 0; d< nb_var; ++d) 
      {
        values[d].push_back(cell_values[d]); 
      }
    }
  }
  for (int d = 0; d < nb_var; ++d)
  {
    this->functions_.insert(std::make_pair(names[d], values[d]));
  }
}


template <class DataType, int DIM>
template <class CellWiseEvaluator>
void CellVisualization<DataType, DIM>::visualize_grad(const CellWiseEvaluator &fun, 
                                                      const std::vector<std::string> &names)
{
  if (this->num_intervals_ != 1)
  { 
    visualize_grad_grid(fun, names);
  }
  else 
  {
    visualize_grad_grid(fun, names);
    //visualize_grad_facet(fun, names);
  }
}

template <class DataType, int DIM>
template <class CellWiseEvaluator>
void CellVisualization<DataType, DIM>::visualize_grad_grid(const CellWiseEvaluator &fun, 
                                                           const std::vector<std::string> &names)
{
  const mesh::Mesh &mesh = space_.mesh();
  const mesh::TDim tdim = mesh.tdim();

  if (this->mapped_pts_.size() == 0) 
  {
    this->visualize_mesh();
  }

  const size_t nb_var = fun.nb_comp();
  assert (nb_var == names.size());
  
  std::vector< std::vector< DataType > > values_grad(nb_var); 
  
  for (size_t d=0; d<nb_var; ++d)
  {
    values_grad[d].reserve( 3*grids_.num_visu_points());
  }
  
  std::vector< Grad > cell_values_grad(nb_var); 
  
  for (mesh::EntityIterator it = mesh.begin(tdim), end_it = mesh.end(tdim);
       it != end_it; ++it) 
  {
    if (parallel_visualization_) 
    {
      int rem_ind = -100;
      it->get<int>("_remote_index_", &rem_ind);
      if (rem_ind != -1) 
      {
#ifdef SKIP_GHOST
        continue;
#endif
      }
    }

    std::vector< Coord > ref_pts = this->grids_.coords(it->cell_type().tag());
    const size_t num_pt = ref_pts.size();
    
    /// loop over points in cell
    for (size_t i=0; i<num_pt; ++i)
    {
      fun.r_evaluate_grad(*it, ref_pts[i], cell_values_grad); //
      assert (cell_values_grad.size() == nb_var);
    
      for (size_t d=0; d<nb_var; ++d)
      {
        for (size_t j = 0; j < DIM; ++j)
        {
          values_grad[d].push_back(cell_values_grad[d][j]);
        }
        for (size_t j = DIM; j< 3; ++j)
        {
          values_grad[d].push_back(0.);
        }
      }    
    }
  }

  for (size_t d=0; d<nb_var; ++d)
  {
    this->functions_grad_.insert(std::make_pair(names[d], values_grad[d]));
  }
}

template <class DataType, int DIM>
template <class CellWiseEvaluator>
void CellVisualization<DataType, DIM>::visualize_grad_facet(const CellWiseEvaluator &fun, 
                                                            const std::vector<std::string> &names)
{
  const mesh::Mesh &mesh = space_.mesh();
  const mesh::TDim tdim = mesh.tdim();
  const mesh::TDim tdim_f = mesh.tdim()-1;

  if (this->mapped_pts_.size() == 0) 
  {
    this->visualize_mesh();
  }
  const size_t nb_var = fun.nb_comp();
  
  std::vector<  std::vector< DataType  > > values_grad(nb_var); 
  
  std::vector< Grad > cell_values_grad(nb_var); 
  
  for (size_t d=0; d<nb_var; ++d)
  {
    values_grad[d].reserve(3 * grids_.num_visu_points());
  }

  for (mesh::EntityIterator it = mesh.begin(tdim_f), end_it = mesh.end(tdim_f);
       it != end_it; ++it) 
  {
   // if (it->get_material_number() != this->mat_num_) {
   //   continue;
    //}

    if (parallel_visualization_) 
    {
      std::vector<int> rem_ind(0);

      for (mesh::IncidentEntityIterator it_cell = it->begin_incident(tdim),
           end_it_cell = it->end_incident(tdim);
           it_cell != end_it_cell; ++it_cell) 
      {
        int ind;
        it_cell->get<int>("_remote_index_", &ind);
        rem_ind.push_back(ind);
      }

      if (std::count(rem_ind.begin(), rem_ind.end(), -1) == 0) 
      {
#ifdef SKIP_GHOST
        continue;
#endif
      }
    }

    cell_values_grad.clear();
    //cell_values.resize(it->num_vertices(), 1.e32);

    std::vector<DataType> coords;
    it->get_coordinates(coords);

    mesh::IncidentEntityIterator it_cell = it->begin_incident(tdim);
    
    const doffem::CellTransformation<DataType, DIM> &cell_trans =
      space_.get_cell_transformation(it_cell->index());

    std::vector<DataType> ref_coords;
    ref_coords.resize(coords.size());

    for (int i = 0; i != it->num_vertices(); ++i) 
    {
      Coord phys_coords;
      Coord ref_coords;
      
      for (int d = 0; d < DIM; d++) 
      {
        phys_coords[d] = coords[i * DIM + d];
      }
      
      cell_trans.inverse(phys_coords, ref_coords);
      
      //evaluate values on each cell, and insert to values
      fun.r_evaluate_grad(*it_cell, ref_coords, cell_values_grad);
      
      for (int d = 0; d< nb_var; ++d) 
      {
        for (size_t j = 0; j < DIM; ++j)
        {
          values_grad[d].push_back(cell_values_grad[d][j]);
        }
        for (size_t j = DIM; j< 3; ++j)
        {
          values_grad[d].push_back(0.);
        }
      }
    }
  }
  for (int d = 0; d < nb_var; ++d)
  {
    this->functions_grad_.insert(std::make_pair(names[d], values_grad[d]));
  }
}

template <class DataType, int DIM>
template <class CoordTrafo>
void CellVisualization<DataType, DIM>::visualize_mesh(CoordTrafo * trafo) 
{
  const mesh::Mesh &mesh = this->space_.mesh();
  const mesh::TDim tdim = mesh.tdim();
  const mesh::GDim gdim = mesh.gdim();

  size_t num_visu_points = this->grids_.num_visu_points();
  size_t num_visu_cells = this->grids_.num_visu_cells();

  ////////// collect point coordinates ////////////////////////////////////
  // determine number of points and allocate vectors for points coordinates
  this->mapped_pts_.clear();
  this->mapped_pts_.reserve(num_visu_points * 3);

  for (mesh::EntityIterator it = mesh.begin(tdim), end_it = mesh.end(tdim);
       it != end_it; ++it) 
  {
    if (this->parallel_visualization_) 
    {
      int rem_ind = -100;
      it->get<int>("_remote_index_", &rem_ind);
      if (rem_ind != -1)
      {
#ifdef SKIP_GHOST
        continue;
#endif
      }
    }
    doffem::CCellTrafoSPtr<DataType, DIM> cell_trans = this->space_.get_cell_transformation(it->index());

    for (size_t p = 0, p_end = this->grids_.num_points(it->cell_type().tag());
         p != p_end; ++p) 
    {
      Coord ref_pt = this->grids_.coords(it->cell_type().tag())[p];
      Coord mapped_pt;
      cell_trans->transform(ref_pt, mapped_pt);

      if (trafo != nullptr) {
        trafo->transform(mapped_pt);
      }

      for (int d=0; d<DIM; ++d)
      {
        this->mapped_pts_.push_back(mapped_pt[d]);
      }
      for (int d=DIM; d<3; ++d)
      {
        this->mapped_pts_.push_back(0.);
      }
    }
  }

  ////////// end collect point coordinates ////////////////////////////////////

  //// collect cell data //////////////////////////////////////////////////////
  size_t p_offset = 0, cell_offset = 0;
  this->verts_.clear();
  this->verts_.reserve(num_visu_cells * 8);

  this->cell_offsets_.clear();
  this->cell_offsets_.reserve(num_visu_cells);

  this->cell_types_.clear();
  this->cell_types_.reserve(num_visu_cells);

  // Connectivity, Offsets, and Types arrays
  static const int vtk_cell_types[] = {1, 3, 5, 9, 10, 12, 14};

  for (mesh::EntityIterator it = mesh.begin(tdim); it != mesh.end(tdim); ++it) 
  {
    if (this->parallel_visualization_) 
    {
      int rem_ind = -100;
      it->get<int>("_remote_index_", &rem_ind);
      if (rem_ind != -1)
      {
#ifdef SKIP_GHOST
        continue;
#endif
      }
    }

    for (size_t c = 0, c_end = this->grids_.num_cells(it->cell_type().tag());
         c != c_end; ++c) 
    {
      const std::vector<int> &verts =
          this->grids_.vertices_of_cell(it->cell_type().tag(), c);
      for (size_t v = 0, v_end = verts.size(); v != v_end; ++v) 
      {
        this->verts_.push_back(verts[v] + p_offset);
      }

      cell_offset += verts.size();
      this->cell_offsets_.push_back(cell_offset);

      // pyr do not refine only into pyrs
      if (static_cast<int>(it->cell_type().tag()) == 6) 
      {
        if (c_end == 1) 
        {
          this->cell_types_.push_back(
              vtk_cell_types[static_cast<int>(it->cell_type().tag())]);
        } 
        else 
        {
          if (c < 6) 
          {
            this->cell_types_.push_back(vtk_cell_types[static_cast<int>(it->cell_type().tag())]);
          } 
          else 
          {
            this->cell_types_.push_back(vtk_cell_types[static_cast<int>(4)]);
          }
        }
      } 
      else 
      {
        this->cell_types_.push_back(vtk_cell_types[static_cast<int>(it->cell_type().tag())]);
      }
    }
    p_offset += this->grids_.num_points(it->cell_type().tag());
  }
}
} // namespace hiflow

#endif
