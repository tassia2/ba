#include "visualization/cell_visualization.h"

#include <cmath>
#include <limits>
#include <string>
#include <sstream>
#include <mpi.h>

#include "fem/cell_trafo/cell_transformation.h"
#include "linear_algebra/vector.h"
#include "mesh/types.h"
#include "mesh/mesh.h"
#include "mesh/attributes.h"
#include "mesh/entity.h"
#include "mesh/iterator.h"
#include "space/vector_space.h"
#include "space/fe_evaluation.h"
#include "space/element.h"

namespace hiflow {

using namespace mesh;

template <class DataType, int DIM>
CellVisualizationGrids<DataType, DIM>::CellVisualizationGrids( const Mesh *mesh,  
                                                               const int num_intervals, 
                                                               DataType origin, 
                                                               DataType side_length)
: tdim_(mesh->tdim()) 
{
  num_visu_points_ = 0;
  num_visu_cells_ = 0;
  for (mesh::EntityIterator it = mesh->begin(tdim_), end_it = mesh->end(tdim_);
       it != end_it; ++it) 
  {
    if (mesh->has_attribute("_remote_index_", tdim_)) 
    {
      int rem_ind;
      it->get<int>("_remote_index_", &rem_ind);
      if (rem_ind != -1) 
      {
#ifdef SKIP_GHOST
        continue;
#endif
      }
    }

    const CellType::Tag cell_tag = it->cell_type().tag();
    if (!grids_[cell_tag]) 
    {
      std::vector<DataType> extents(2 * tdim_);
      for (size_t i = 0; i < static_cast<size_t>(tdim_); ++i) 
      {
        extents[2 * i] = origin;
        extents[2 * i + 1] = origin + side_length;
      }
      BBox<DataType, DIM> bbox(extents);
      std::vector<int> num_intervals_vec(tdim_, num_intervals);
      grids_[cell_tag].reset(
          new Grid<DataType, DIM>(cell_tag, num_intervals_vec, bbox));
    }
    num_visu_points_ += grids_[cell_tag]->get_num_points();
    num_visu_cells_ += grids_[cell_tag]->get_num_cells();
  }
}

template <class DataType, int DIM>
int CellVisualizationGrids<DataType, DIM>::num_visu_points() const {
  return num_visu_points_;
}

template <class DataType, int DIM>
int CellVisualizationGrids<DataType, DIM>::num_visu_cells() const {
  return num_visu_cells_;
}

template <class DataType, int DIM>
int CellVisualizationGrids<DataType, DIM>::num_points(CellType::Tag cell_tag) const {
  return grids_[cell_tag]->get_num_points();
}

template <class DataType, int DIM>
int CellVisualizationGrids<DataType, DIM>::num_cells(CellType::Tag cell_tag) const {
  return grids_[cell_tag]->get_num_cells();
}

template <class DataType, int DIM>
const std::vector<int> &
CellVisualizationGrids<DataType, DIM>::vertices_of_cell(CellType::Tag cell_tag,
                                                   int i) const {
  return grids_[cell_tag]->vertices_of_cell(i);
}

template <class DataType, int DIM>
const std::vector< typename CellVisualizationGrids<DataType, DIM>::Coord >&
CellVisualizationGrids<DataType, DIM>::coords(CellType::Tag cell_tag) const {
  return grids_[cell_tag]->coords();
}

template class CellVisualizationGrids<float, 1>;
template class CellVisualizationGrids<float, 2>;
template class CellVisualizationGrids<float, 3>;

template class CellVisualizationGrids<double, 1>;
template class CellVisualizationGrids<double, 2>;
template class CellVisualizationGrids<double, 3>;

///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////

template <class DataType, int DIM>
CellVisualization<DataType, DIM>::CellVisualization( const VectorSpace<DataType, DIM> &space, 
                                                     const int num_intervals)
    : VisualizationData<DataType, DIM> (space.get_mpi_comm()),
    space_(space), 
    num_intervals_(num_intervals),
    grids_(&space.mesh(), num_intervals, 0., 1.) 
{
  parallel_visualization_ = space_.mesh().has_attribute("_remote_index_", space_.mesh().tdim());
  this->tdim_ = space_.mesh().tdim();
  this->gdim_ = space_.mesh().gdim();
}

template <class DataType, int DIM>
CellVisualization<DataType, DIM>::CellVisualization( const VectorSpace<DataType, DIM> &space)
    : VisualizationData<DataType, DIM> (space.get_mpi_comm()),
    space_(space),
    num_intervals_(0),
    grids_(&space.mesh(), 1, 0., 1.)
{
  parallel_visualization_ = space_.mesh().has_attribute("_remote_index_", space_.mesh().tdim());
  this->tdim_ = space_.mesh().tdim();
  this->gdim_ = space_.mesh().gdim();
}

template <class DataType, int DIM>
void CellVisualization<DataType, DIM>::visualize(const la::Vector<DataType>& coeff,
                                                 const std::vector<size_t>& vars, 
                                                 const std::vector<std::string> &names) 
{
  assert (vars.size() > 0);
  assert (vars.size() == names.size());
#ifndef NDEBUG
  for (size_t l=0; l<vars.size(); ++l)
  {
    assert (vars[l] < this->space_.nb_var()); 
  }
#endif
  this->visualize(FeEvalCell<DataType, DIM>(this->space_, coeff, vars), names); 
}

template <class DataType, int DIM>
void CellVisualization<DataType, DIM>::visualize(const la::Vector<DataType>& coeff,
                                                 const size_t var, 
                                                 const std::string name) 
{
  assert (var < this->space_.nb_var());
  std::vector<size_t> vars(1, var);
  std::vector< std::string > names(1,name);
  
  this->visualize(coeff, vars, names); 
}

template <class DataType, int DIM>
void CellVisualization<DataType, DIM>::visualize_grad(const la::Vector<DataType>& coeff,
                                                      const std::vector<size_t>& vars, 
                                                      const std::vector<std::string> &names) 
{
  assert (vars.size() > 0);
  assert (vars.size() == names.size());
#ifndef NDEBUG
  for (size_t l=0; l<vars.size(); ++l)
  {
    assert (vars[l] < this->space_.nb_var()); 
  }
#endif

  this->visualize_grad(FeEvalCell<DataType, DIM>(this->space_, coeff, vars), names); 
}

template <class DataType, int DIM>
void CellVisualization<DataType, DIM>::visualize_grad(const la::Vector<DataType>& coeff,
                                                      const size_t var, 
                                                      const std::string name) 
{
  assert (var < this->space_.nb_var());
  std::vector<size_t> vars(1, var);
  std::vector< std::string > names(1,name);
  
  this->visualize_grad(coeff, vars, names); 
}

template <class DataType, int DIM>
void CellVisualization<DataType, DIM>::visualize_cell_data( const std::vector<DataType> &cell_data, 
                                                            const std::string name) 
{
  const mesh::Mesh &mesh = space_.mesh();
  const mesh::TDim tdim = mesh.tdim();

  if (this->mapped_pts_.size() == 0) 
  {
    this->visualize_mesh();
  }

  std::vector<DataType> values;

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

    values.insert(values.end(), grids_.num_cells(it->cell_type().tag()),
                  cell_data[it->index()]);
  }

  this->functions_cell_.insert(std::make_pair(name, values));
}

template class CellVisualization<float, 1>;
template class CellVisualization<float, 2>;
template class CellVisualization<float, 3>;

template class CellVisualization<double, 1>;
template class CellVisualization<double, 2>;
template class CellVisualization<double, 3>;


} // namespace hiflow
