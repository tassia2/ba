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

#ifndef HIFLOW_VISU_VISUALIZATION_DATA
#define HIFLOW_VISU_VISUALIZATION_DATA

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
#include "common/vector_algebra.h"
#include "mesh/entity.h"


namespace hiflow {

template < class DataType, int DIM > class VectorSpace;

template < class DataType, int DIM > 
class VisualizationData 
{
public:
  typedef boost::function1<void, Vec<DIM, DataType>&> CoordTrafoFunction;
   
  ~VisualizationData () {
    this->clear();
  }
  
  void communicate_data(int num_writers) const;
  
  void clear() 
  {
    this->clear_mesh_data();
    this->clear_function_data();
    this->clear_cell_data();
  }
  
  void clear_mesh_data() 
  {
    this->mapped_pts_.clear();
    this->verts_.clear();
    this->cell_offsets_.clear();
    this->cell_types_.clear();
  }
  
  void clear_function_data() 
  {
    this->functions_.clear();
  }
  
  void clear_cell_data() 
  {
    this->functions_cell_.clear();
  }
  
  std::map< std::string, std::vector< DataType > >& get_point_values()
  {
    return this->functions_;
  }
  
  std::map< std::string, std::vector< DataType > >& get_cell_values()
  {
    return this->functions_cell_;
  }

  std::map<std::string, std::vector< DataType > > & get_grad_values()

  {
    return this->functions_grad_;
  }
  
  std::vector<DataType>& get_point_coords()
  {
    return this->mapped_pts_;
  }
  
  std::vector<int>& get_point_ind()
  {
    return this->verts_;
  }
  
  std::vector<size_t>& get_cell_offsets()
  {
    return this->cell_offsets_;
  }
  
  std::vector<int>& get_cell_types()
  {
    return this->cell_types_;
  }
  
  int gdim() const 
  {
    return this->gdim_;
  }
  
  int tdim() const 
  {
    return this->tdim_;
  }
  
protected:
  explicit VisualizationData(const MPI_Comm &comm)
  : comm_(comm), tdim_(-1), gdim_(-1)
  {}
  
  int gdim_;
  int tdim_;

  MPI_Comm comm_;
  
  mutable std::map< std::string, std::vector< DataType > > functions_;

  mutable std::map< std::string, std::vector< DataType > > functions_grad_;

  mutable std::map< std::string, std::vector< DataType > > functions_cell_;
  
  mutable std::vector<DataType> mapped_pts_;
  mutable std::vector<int> verts_;
  mutable std::vector<size_t> cell_offsets_;
  mutable std::vector<int> cell_types_;
};


template <class DataType, int DIM>
void VisualizationData<DataType, DIM>::communicate_data(int num_write_procs) const 
{
  int my_rank = -1;
  int num_proc = -1;
  MPI_Comm_rank(this->comm_, &my_rank);
  MPI_Comm_size(this->comm_, &num_proc);

  assert(num_write_procs <= num_proc);

  // determine corresponding writing process for each process
  int recv_proc = -1;
  if (my_rank >= num_write_procs) {
    recv_proc = my_rank % num_write_procs;
  }

  if (recv_proc >= 0) {
    // send data to corresponding writing process
    // send point coordinates
    std::vector<size_t> send_sizes(4);
    send_sizes[0] = this->mapped_pts_.size();
    send_sizes[1] = this->verts_.size();
    send_sizes[2] = this->cell_offsets_.size();
    send_sizes[3] = this->cell_types_.size();

    std::vector<size_t> functions_sizes;
    //std::vector<size_t> functions_grad_sizes;
    std::vector<size_t> functions_cell_sizes;
    for (typename std::map<std::string, std::vector<DataType> >::const_iterator
             it = this->functions_.begin(),
             end_it = this->functions_.end();
         it != end_it; ++it) {
      functions_sizes.push_back(it->second.size());
    }
    /*for (typename std::map<std::string, std::vector<Vec<DIM, DataType>> >::const_iterator
             it = this->functions_grad_.begin(),
             end_it = this->functions_grad_.end();
         it != end_it; ++it) {
      functions_grad_sizes.push_back(it->second.size());
    }*/
    for (typename std::map<std::string, std::vector<DataType> >::const_iterator
             it = this->functions_cell_.begin(),
             end_it = this->functions_cell_.end();
         it != end_it; ++it) {
      functions_cell_sizes.push_back(it->second.size());
    }

    MPI_Send(&send_sizes[0], 4, MPI_UNSIGNED_LONG, recv_proc, 0,
             this->comm_);
    MPI_Send(&functions_sizes[0], this->functions_.size(), MPI_UNSIGNED_LONG,
             recv_proc, 1, this->comm_);
    //MPI_Send(&functions_sizes[0], this->functions_grad_.size(), MPI_UNSIGNED_LONG,
     //        recv_proc, 2, this->comm_);
    MPI_Send(&functions_cell_sizes[0], this->functions_cell_.size(),
             MPI_UNSIGNED_LONG, recv_proc, 2, this->comm_);

    MPI_Send(&this->mapped_pts_[0], this->mapped_pts_.size(), MPI_DOUBLE,
             recv_proc, 3, this->comm_);
    MPI_Send(&this->verts_[0], this->verts_.size(), MPI_INT, recv_proc, 4,
             this->comm_);
    MPI_Send(&this->cell_offsets_[0], this->cell_offsets_.size(),
             MPI_UNSIGNED_LONG, recv_proc, 5, this->comm_);
    MPI_Send(&this->cell_types_[0], this->cell_types_.size(), MPI_INT,
             recv_proc, 6, this->comm_);

    int tag = 7;
    for (typename std::map<std::string, std::vector<DataType> >::const_iterator
             it = this->functions_.begin(),
             end_it = this->functions_.end();
         it != end_it; ++it) {
      MPI_Send(&(it->second[0]), it->second.size(), MPI_DOUBLE, recv_proc, tag,
               this->comm_);
      ++tag;
    }
    
    /*for (typename std::map<std::string, std::vector<Vec<DIM, DataType>> >::const_iterator
             it = this->functions_grad_.begin(),
             end_it = this->functions_grad_.end();
         it != end_it; ++it) {
      MPI_Send(&(it->second[0]), it->second.size(), MPI_DOUBLE, recv_proc, tag,
               this->comm_);
      ++tag;
    }*/
    
    for (typename std::map<std::string, std::vector<DataType> >::const_iterator
             it = this->functions_cell_.begin(),
             end_it = this->functions_cell_.end();
         it != end_it; ++it) {
      MPI_Send(&(it->second[0]), it->second.size(), MPI_DOUBLE, recv_proc, tag,
               this->comm_);
      ++tag;
    }

  } else {
    for (int p = num_write_procs; p < num_proc; ++p) {
      if (p % num_write_procs == my_rank) {
        // proc is writing process: receive data
        std::vector<size_t> send_sizes(4);

        MPI_Status status;
        MPI_Recv(&send_sizes[0], 4, MPI_UNSIGNED_LONG, p, 0, this->comm_,
                 &status);

        std::vector<size_t> functions_sizes(this->functions_.size(), 0);
        //std::vector<size_t> functions_grad_sizes(this->functions_grad_.size(), 0.);
        std::vector<size_t> functions_cell_sizes(this->functions_cell_.size(),
                                                 0);

        MPI_Recv(&functions_sizes[0], this->functions_.size(),
                 MPI_UNSIGNED_LONG, p, 1, this->comm_, &status);
        //MPI_Recv(&functions_grad_sizes[0], this->functions_grad_.size(),
          //       MPI_UNSIGNED_LONG, p, 2, this->comm_, &status);
        MPI_Recv(&functions_cell_sizes[0], this->functions_cell_.size(),
                 MPI_UNSIGNED_LONG, p, 2, this->comm_, &status);

        std::vector<DataType> recv_pt(send_sizes[0], 0.);
        std::vector<int> recv_verts(send_sizes[1], 0);
        std::vector<size_t> recv_offsets(send_sizes[2], 0);
        std::vector<int> recv_types(send_sizes[3], 0);

        MPI_Recv(&recv_pt[0], send_sizes[0], MPI_DOUBLE, p, 3, this->comm_,
                 &status);
        MPI_Recv(&recv_verts[0], send_sizes[1], MPI_INT, p, 4, this->comm_,
                 &status);
        MPI_Recv(&recv_offsets[0], send_sizes[2], MPI_UNSIGNED_LONG, p, 5,
                 this->comm_, &status);
        MPI_Recv(&recv_types[0], send_sizes[3], MPI_INT, p, 6, this->comm_,
                 &status);

        this->mapped_pts_.insert(this->mapped_pts_.end(), recv_pt.begin(),
                                 recv_pt.end());
        this->cell_types_.insert(this->cell_types_.end(), recv_types.begin(),
                                 recv_types.end());

        size_t old_max_vert = this->verts_.size();
        for (size_t l = 0; l < recv_verts.size(); ++l) {
          recv_verts[l] += old_max_vert;
        }
        this->verts_.insert(this->verts_.end(), recv_verts.begin(),
                            recv_verts.end());

        size_t old_offset = 0;
        if (this->cell_offsets_.size() > 0) {
          old_offset = this->cell_offsets_[this->cell_offsets_.size() - 1];
        }
        for (size_t l = 0; l < recv_offsets.size(); ++l) {
          recv_offsets[l] += old_offset;
        }
        this->cell_offsets_.insert(this->cell_offsets_.end(),
                                   recv_offsets.begin(), recv_offsets.end());

        int tag = 7;
        int counter = 0;
        for (typename std::map<std::string, std::vector<DataType> >::iterator
                 it = this->functions_.begin(),
                 end_it = this->functions_.end();
             it != end_it; ++it) {
          size_t count = functions_sizes[counter];
          std::vector<DataType> recv_functions(count, 0.);
          MPI_Recv(&recv_functions[0], count, MPI_DOUBLE, p, tag,
                   this->comm_, &status);

          it->second.insert(it->second.end(), recv_functions.begin(),
                            recv_functions.end());
          ++counter;
          ++tag;
        }
/*
        counter = 0;
        for (typename std::map<std::string, std::vector<Vec<DIM, DataType>>>::iterator
                 it = this->functions_grad_.begin(),
                 end_it = this->functions_grad_.end();
             it != end_it; ++it) {
          size_t count = functions_grad_sizes[counter];
          std::vector<Vec<DIM, DataType>> recv_functions_grad(count);
          MPI_Recv(&recv_functions_grad[0], count, MPI_DOUBLE, p, tag,
                   this->comm_, &status);

          it->second.insert(it->second.end(), recv_functions_grad.begin(),
                            recv_functions_grad.end());
          ++counter;
          ++tag;
        }
        */
        counter = 0;
        for (typename std::map<std::string, std::vector<DataType> >::iterator
                 it = this->functions_cell_.begin(),
                 end_it = this->functions_cell_.end();
             it != end_it; ++it) {
          size_t count = functions_cell_sizes[counter];
          std::vector<DataType> recv_functions_cell(count, 0.);
          MPI_Recv(&recv_functions_cell[0], count, MPI_DOUBLE, p, tag,
                   this->comm_, &status);

          it->second.insert(it->second.end(), recv_functions_cell.begin(),
                            recv_functions_cell.end());
          ++counter;
          ++tag;
        }
      }
    }
  }
}

} // namespace hiflow

#endif
