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

#include "config.h"

#include <fstream>
#include <iostream>
#include <list>
#include <map>
#include <mpi.h>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <vector>
#include <iterator>

#include "common/log.h"
#include "visualization/xdmf_writer.h"
#include "mesh/attributes.h"
#include "mesh/entity.h"
#include "mesh/iterator.h"

#ifdef WITH_HDF5
#include "common/hdf5_tools.h"
#else
#define ERROR                                                                  \
  LOG_ERROR("HiFlow was not compiled with HDF5 support!");                     \
  exit(-1);
#endif

#include "linear_algebra/coupled_vector.h"
#ifdef WITH_HYPRE
#include "linear_algebra/hypre_vector.h"
#endif

#include <tinyxml2.h>

namespace hiflow {

// clear topology data
template <class DataType, int DIM>
void XDMFWriter<DataType, DIM>::clear_topology_data() 
{
  this->xdmf_topology_.clear();
}

// clear
template <class DataType, int DIM>
void XDMFWriter<DataType, DIM>::clear() 
{
  this->clear_topology_data();
}

// get global dimensions
template <class DataType,int DIM>
void XDMFWriter<DataType, DIM>::get_global_dimensions() 
{

  const int local_element = this->cell_types_.size();
  const int local_topology = this->verts_.size() + this->cell_types_.size();
  const int local_geometry = this->mapped_pts_.size();

  // allreduce from all processors
  MPI_Allreduce(&local_element, &this->global_element_, 1, MPI_INT, MPI_SUM,
                this->comm_);
  MPI_Allreduce(&local_topology, &this->global_topology_, 1, MPI_INT, MPI_SUM,
                this->comm_);
  MPI_Allreduce(&local_geometry, &this->global_geometry_, 1, MPI_INT, MPI_SUM,
                this->comm_);

  // pointwise
  for (typename std::map<std::string, std::vector<DataType> >::const_iterator it =
         this->functions_.begin(),  end_it = this->functions_.end();
       it != end_it; ++it) {

    const int local_attribute = it->second.size();
    int tmp_dim_attr;
    MPI_Allreduce(&local_attribute, &tmp_dim_attr, 1, MPI_INT, MPI_SUM,
                  this->comm_);

    this->global_attributes_[it->first] = tmp_dim_attr;

  }
  
  for (typename std::map<std::string, std::vector< DataType >>::const_iterator it =
         this->functions_grad_.begin(),  end_it = this->functions_grad_.end();
       it != end_it; ++it) {

    const int local_attribute = it->second.size();
    int tmp_dim_attr;
    MPI_Allreduce(&local_attribute, &tmp_dim_attr, 1, MPI_INT, MPI_SUM,
                  this->comm_);

    this->global_attributes_grad_[it->first] = tmp_dim_attr;

  }

  // cellwise
  for (typename std::map<std::string, std::vector<DataType> >::const_iterator it =
         this->functions_cell_.begin(),  end_it = this->functions_cell_.end();
       it != end_it; ++it) {

    this->global_attributes_cell_[it->first] = this->global_element_;

  }

}

// get offsets
template <class DataType, int DIM>
void XDMFWriter<DataType, DIM>::get_offsets() 
{

  // mpi rank
  int my_rank = -1;
  int size_rank = -1;
  MPI_Comm_rank(this->comm_, &my_rank);
  MPI_Comm_size(this->comm_, &size_rank);

  // Topology
  this->offset_topology_ = 0;
  if (my_rank > 0) {
    MPI_Status status;
    MPI_Recv(&this->offset_topology_, 1, MPI_INT, my_rank - 1, my_rank - 1,
             this->comm_, &status);
  }
  int next_offset_topology = this->offset_topology_ + this->verts_.size() +
                             this->cell_types_.size();
  if (my_rank < size_rank - 1) {
    MPI_Send(&next_offset_topology, 1, MPI_INT, (my_rank + 1), my_rank,
             this->comm_);
  }

  // Geometry
  this->offset_geometry_ = 0;
  if (my_rank > 0) {
    MPI_Status status;
    MPI_Recv(&this->offset_geometry_, 1, MPI_INT, my_rank - 1, my_rank - 1,
             this->comm_, &status);
  }
  int next_offset_geometry = this->offset_geometry_ + this->mapped_pts_.size();
  if (my_rank < size_rank - 1) {
    MPI_Send(&next_offset_geometry, 1, MPI_INT, (my_rank + 1), my_rank,
             this->comm_);
  }

  // Attributes
  // pointwise
  for (typename std::map<std::string, std::vector<DataType> >::const_iterator it =
         this->functions_.begin(),  end_it = this->functions_.end();
       it != end_it; ++it) {

    int offset_attr = 0;
    if (my_rank > 0) {
      MPI_Status status;
      MPI_Recv(&offset_attr, 1, MPI_INT, my_rank - 1, my_rank - 1, this->comm_,
               &status);
    }
    int next_offset_attr = offset_attr + it->second.size();
    if (my_rank < size_rank - 1) {
      MPI_Send(&next_offset_attr, 1, MPI_INT, (my_rank + 1), my_rank,
               this->comm_);
    }

    this->offset_attributes_[it->first] = offset_attr;

  }
  
  for (typename std::map<std::string, std::vector< DataType >>::const_iterator it =
         this->functions_grad_.begin(),  end_it = this->functions_grad_.end();
       it != end_it; ++it) {

    int offset_attr = 0;
    if (my_rank > 0) {
      MPI_Status status;
      MPI_Recv(&offset_attr, 1, MPI_INT, my_rank - 1, my_rank - 1, this->comm_,
               &status);
    }
    int next_offset_attr = offset_attr + it->second.size();
    if (my_rank < size_rank - 1) {
      MPI_Send(&next_offset_attr, 1, MPI_INT, (my_rank + 1), my_rank,
               this->comm_);
    }

    this->offset_attributes_grad_[it->first] = offset_attr;

  }

  // cellwise
  for (typename std::map<std::string, std::vector<DataType> >::const_iterator it =
         this->functions_cell_.begin(),  end_it = this->functions_cell_.end();
       it != end_it; ++it) {

    int offset_attr = 0;
    if (my_rank > 0) {
      MPI_Status status;
      MPI_Recv(&offset_attr, 1, MPI_INT, my_rank - 1, my_rank - 1, this->comm_,
               &status);
    }
    int next_offset_attr = offset_attr + it->second.size();
    if (my_rank < size_rank - 1) {
      MPI_Send(&next_offset_attr, 1, MPI_INT, (my_rank + 1), my_rank,
               this->comm_);
    }

    this->offset_attributes_cell_[it->first] = offset_attr;

  }
}

// create xdmf topology
template <class DataType, int DIM>
void XDMFWriter<DataType, DIM>::create_xdmf_topology(const int& offset) const 
{

  std::map<int, int> cell_types_vtk_to_xdmf;
  cell_types_vtk_to_xdmf[1] = 1;
  cell_types_vtk_to_xdmf[3] = 2;
  cell_types_vtk_to_xdmf[5] = 4;
  cell_types_vtk_to_xdmf[9] = 5;
  cell_types_vtk_to_xdmf[10] = 6;
  cell_types_vtk_to_xdmf[12] = 9;
  cell_types_vtk_to_xdmf[14] = 7;

  this->xdmf_topology_.clear();
  this->xdmf_topology_.reserve(this->verts_.size() + this->cell_types_.size());

  for (int i = 0; i != this->cell_types_.size(); ++i) {
    assert(this->cell_types_.size() == this->cell_offsets_.size());

    /// adding xdmf cell type
    this->xdmf_topology_.push_back(cell_types_vtk_to_xdmf[this->cell_types_[i]]);

    /// adding vertices
    const int pos_begin = (i - 1 < 0) ? 0 : this->cell_offsets_[i-1];
    const int pos_end = this->cell_offsets_[i];

    for (int j = pos_begin; j != pos_end; ++j) {

      this->xdmf_topology_.push_back(this->verts_[j] + offset);

    }
  }
}

/// \details This function writes the xdmf file and hdf5 data. In the simplest 
/// case, only filename has to be specified, the other three parameters have 
/// default values. In case only the filename is provided, the mesh and 
/// solution will be written in same hdf5 file. Filepath is used to set user
/// defined output path. Once filename_mesh is provided, the solution vector
/// and mesh information are written in different hdf5 files. write_mesh
/// should be used together with filename_mesh, it decides whether the mesh
/// data should be written into a hdf5 file. If not, the xdmf file will 
/// simply link the mesh data as filename_mesh provides.
///
///
/// \param filename       output file name.  
/// \param filepath       output path
/// \param filename_mesh  mesh data name
/// \param write_mesh     flag whether the mesh data should be written
template <class DataType, int DIM>
void XDMFWriter<DataType, DIM>::write(const std::string& filename,
                                        const std::string& filepath,
                                        const std::string& filename_mesh,
                                        const bool& write_mesh) {

  // set file names
  std::string name_hdf5 = filename;
  name_hdf5 += ".h5";

  std::string name_xdmf = filename;
  name_xdmf += ".xdmf";

  std::string name_hdf5_mesh = filename_mesh;
  if (filename_mesh.empty()) {
    name_hdf5_mesh = name_hdf5;
  } else {
    name_hdf5_mesh += ".h5";
  }


  // get global dimension and offsets for writing hdf5 file
  this->get_global_dimensions();
  this->get_offsets();

  // create xdmf topology, because it is not contained in CellVisualization
  this->create_xdmf_topology(this->offset_geometry_ / 3);


  // mpi rank
  int my_rank = -1;
  int size_rank = -1;
  MPI_Comm_rank(this->comm_, &my_rank);
  MPI_Comm_size(this->comm_, &size_rank);

#ifdef WITH_HDF5

  // write hdf5 file
  // write mesh
  if (write_mesh) {

    H5FilePtr file_ptr_mesh(new H5File(filepath + name_hdf5_mesh, "w",
                                       this->comm_));

    // create group name for Topology and Geometry
    H5GroupPtr group_ptr_mesh(new H5Group(file_ptr_mesh, "mesh", "w"));

    // write Topology
    H5DatasetPtr dataset_ptr_topology(new H5Dataset(group_ptr_mesh,
                                      this->global_topology_, "topology", "w",
                                      &this->xdmf_topology_[0]));

    dataset_ptr_topology->write(this->xdmf_topology_.size(), this->offset_topology_,
                                &this->xdmf_topology_[0]);

    // write Geometry
    H5DatasetPtr dataset_ptr_geometry(new H5Dataset(group_ptr_mesh,
                                      this->global_geometry_, "geometry", "w", 
				      &this->mapped_pts_[0]));

    dataset_ptr_geometry->write(this->mapped_pts_.size(), this->offset_geometry_,
                                &this->mapped_pts_[0]);

  } // if (write_mesh)

  // write solutions
  H5FilePtr file_ptr(new H5File(filepath + name_hdf5, "w", this->comm_));

  // pointwise
  H5GroupPtr group_ptr_solution(new H5Group(file_ptr, "solution", "w"));
  for (typename std::map<std::string, std::vector<DataType> >::const_iterator it =
         this->functions_.begin(),  end_it = this->functions_.end();
       it != end_it; ++it) {

    H5DatasetPtr dataset_ptr_solution(new H5Dataset(group_ptr_solution,
                                      this->global_attributes_[it->first], it->first.c_str(),
                                      "w", vec2ptr(it->second)));

    dataset_ptr_solution->write(it->second.size(),
                                this->offset_attributes_[it->first],
                                vec2ptr(it->second));
  }
  
  for (typename std::map<std::string, std::vector< DataType > >::const_iterator it =
         this->functions_grad_.begin(),  end_it = this->functions_grad_.end();
       it != end_it; ++it) {

    H5DatasetPtr dataset_ptr_solution(new H5Dataset(group_ptr_solution,
                                      this->global_attributes_grad_[it->first], it->first.c_str(),
                                      "w", vec2ptr(it->second)));

    dataset_ptr_solution->write(it->second.size(),
                                this->offset_attributes_grad_[it->first],
                                vec2ptr(it->second));
  }

  // cellwise
  H5GroupPtr group_ptr_cell_solution(new H5Group(file_ptr, "cell_solution", "w"));
  for (typename std::map<std::string, std::vector<DataType> >::const_iterator it =
         this->functions_cell_.begin(),  end_it = this->functions_cell_.end();
       it != end_it; ++it) {

    H5DatasetPtr dataset_ptr_cell_solution(new H5Dataset(group_ptr_cell_solution,
                                           this->global_attributes_cell_[it->first], 
					   it->first.c_str(),
                                           "w", vec2ptr(it->second)));

    dataset_ptr_cell_solution->write(it->second.size(),
                                     this->offset_attributes_cell_[it->first],
                                     vec2ptr(it->second));
  }

#else
  ERROR;
#endif

  // writing xdmf file
  if (my_rank == this->master_rank_) {

    // header of xdmf file
    tinyxml2::XMLDeclaration *decl = this->xdmf_file_.NewDeclaration();
    this->xdmf_file_.InsertEndChild(decl);

    tinyxml2::XMLElement* root = this->xdmf_file_.NewElement("Xdmf");
    root->SetAttribute("xmlns:xi", "http://www.w3.org/2001/XInclude");
    root->SetAttribute("Version", "2.0");

    // Domain
    tinyxml2::XMLElement* domain = this->xdmf_file_.NewElement("Domain");

    // Grid
    tinyxml2::XMLElement* grid = this->xdmf_file_.NewElement("Grid");

    grid->SetAttribute("Name", "Mesh");
    grid->SetAttribute("GridType", "Uniform");

    // write Topology
    tinyxml2::XMLElement* topology = this->xdmf_file_.NewElement("Topology");
    topology->SetAttribute("TopologyType", "Mixed");
    topology->SetAttribute("NumberOfElements", this->global_element_);

    this->write_xdmf_dataitem(topology, this->global_topology_, "Int", "HDF",
                              name_hdf5_mesh, "mesh", "topology");

    grid->InsertEndChild(topology);

    // write Geometry
    tinyxml2::XMLElement* geometry = this->xdmf_file_.NewElement("Geometry");
    geometry->SetAttribute("Type", "XYZ");
    grid->InsertEndChild(geometry);

    this->write_xdmf_dataitem(geometry, this->global_geometry_, "Float", "HDF",
                              name_hdf5_mesh, "mesh", "geometry", 8);


    // write solutions
    // pointwise
    for (typename std::map<std::string, std::vector<DataType> >::const_iterator it =
           this->functions_.begin(),  end_it = this->functions_.end();
         it != end_it; ++it) {

      tinyxml2::XMLElement* attribute = this->xdmf_file_.NewElement("Attribute");
      attribute->SetAttribute("Name", it->first.c_str());
      attribute->SetAttribute("AttributeType", "Scalar");
      attribute->SetAttribute("Center", "Node");

      this->write_xdmf_dataitem(attribute, this->global_attributes_[it->first],
                                "Float", "HDF",
                                name_hdf5, "solution", it->first.c_str(), 8);

      grid->InsertEndChild(attribute);
    }
    
    for (typename std::map<std::string, std::vector< DataType >>::const_iterator it =
           this->functions_grad_.begin(),  end_it = this->functions_grad_.end();
         it != end_it; ++it) {

      tinyxml2::XMLElement* attribute = this->xdmf_file_.NewElement("Attribute");
      attribute->SetAttribute("Name", it->first.c_str());
      attribute->SetAttribute("Type", "XYZ");
      attribute->SetAttribute("Center", "Node");
      attribute->SetAttribute("AttributeType", "Vector");
      
      
      this->write_xdmf_dataitem(attribute, this->global_attributes_grad_[it->first],
                                "Float", "HDF",
                                name_hdf5, "solution", it->first.c_str(), 8);

      grid->InsertEndChild(attribute);
    }

    // cellwise
    for (typename std::map<std::string, std::vector<DataType> >::const_iterator it =
           this->functions_cell_.begin(),  end_it = this->functions_cell_.end();
         it != end_it; ++it) {

      tinyxml2::XMLElement* attribute = this->xdmf_file_.NewElement("Attribute");
      attribute->SetAttribute("Name", it->first.c_str());
      attribute->SetAttribute("AttributeType", "Scalar");
      attribute->SetAttribute("Center", "Cell");

      this->write_xdmf_dataitem(attribute, this->global_attributes_cell_[it->first],
                                "Float", "HDF",
                                name_hdf5, "cell_solution", it->first.c_str(), 8);

      grid->InsertEndChild(attribute);
    }

    domain->InsertEndChild(grid);

    root->InsertEndChild(domain);
    this->xdmf_file_.InsertEndChild(root);

    this->xdmf_file_.SaveFile((filepath + name_xdmf).c_str());

  }

}

template <class DataType, int DIM>
void XDMFWriter<DataType, DIM>::write_xdmf_dataitem(tinyxml2::XMLElement*
    xdmf_element, const int& dim, const std::string& nb_type,
    const std::string& format, const std::string& filename,
    const std::string& groupname, const std::string& datasetname,
    const int& precision) {

  tinyxml2::XMLElement *dataitem = this->xdmf_file_.NewElement("DataItem");

  dataitem->SetAttribute("Dimensions", dim);
  dataitem->SetAttribute("NumberType", nb_type.c_str());

  if (precision > 0) {
    dataitem->SetAttribute("Precision", precision);
  }

  dataitem->SetAttribute("Format", format.c_str());

  std::string data_name;
  data_name += filename;
  data_name += ":";
  data_name += "/";
  data_name += groupname;
  data_name += "/";
  data_name += datasetname;

  tinyxml2::XMLText *data = this->xdmf_file_.NewText(data_name.c_str());

  dataitem->InsertEndChild(data);

  xdmf_element->InsertEndChild(dataitem);
}

template class XDMFWriter<float,1>;
template class XDMFWriter<float,2>;
template class XDMFWriter<float,3>;
template class XDMFWriter<double,1>;
template class XDMFWriter<double,2>;
template class XDMFWriter<double,3>;

} // namespace hiflow
