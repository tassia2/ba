#include "visualization/vtk_writer.h"

#include <map>
#include <string>
#include <vector>
#include <cmath>
#include <limits>
#include <sstream>
#include <tinyxml2.h>
#include <boost/function.hpp>
#include <mpi.h>

#include "common/log.h"
#include "mesh/types.h"
#include "mesh/mesh.h"
#include "mesh/attributes.h"

namespace hiflow {

using namespace mesh;

template <class DataType, int DIM>
void VTKWriter<DataType, DIM>::write_vtu(const std::string &filename) const 
{

  size_t num_grid_points = this->mapped_pts_.size() / 3;
  size_t num_grid_cells = this->cell_types_.size();

  if (num_grid_points == 0) {
    LOG_INFO("Write", "No grid points visualized");
    return;
  }

  const mesh::TDim tdim = this->visu_.tdim();
  const mesh::GDim gdim = this->visu_.gdim();

  tinyxml2::XMLDocument doc;
  tinyxml2::XMLDeclaration *decl = doc.NewDeclaration();
  doc.LinkEndChild(decl);

  tinyxml2::XMLElement *vtkFile = doc.NewElement("VTKFile");
  vtkFile->SetAttribute("type", "UnstructuredGrid");
  vtkFile->SetAttribute("version", "0.1");
  vtkFile->SetAttribute("byte_order", "LittleEndian");
  vtkFile->SetAttribute("compressor", "vtkZLibDataCompressor");
  doc.LinkEndChild(vtkFile);

  tinyxml2::XMLElement *umesh = doc.NewElement("UnstructuredGrid");
  vtkFile->LinkEndChild(umesh);

  tinyxml2::XMLElement *piece = doc.NewElement("Piece");

  // TODO: checken
  piece->SetAttribute("NumberOfPoints", static_cast<int>(num_grid_points));
  piece->SetAttribute("NumberOfCells", static_cast<int>(num_grid_cells));

  umesh->LinkEndChild(piece);

  // Scratch variables for data.
  tinyxml2::XMLElement *data_array;
  std::stringstream os;

  //// Write points ////////////////////////////////////////////////////////////
  tinyxml2::XMLElement *points = doc.NewElement("Points");
  piece->LinkEndChild(points);
  data_array = doc.NewElement("DataArray");

  // Set correct length of float in dependence of DataType
  std::ostringstream type_float;
  type_float << "Float" << sizeof(DataType) * 8;
  data_array->SetAttribute("type", type_float.str().c_str());

  data_array->SetAttribute("Name", "Array");
  // always 3 comps, since vtk doesn:t handle 2D.
  data_array->SetAttribute("NumberOfComponents", "3");

  data_array->SetAttribute("format", "ascii");

  DataType range_min = std::numeric_limits<DataType>::max();
  DataType range_max = std::numeric_limits<DataType>::min();
  int cell_type_min = std::numeric_limits<int>::max();
  int cell_type_max = std::numeric_limits<int>::min();

  size_t cell_offset_max = std::numeric_limits<size_t>::min();

  //std::cout << "num grid points " << num_grid_points << std::endl;
  
  for (size_t p = 0; p < num_grid_points; ++p) {
    for (int c = 0; c < 3; ++c) {
      range_min = std::min(range_min, this->mapped_pts_[3 * p + c]);
      range_max = std::max(range_max, this->mapped_pts_[3 * p + c]);

      os << this->mapped_pts_[3 * p + c] << " ";
    }
  }
  tinyxml2::XMLText *coords;
  coords = doc.NewText(os.str().c_str());

  data_array->SetAttribute("RangeMin", range_min);
  data_array->SetAttribute("RangeMax", range_max);

  points->LinkEndChild(data_array);
  data_array->LinkEndChild(coords);
  os.str("");
  os.clear();
  //// End write points
  ////////////////////////////////////////////////////////////

  //// Write cells
  /////////////////////////////////////////////////////////////////
  tinyxml2::XMLElement *cells = doc.NewElement("Cells");
  piece->LinkEndChild(cells);

  // Connectivity, Offsets, and Types arrays
  std::ostringstream off_os, type_os;

  for (size_t c = 0; c < num_grid_cells; ++c) {
    type_os << this->cell_types_[c] << " ";

    cell_type_min = std::min(cell_type_min, this->cell_types_[c]);
    cell_type_max = std::max(cell_type_max, this->cell_types_[c]);
  }

  for (size_t v = 0; v != this->verts_.size(); ++v) {
    os << this->verts_[v] << " ";
  }

  for (size_t c = 0; c != this->cell_offsets_.size(); ++c) {
    off_os << this->cell_offsets_[c] << " ";
    cell_offset_max = std::max(cell_offset_max, this->cell_offsets_[c]);
  }

  data_array = doc.NewElement("DataArray");
  data_array->SetAttribute("type", "Int64");
  data_array->SetAttribute("Name", "connectivity");
  data_array->SetAttribute("format", "ascii");
  data_array->SetAttribute("RangeMin", 0);
  data_array->SetAttribute("RangeMax", static_cast<int>(num_grid_points));

  tinyxml2::XMLText *conns = doc.NewText(os.str().c_str());
  data_array->LinkEndChild(conns);
  cells->LinkEndChild(data_array);
  os.str("");
  os.clear();

  data_array = doc.NewElement("DataArray");
  data_array->SetAttribute("type", "Int64");
  data_array->SetAttribute("Name", "offsets");
  data_array->SetAttribute("format", "ascii");
  data_array->SetAttribute("RangeMin", 0);
  data_array->SetAttribute("RangeMax", static_cast<int>(cell_offset_max));

  tinyxml2::XMLText *offs = doc.NewText(off_os.str().c_str());
  data_array->LinkEndChild(offs);
  cells->LinkEndChild(data_array);
  off_os.str("");
  off_os.clear();

  data_array = doc.NewElement("DataArray");
  data_array->SetAttribute("type", "UInt8");
  data_array->SetAttribute("Name", "types");
  data_array->SetAttribute("format", "ascii");
  data_array->SetAttribute("RangeMin", cell_type_min);
  data_array->SetAttribute("RangeMax", cell_type_max);

  tinyxml2::XMLText *types = doc.NewText(type_os.str().c_str());
  data_array->LinkEndChild(types);
  cells->LinkEndChild(data_array);
  type_os.str("");
  type_os.clear();

  //// End Write cells
  /////////////////////////////////////////////////////////////

  //// Write point data
  ////////////////////////////////////////////////////////////
  tinyxml2::XMLElement *point_data = doc.NewElement("PointData");
  piece->LinkEndChild(point_data);

  for (typename std::map<std::string, std::vector<DataType> >::const_iterator
           it = this->functions_.begin(),
           end_it = this->functions_.end();
       it != end_it; ++it) 
  {
    data_array = doc.NewElement("DataArray");
    data_array->SetAttribute("Name", it->first.c_str());
    data_array->SetAttribute("type", type_float.str().c_str());
    data_array->SetAttribute("format", "ascii");

    //std::cout << "num data points " << it->second.size() << std::endl;
    
    for (size_t i = 0, end_i = it->second.size(); i != end_i; ++i) 
    {
      os << (it->second)[i] << " ";
    }

    tinyxml2::XMLText *data = doc.NewText(os.str().c_str());
    data_array->LinkEndChild(data);
    point_data->LinkEndChild(data_array);
    os.str("");
    os.clear();
  }

  //tinyxml2::XMLElement *point_data_grad = doc.NewElement("PointDataGrad");
  //piece->LinkEndChild(point_data_grad);

  for (typename std::map<std::string, std::vector<DataType> >::const_iterator
           it = this->functions_grad_.begin(),
           end_it = this->functions_grad_.end();
       it != end_it; ++it) {
    data_array = doc.NewElement("DataArray");
    data_array->SetAttribute("Name", it->first.c_str());
    data_array->SetAttribute("type", type_float.str().c_str());
    data_array->SetAttribute("format", "ascii");

    data_array->SetAttribute("NumberOfComponents", 3);    //VTK apparently only handles 3D data

    for (size_t i = 0; i < num_grid_points; ++i) {
      for (size_t d = 0; d< 3; ++d) {
        os << (it->second)[3*i +d] << " ";
      }
 

    }

    tinyxml2::XMLText *data = doc.NewText(os.str().c_str());
    data_array->LinkEndChild(data);

    point_data->LinkEndChild(data_array);
    os.str("");
    os.clear();
  }

  tinyxml2::XMLElement *cell_data = doc.NewElement("CellData");
  piece->LinkEndChild(cell_data);

  for (typename std::map<std::string, std::vector<DataType> >::const_iterator
           it = this->functions_cell_.begin(),
           end_it = this->functions_cell_.end();
       it != end_it; ++it) {

    data_array = doc.NewElement("DataArray");
    data_array->SetAttribute("Name", it->first.c_str());
    data_array->SetAttribute("type", type_float.str().c_str());
    data_array->SetAttribute("format", "ascii");

    for (size_t i = 0, end_i = it->second.size(); i != end_i; ++i) {
      os << (it->second)[i] << " ";
    }

    tinyxml2::XMLText *data = doc.NewText(os.str().c_str());
    data_array->LinkEndChild(data);
    cell_data->LinkEndChild(data_array);
    os.str("");
    os.clear();
  }
  doc.SaveFile(filename.c_str());
}

template <class DataType, int DIM>
void VTKWriter<DataType, DIM>::write_pvtu(const std::string &filename, 
                                          const std::string &path_pvtu,
                                          const std::string &path_pvtu2path_vtu, 
                                          int num_writers) const 
{
                                                        
  // const mesh::Mesh& mesh = this->space_.mesh();
  // const mesh::TDim tdim = mesh.tdim();
  // const mesh::GDim gdim = mesh.gdim();
  // get MPI rank
  int rank = -1, num_procs = -1;
  MPI_Comm_rank(comm_, &rank);
  MPI_Comm_size(comm_, &num_procs);

  std::stringstream s;
  s << rank;

  // get the correct filename including the path
  std::istringstream filename_root_dir(filename);

  std::size_t dir = filename_root_dir.str().find_last_of('.');
  LOG_DEBUG(1, "Filename: " << filename);

  std::string filename_without_suffix = filename_root_dir.str().substr(0, dir);

  assert(!filename_without_suffix.empty());

  std::string str_src_filename =
      (path_pvtu + path_pvtu2path_vtu + filename_without_suffix + "_" + s.str() + ".vtu");
  LOG_DEBUG(1, "Filename without suffix: " << filename_without_suffix);
  assert(!str_src_filename.empty());

  int num_write_procs = num_procs;
  if (num_writers <= 0) 
  {
    // Each process writes its vtu file
    this->write_vtu(str_src_filename);
  } 
  else 
  {
    num_write_procs = std::min(num_writers, num_procs);

    // send data to writing procs
    this->visu_.communicate_data(num_write_procs);

    // write data
    if (rank < num_write_procs) {
      this->write_vtu(str_src_filename);
    }
  }

  // Master writes pvtu file
  if (rank == master_rank_) 
  {
    tinyxml2::XMLDocument doc;
    tinyxml2::XMLDeclaration *decl = doc.NewDeclaration();
    doc.LinkEndChild(decl);

    tinyxml2::XMLElement *vtkFile = doc.NewElement("VTKFile");
    vtkFile->SetAttribute("type", "PUnstructuredGrid");
    vtkFile->SetAttribute("version", "0.1");
    vtkFile->SetAttribute("byte_order", "LittleEndian");
    vtkFile->SetAttribute("compressor", "vtkZLibDataCompressor");
    doc.LinkEndChild(vtkFile);

    tinyxml2::XMLElement *pumesh = doc.NewElement("PUnstructuredGrid");
    // GhostLevel in PUnstructuredGrid is always 0
    pumesh->SetAttribute("GhostLevel", 0);
    vtkFile->LinkEndChild(pumesh);

    tinyxml2::XMLElement *p_point_data = doc.NewElement("PPointData");
    pumesh->LinkEndChild(p_point_data);

    // Set correct length of float in dependence of DataType
    std::ostringstream type_float;
    type_float << "Float" << sizeof(DataType) * 8;

    tinyxml2::XMLElement *p_data_array;
    for (typename std::map<std::string, std::vector<DataType> >::const_iterator
             it = this->functions_.begin(),
             end_it = this->functions_.end();
         it != end_it; ++it) {

      p_data_array = doc.NewElement("PDataArray");
      p_data_array->SetAttribute("Name", it->first.c_str());
      p_data_array->SetAttribute("type", type_float.str().c_str());
      p_data_array->SetAttribute("format", "ascii");
      p_point_data->LinkEndChild(p_data_array);
    }

    /*tinyxml2::XMLElement *p_point_data_grad = doc.NewElement("PPointDataGrad");
    pumesh->LinkEndChild(p_point_data_grad);
    
    tinyxml2::XMLElement *p_data_array_grad;*/
    for (typename std::map<std::string, std::vector<DataType > >::const_iterator
             it = this->functions_grad_.begin(),
             end_it = this->functions_grad_.end();
         it != end_it; ++it) {


      p_data_array = doc.NewElement("PDataArray");
      p_data_array->SetAttribute("Name", it->first.c_str());
      p_data_array->SetAttribute("type", type_float.str().c_str());
      p_data_array->SetAttribute("format", "ascii");
      p_data_array->SetAttribute("NumberOfComponents", 3);
      p_point_data->LinkEndChild(p_data_array);
    }

    tinyxml2::XMLElement *p_cell_data = doc.NewElement("PCellData");
    pumesh->LinkEndChild(p_cell_data);
    // int tdim = mesh.tdim();

    // write cell data
    // TODO currently only Float64 is supported
    tinyxml2::XMLElement *data_array;
    for (typename std::map<std::string, std::vector<DataType> >::const_iterator
             it = this->functions_cell_.begin();
         it != this->functions_cell_.end(); ++it) {
      data_array = doc.NewElement("PDataArray");
      data_array->SetAttribute("Name", it->first.c_str());
      data_array->SetAttribute("type", type_float.str().c_str());
      data_array->SetAttribute("format", "ascii");
      p_cell_data->LinkEndChild(data_array);
      p_cell_data->SetAttribute("Scalars", it->first.c_str());
    }

    // NB: This has to be AFTER the the other elements, since
    // the same order in the vtu and pvtu file is needed!

    tinyxml2::XMLElement *p_points = doc.NewElement("PPoints");
    pumesh->LinkEndChild(p_points);

    tinyxml2::XMLElement *p_points_data_array = doc.NewElement("PDataArray");
    p_points_data_array->SetAttribute("type", type_float.str().c_str());
    p_points_data_array->SetAttribute("NumberOfComponents", "3");
    p_points->LinkEndChild(p_points_data_array);

    // get the correct filename without the path
    std::size_t pos = filename_root_dir.str().find_last_of("/\\");
    assert(!filename_root_dir.str()
                .substr(pos + 1, filename_root_dir.str().length())
                .empty());

    std::stringstream str_proc_id;
    for (int proc_id = 0; proc_id < num_write_procs; ++proc_id) {
      tinyxml2::XMLElement *piece =
          doc.NewElement("Piece"); // needs to be inside the loop!
      str_proc_id << proc_id;
      std::string source_str =
          path_pvtu2path_vtu +
          filename_root_dir.str().substr(pos + 1, dir - pos - 1) + "_" +
          str_proc_id.str() + ".vtu";
      piece->SetAttribute("Source", source_str.c_str());
      pumesh->LinkEndChild(piece);
      str_proc_id.str("");
      str_proc_id.clear();
    }

    const std::string &tmp_path_pvtu = path_pvtu;
    std::string str_filename = (tmp_path_pvtu + filename_without_suffix + ".pvtu");
    LOG_DEBUG(1,  "Write pvtu to " << str_filename.c_str());
    
    FILE *pFile;
    pFile = fopen(str_filename.c_str(), "w");
    if (pFile != nullptr) {
      doc.SaveFile(pFile);
      fclose(pFile);
    } else {
      std::stringstream err;
      err << "Path to write the files (" << str_filename << ") does not exist!";
      LOG_ERROR(err.str());
      throw std::runtime_error(err.str());
    }
  }
}

template class VTKWriter <float,1>;
template class VTKWriter <float,2>;
template class VTKWriter <float,3>;
template class VTKWriter <double,1>;
template class VTKWriter <double,2>;
template class VTKWriter <double,3>;

} // namespace hiflow

