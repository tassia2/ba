#include "pc_mesh.h"

namespace hiflow {
namespace polynomialchaos {

// get refinement level for every index > 0
int get_ref_level(unsigned int index) {
  assert(index != 0U);
  int level = 0;
  while (index > 1U) {
    level++;
    index = index >> 1;
  }
  // index should be 1 here
  assert(index == 1U);
  return level;
}

// ************************************
// ************** PCCell **************
// ************************************
PCCell::PCCell() {
  ref_dim_ = -1;
  left_child_ = NULL;
  right_child_ = NULL;
}

void PCCell::set_ref_type(int type) {
  if (left_child_ == NULL) {
    left_child_ = new PCCell();
    right_child_ = new PCCell();
  }
  ref_dim_ = type;
}

void PCCell::clean_deletion() {
  if (ref_dim_ != -1) {
    left_child_->clean_deletion();
    right_child_->clean_deletion();
    delete left_child_;
    delete right_child_;
  }
}

const PCCell *PCCell::get_cell(unsigned int index) const {
  assert(index != 0);
  if (index == 1U) {
    return this;
  }
  if ((index & 1) == 0u) // check "last" bit to get the next cell
  {
    return left_child_->get_cell(index >> 1); // remove last bit and redo
  }
  return right_child_->get_cell(index >> 1);
}

void PCCell::ref_cell(unsigned int index, int ref_dim) {
  if (index == 1U) {
    if (this->ref_dim_ == -1) {
      this->set_ref_type(ref_dim);
      return;
    }
    // std::cerr << "Err: Cell already refined!" << std::endl;
    return;
  }
  if ((index & 1) == 0u) {
    left_child_->ref_cell(index >> 1, ref_dim);
  } else {
    right_child_->ref_cell(index >> 1, ref_dim);
  }
}

void PCCell::get_cell_coordinates(unsigned int index,
                                  std::vector< double > &origin,
                                  std::vector< double > &extents) const {
  if (index == 1U) {
    return;
  }

  extents[this->ref_dim_] /= 2;
  if ((index & 1) == 0u) {
    left_child_->get_cell_coordinates(index >> 1, origin, extents);
  } else {
    origin[this->ref_dim_] += extents[ref_dim_];
    right_child_->get_cell_coordinates(index >> 1, origin, extents);
  }
}

void PCCell::print_cell(unsigned int index, std::vector< double > origin,
                        std::vector< double > extents) const {
  this->get_cell_coordinates(index, origin, extents);
  std::cout << "Cell with index " << index << std::endl;
  std::cout << "With ref_dim_ " << this->get_cell(index)->get_ref_dim()
            << std::endl;
  std::cout << "Origin: ";
  for (size_t i = 0; i < origin.size(); ++i)
    std::cout << origin[i] << " ";
  std::cout << std::endl;
  std::cout << "extents: ";
  for (size_t i = 0; i < extents.size(); ++i)
    std::cout << extents[i] << " ";
  std::cout << std::endl;
}

// ************************************
// ************ PCDatabase ************
// ************************************

PCDatabaseRegular::PCDatabaseRegular(
    const std::vector< double > &origin,
    const std::vector< std::vector< double > > &extents)
    : PCDatabase(origin.size()), origin_(origin), extents_(extents) {
  assert(extents_.size() == dim_);
  int num_cells = 1;
  for (int i = 0; i < dim_; ++i) {
    num_cells *= extents_[i].size();
  }
  base_cells_.resize(num_cells, PCCell());
}

void PCDatabaseRegular::get_base_cell_coords(int id,
                                             std::vector< double > &origin,
                                             std::vector< double > &extents) {
  origin.resize(dim_);
  extents.resize(dim_);
  std::vector< int > pos = id_to_pos(id);

  for (int n = 0; n < dim_; ++n) {
    origin[n] = origin_[n];
    extents[n] = extents_[n][pos[n]];
    if (pos[n] != 0) {
      origin[n] += extents_[n][pos[n] - 1];
      extents[n] -= extents_[n][pos[n] - 1];
    }
  }
}

std::vector< int > PCDatabaseRegular::id_to_pos(int id) {
  std::vector< int > pos(dim_);
  for (int n = 0; n < dim_; ++n) {
    pos[n] = id % extents_[n].size();
    id -= pos[n];
    id /= extents_[n].size();
  }
  return pos;
}

int PCDatabaseRegular::pos_to_id(std::vector< int > pos) // Horner's method
{
  int id = pos[dim_ - 1];
  for (int n = dim_ - 2; n > -1; --n) {
    id *= extents_[n].size();
    id += pos[n];
  }
  return id;
}

PCDatabaseIrregular::PCDatabaseIrregular(
    const std::vector< std::vector< double > > &origins,
    const std::vector< std::vector< double > > &extents)
    : PCDatabase(origins_[0].size()), origins_(origins), extents_(extents) {
  assert(origins_.size() == extents_.size());
  base_cells_.resize(origins_.size(), PCCell());
}

void PCDatabaseIrregular::get_base_cell_coords(int id,
                                               std::vector< double > &origin,
                                               std::vector< double > &extents) {
  origin = origins_[id];
  extents = extents_[id];
}

// ************************************
// ************** PCMesh **************
// ************************************

PCMesh::PCMesh(const PCDatabasePtr &pcdb,
               const std::vector< unsigned int > &cell_ids,
               const std::vector< int > &base_cell_offsets)
    : pcdb_(pcdb), index_to_id_(cell_ids), offsets_(base_cell_offsets),
      dim_(pcdb_->get_dim()) {}

PCMeshPtr
PCMesh::refine(const std::vector< std::vector< int > > &refinements) const {
  assert(refinements.size() == this->num_cells());
  std::vector< unsigned int > new_index_to_id;
  std::vector< int > new_offsets(offsets_.size(), 0);

  int base_cell_id = 0;
  int curr_cell = 0;
  // prepare attribute stuff
  std::vector< std::string > attribute_names = this->get_cell_attribute_names();
  int num_attributes = attribute_names.size();

  // save type of attribute: 'd' = double, 'i' = int
  std::vector< char > attribute_type(0);
  std::vector< std::vector< int > > int_data(attribute_names.size());
  std::vector< std::vector< double > > double_data(attribute_names.size());
  std::vector< int > is_new(0);
  std::vector< int > parent_index(0);
  LOG_INFO("Refine PCMesh", "Inheriting " << num_attributes << " attributes.");
  for (int i = 0; i < num_attributes; ++i) {
    mesh::AttributePtr curr_attribute =
        this->attributes_.get_attribute(attribute_names[i]);
    mesh::Attribute *attr = curr_attribute.get();
    mesh::IntAttribute *int_attr;
    mesh::DoubleAttribute *double_attr;
    // BoolAttribute* bool_attr;

    if ((int_attr = dynamic_cast< mesh::IntAttribute * >(attr)) != 0) {
      attribute_type.push_back('i');
    } else if ((double_attr = dynamic_cast< mesh::DoubleAttribute * >(attr)) !=
               0) {
      attribute_type.push_back('d');
    } else {
      LOG_ERROR("Attribute type not supported!");
      exit(-1);
    }
  }

  CellIter end = this->end();
  for (CellIter iter = this->begin(); iter != end; ++iter, ++curr_cell) {
    // check if we switched base cell in this iteration
    if (base_cell_id != iter->get_base_cell_id()) // avoid this!!
    {
      // update base cell
      base_cell_id++;
      assert(base_cell_id == iter->get_base_cell_id());
      // start offset of new base cell at end of last one
      new_offsets[base_cell_id] = new_offsets[base_cell_id - 1];
    }

    int starting_ref_dir = 0;
    if (!refinements[curr_cell].empty()) {
      if (refinements[curr_cell][0] < 0) // coarsening case
      {
        // Coarsening will only be done if it is a left child
        // and its neighbour exists and it is also going to be refined.
        // If it is a right child it will be checked if the left
        // child was going to be coarsened. If yes the right
        // cell will be skipped.
        // Otherwise the remaining refinements will be done
        // without coarsening.
        unsigned int cell_id = iter->get_local_cell_id();
        int ref_level = get_ref_level(cell_id);
        if (ref_level > 1) {
          if ((cell_id & (1U << (ref_level - 1))) == 0u) // is left child
          {
            unsigned int neighbour_id = cell_id;
            neighbour_id |= 1U << (ref_level - 1);
            // id to index
            int neighbour_index = this->id_to_index(neighbour_id, base_cell_id);

            if ((neighbour_index == -1) ||
                (refinements[neighbour_index].empty()) ||
                (refinements[neighbour_index][0] >= 0)) {
              starting_ref_dir = 1;
            }
          } else { // is right child
            unsigned int neighbour_id = cell_id;
            neighbour_id &= ~(1U << (ref_level - 1));
            // id to index
            int neighbour_index = this->id_to_index(neighbour_id, base_cell_id);
            if ((neighbour_index == -1) ||
                (refinements[neighbour_index].empty()) ||
                (refinements[neighbour_index][0] >= 0)) {
              starting_ref_dir = 1;
            } else { // if left child was going to be refined skip the
                     // refinements of this cell
              continue;
            }
          }
        } else {
          starting_ref_dir = 1;
        }
      }
    }

    // refine entity with given refinements
    std::vector< unsigned int > new_ids =
        iter->refine_entity(refinements[curr_cell], starting_ref_dir);

    // insert the ids of the newly created cells
    new_index_to_id.insert(new_index_to_id.end(), new_ids.begin(),
                           new_ids.end());

    // the childs inherit the attribute
    for (int i = 0; i < num_attributes; ++i) {
      if (attribute_type[i] == 'i') {
        int value;
        this->get_cell_attribute_value(attribute_names[i], curr_cell, value);
        int_data[i].resize(int_data[i].size() + new_ids.size(), value);
      } else if (attribute_type[i] == 'd') {
        double value;
        this->get_cell_attribute_value(attribute_names[i], curr_cell, value);
        double_data[i].resize(double_data[i].size() + new_ids.size(), value);
      }
    }
    parent_index.resize(parent_index.size() + new_ids.size(), curr_cell);
    if ((new_ids.size() == 1) &&
        (new_ids[0] == iter->get_local_cell_id())) // no refinement case
    {
      is_new.push_back(0);
    } else {
      is_new.resize(is_new.size() + new_ids.size(), 1);
    }
    new_offsets[base_cell_id] += new_ids.size();
  }

  PCMeshPtr refined_mesh(new PCMesh(pcdb_, new_index_to_id, new_offsets));
  for (int i = 0; i < num_attributes; ++i) {
    if (attribute_type[i] == 'i') {
      assert(int_data[i].size() == new_index_to_id.size());
      assert(double_data[i].empty());
      refined_mesh->set_cell_attribute(
          attribute_names[i],
          mesh::AttributePtr(new mesh::IntAttribute(int_data[i])));
      LOG_DEBUG(0, "Setting inherited IntAttribute with name "
                       << attribute_names[i]);
    } else if (attribute_type[i] == 'd') {
      assert(double_data[i].size() == new_index_to_id.size());
      assert(int_data[i].empty());
      refined_mesh->set_cell_attribute(
          attribute_names[i],
          mesh::AttributePtr(new mesh::DoubleAttribute(double_data[i])));
      LOG_DEBUG(0, "Setting inherited DoubleAttribute with name "
                       << attribute_names[i]);
    } else {
      LOG_DEBUG(
          0, "Attribute is neither double nor integer: " << attribute_names[i]);
    }
  }
  refined_mesh->set_cell_attribute(
      "newCell", mesh::AttributePtr(new mesh::IntAttribute(is_new)));
  refined_mesh->set_cell_attribute(
      "parentCell", mesh::AttributePtr(new mesh::IntAttribute(parent_index)));
  return refined_mesh;
}

int PCMesh::get_base_cell_id(int index) const {
  assert(index < this->num_cells() && index >= 0);
  int base_cell_id = 0;
  while (this->offsets_[base_cell_id] <= index) {
    ++base_cell_id;
  }
  return base_cell_id;
}

PCEntity PCMesh::get_cell(int index) const {
  assert(index < this->num_cells() && index >= 0);
  int base_cell_id = this->get_base_cell_id(index);
  return PCEntity(pcdb_, base_cell_id, index_to_id_[index]);
}

int PCMesh::id_to_index(unsigned int cell_id, int base_cell_id) const {
  int start_index = 0;
  int end_index = offsets_[base_cell_id];
  if (base_cell_id > 0) {
    start_index = offsets_[base_cell_id - 1];
  }

  // iterator implementation is much slower :(
  // std::vector<unsigned int>::const_iterator it = index_to_id_.begin() +
  // start_index; std::vector<unsigned int>::const_iterator end_it =
  // index_to_id_.begin() + end_index; for(;it != end_it; ++it, ++start_index)
  //{
  //    if(*it == cell_id)
  //        return start_index;
  //}
  for (; start_index < end_index; ++start_index) {
    if (index_to_id_[start_index] == cell_id) {
      return start_index;
    }
  }
  return -1;
}

CellIter PCMesh::begin() const { return CellIter(this); }

CellIter PCMesh::end() const { return CellIter(this, index_to_id_.size()); }

// template<>
void PCMesh::set_cell_attribute_value(const std::string &name, int index,
                                      int value) {
  if (!attributes_.has_attribute(name)) {
    mesh::AttributePtr new_attr(
        new mesh::IntAttribute(std::vector< int >(this->num_cells(), 0)));
    attributes_.add_attribute(name, new_attr);
    LOG_INFO("PCMesh", "Added new IntAttribute with name " + name);
  }
  attributes_.set(name, index, value);
}

// template<>
void PCMesh::set_cell_attribute_value(const std::string &name, int index,
                                      double value) {
  if (!attributes_.has_attribute(name)) {
    mesh::AttributePtr new_attr(new mesh::DoubleAttribute(
        std::vector< double >(this->num_cells(), 0.)));
    attributes_.add_attribute(name, new_attr);
    LOG_INFO("PCMesh", "Added new DoubleAttribute with name " + name);
  }
  attributes_.set(name, index, value);
}

void PCMesh::get_cell_attribute_value(const std::string &name, int index,
                                      int &value) const {
  attributes_.get(name, index, &value);
}

void PCMesh::get_cell_attribute_value(const std::string &name, int index,
                                      double &value) const {
  attributes_.get(name, index, &value);
}

std::vector< std::string > PCMesh::get_cell_attribute_names() const {
  return attributes_.get_attribute_names();
}

void PCMesh::set_cell_attribute(const std::string &name,
                                const mesh::AttributePtr &attribute) {
  attributes_.add_attribute(name, attribute);
}

mesh::AttributePtr PCMesh::get_cell_attribute(const std::string &name) const {
  return attributes_.get_attribute(name);
}

bool PCMesh::has_attribute(const std::string &name) const {
  return attributes_.has_attribute(name);
}

// ************************************
// ************* PCEntity *************
// ************************************

PCEntity::PCEntity(const PCDatabasePtr &pcdb, int base_cell_id,
                   unsigned int local_cell_id)
    : pcdb_(pcdb), base_cell_id_(base_cell_id), local_cell_id_(local_cell_id) {}

int PCEntity::get_last_ref_dir() const {
  return pcdb_->get_base_cell(base_cell_id_)
      ->get_cell(local_cell_id_)
      ->get_ref_dim();
}

bool PCEntity::has_children() const { return this->get_last_ref_dir() != -1; }

PCEntity PCEntity::get_child(int child_index) const {
  assert(child_index == 0 || child_index == 1);
  assert(this->has_children());
  unsigned int cell_id = this->local_cell_id_;
  int ref_level = get_ref_level(cell_id);
  if (child_index == 0) // left_child
  {
    unsigned int id_left = cell_id;

    id_left &= ~(1U << ref_level);
    id_left |= 1U << (ref_level + 1);
    return PCEntity(pcdb_, base_cell_id_, id_left);
  } // right child
  unsigned int id_right = cell_id;
  id_right |= 1U << (ref_level + 1);
  return PCEntity(pcdb_, base_cell_id_, id_right);
}

bool PCEntity::has_parent() const { return this->local_cell_id_ != 1U; }

PCEntity PCEntity::get_parent() const {
  assert(this->has_parent());
  unsigned int cell_id = this->local_cell_id_;
  int ref_level = get_ref_level(cell_id);
  unsigned int id_parent = cell_id;
  id_parent |= 1U << (ref_level - 1);
  id_parent &= ~(1U << ref_level); // delete the leading 1
  return PCEntity(pcdb_, base_cell_id_, id_parent);
}

void PCEntity::get_coordinates(std::vector< double > &origin,
                               std::vector< double > &extents) const {
  // get base cell coords from database
  pcdb_->get_base_cell_coords(base_cell_id_, origin, extents);
  // calc coords of mesh cell
  pcdb_->get_base_cell(base_cell_id_)
      ->get_cell_coordinates(local_cell_id_, origin, extents);
}

void PCEntity::get_vertices(
    std::vector< std::vector< double > > &vertices) const {
  std::vector< double > origin, extents;
  this->get_coordinates(origin, extents);
  calculate_vertices_from_origin_and_extents(origin, extents, vertices);
}

double PCEntity::get_relative_volume() const {
  unsigned int cell_id = this->local_cell_id_;
  int ref_level = get_ref_level(cell_id);
  return pow(2., -ref_level);
}

// refine the entity in the given directions and return the new cell ids
std::vector< unsigned int >
PCEntity::refine_entity(const std::vector< int > &ref_dims,
                        int ref_index) const {
  if (ref_index >= ref_dims.size()) { // no refinement at all
    return std::vector< unsigned int >(1, local_cell_id_);
  }

  if (ref_dims[ref_index] >= 0) // actual refinement
  {
    // refine the current cell
    pcdb_->get_base_cell(base_cell_id_)
        ->ref_cell(local_cell_id_, ref_dims[ref_index]);
    int cell_id = local_cell_id_;
    // calculate refinement level to calculate new ids
    int ref_level = get_ref_level(cell_id);
    // calculate ids of new cells
    unsigned int new_id_left = cell_id;
    unsigned int new_id_right = cell_id;
    new_id_left &= ~(1U << ref_level);
    new_id_left |= 1U << (ref_level + 1);
    new_id_right |= 1U << (ref_level + 1);
    // call refinement functions in children with next refinement direction
    ++ref_index;
    local_cell_id_ = new_id_left; // change cell to left cell
    std::vector< unsigned int > left_ids =
        this->refine_entity(ref_dims, ref_index);
    local_cell_id_ = new_id_right; // change cell to right cell
    std::vector< unsigned int > right_ids =
        this->refine_entity(ref_dims, ref_index);
    local_cell_id_ = cell_id; // change cell back to original cell
    left_ids.insert(left_ids.end(), right_ids.begin(), right_ids.end());
    return left_ids; // return combined left and right ids
  }
  // in case of coarsening it is assumed that the mesh checked
  // that the coarsening doesn't lead to problems
  unsigned int cell_id = local_cell_id_;
  int ref_level = get_ref_level(cell_id);
  assert(!(cell_id &
           (1U << (ref_level - 1)))); // only left childs are allowed to coarsen

  // calc parent id
  unsigned int new_id_parent = cell_id;
  new_id_parent |= 1U << (ref_level - 1);
  new_id_parent &= ~(1U << ref_level); // delete the leading 1
  // call refinement function in parent with next refinement direction
  ++ref_index;
  local_cell_id_ = new_id_parent;

  std::vector< unsigned int > new_ids =
      this->refine_entity(ref_dims, ref_index);
  local_cell_id_ = cell_id;
  return new_ids;
}

// this creates a mesh with the base cells as mesh cells
PCMeshPtr create_mesh_from_database(const PCDatabasePtr &wdb) {
  int num_cells = wdb->get_num_base_cells();
  std::vector< unsigned int > index_to_id(num_cells,
                                          1U); // 1 is id of base cells
  std::vector< int > offsets(num_cells);
  int counter = 1;
  for (std::vector< int >::iterator iter = offsets.begin();
       iter != offsets.end(); ++iter, ++counter) {
    *iter = counter;
  }

  PCMeshPtr mesh(new PCMesh(wdb, index_to_id, offsets));
  return mesh;
}

void calculate_vertices_from_origin_and_extents(
    std::vector< double > &origin, std::vector< double > &extents,
    std::vector< std::vector< double > > &vertices) {
  int dimension = origin.size();
  assert(origin.size() == extents.size());
  int num_vertices = (int)pow(2., (double)dimension);
  vertices.resize(num_vertices, std::vector< double >(dimension, 0.));
  for (int i = 0; i < num_vertices; ++i) {
    vertices[i] = origin;
    int index_mod = i;
    for (int j = 0; j < dimension; ++j) {
      if (index_mod >= (int)pow(2., (double)dimension - 1 - j))
        vertices[i][dimension - 1 - j] += extents[dimension - 1 - j];
      index_mod = index_mod % (int)pow(2., (double)dimension - 1 - j);
    }
  }
}

void mesh_ucd_writer(const PCMeshPtr &pc_mesh, const std::string &filename) {

  int dim = pc_mesh->get_dim();
  if (dim > 3) {
    LOG_ERROR("Can't write stochastic mesh with dimension > 3! Exiting!");
    exit(-1);
  }
  const int num_cells = pc_mesh->num_cells();
  int num_vertices_per_cell = (int)pow(2., (double)dim);
  if (dim == 1) {
    // in 1D we still do 2D visualization for better viewability
    num_vertices_per_cell *= 2;
  }
  const int num_vertices = num_cells * num_vertices_per_cell;

  // std::cout << "num_vertices_per_cell: " << num_vertices_per_cell << "
  // num_cells: " << num_cells << " dim: " << dim << std::endl; std::cout <<
  // "Allocating " << num_vertices * 3 * 8 << " byte!" << std::endl;

  std::vector< std::vector< double > > vertices(num_vertices,
                                                std::vector< double >(3, 0.));

  // get length of mesh in 1d case to calculate a good width
  double mesh_length = 0.;
  if (dim == 1) {
    for (CellIter iter = pc_mesh->begin(); iter != pc_mesh->end(); ++iter) {
      std::vector< double > cell_origin;
      std::vector< double > cell_extents;
      iter->get_coordinates(cell_origin, cell_extents);
      mesh_length += cell_extents[0];
    }
  }
  double desired_ratio = 0.15;

  // collect data
  int curr_cell_index = 0;
  int curr_vertex_index = 0;
  for (CellIter iter = pc_mesh->begin(); iter != pc_mesh->end();
       ++iter, ++curr_cell_index) {
    // fill vertices
    curr_vertex_index = curr_cell_index * num_vertices_per_cell;

    // "naive", hardcoded version
    /*
    std::vector<double> cell_origin;
    std::vector<double> cell_extents;
    iter->get_coordinates(cell_origin, cell_extents);
    cell_origin.resize(3,0.);
    cell_extents.resize(3,0.);

    vertices[curr_vertex_index] = cell_origin;
    vertices[curr_vertex_index + 1] = cell_origin;
    vertices[curr_vertex_index + 1][0] += cell_extents[0];
    if(dim > 1)
    {
        vertices[curr_vertex_index + 2] = cell_origin;
        vertices[curr_vertex_index + 2][1] += cell_extents[1];
        vertices[curr_vertex_index + 3] = cell_origin;
        vertices[curr_vertex_index + 3][0] += cell_extents[0];
        vertices[curr_vertex_index + 3][1] += cell_extents[1];
        if(dim > 2)
        {
            vertices[curr_vertex_index + 4] = cell_origin;
            vertices[curr_vertex_index + 4][2] += cell_extents[2];
            vertices[curr_vertex_index + 5] = cell_origin;
            vertices[curr_vertex_index + 5][0] += cell_extents[0];
            vertices[curr_vertex_index + 5][2] += cell_extents[2];
            vertices[curr_vertex_index + 6] = cell_origin;
            vertices[curr_vertex_index + 6][1] += cell_extents[1];
            vertices[curr_vertex_index + 6][2] += cell_extents[2];
            vertices[curr_vertex_index + 7] = cell_origin;
            vertices[curr_vertex_index + 7][0] += cell_extents[0];
            vertices[curr_vertex_index + 7][1] += cell_extents[1];
            vertices[curr_vertex_index + 7][2] += cell_extents[2];
        }
    }*/

    // alternative, "nice" version
    if (dim != 1) {
      std::vector< std::vector< double > > vertices_on_cell(
          num_vertices_per_cell, std::vector< double >(dim, 0.));
      iter->get_vertices(vertices_on_cell);
      for (int i = 0; i < num_vertices_per_cell; ++i) {
        vertices[curr_vertex_index + i] = vertices_on_cell[i];
        vertices[curr_vertex_index + i].resize(
            3, 0.); // needs 3rd dimension for ucd format
      }
    } else {
      std::vector< std::vector< double > > vertices_on_cell(
          2, std::vector< double >(dim, 0.));
      iter->get_vertices(vertices_on_cell);
      for (int i = 0; i < 2; ++i) {
        vertices[curr_vertex_index + i] = vertices_on_cell[i];
        vertices[curr_vertex_index + i].resize(
            3, 0.); // needs 3rd dimension for ucd format
        vertices[curr_vertex_index + i + 2] = vertices_on_cell[i];
        vertices[curr_vertex_index + i + 2].resize(3, 0.);
        vertices[curr_vertex_index + i + 2][1] = mesh_length * desired_ratio;
      }
    }
  }
  // write data
  std::ofstream file(filename.c_str());

  // info about cell data
  std::vector< std::string > attribute_names =
      pc_mesh->get_cell_attribute_names();
  int ncell_data = attribute_names.size();
  // info about node data
  int nnode_data = 0;
  // header
  file << num_vertices << " " << num_cells << " " << nnode_data << " "
       << ncell_data << " " << 0 << std::endl;
  // write points
  for (int i = 0; i < num_vertices; ++i) {
    file << i << " "
         << string_from_range(vertices[i].begin(), vertices[i].end())
         << std::endl;
  }
  int material_id = 0;
  std::string cell_type;
  if (dim == 1) {
    cell_type = "quad";
  } else if (dim == 2) {
    cell_type = "quad";
  } else if (dim == 3) {
    cell_type = "hex";
  } else {
    cell_type = "dim too high";
  }

  // write cells
  int permutation[] = {0, 1, 3, 2, 4, 5, 7, 6};
  for (int i = 0; i < num_cells; ++i) {
    curr_vertex_index = num_vertices_per_cell * i;
    file << i << " " << material_id << " " << cell_type;
    for (int j = 0; j < num_vertices_per_cell; ++j) {
      file << " " << curr_vertex_index + permutation[j];
    }
    file << std::endl;
  }
  // write cell data
  if (ncell_data > 0) {
    file << ncell_data;
    for (int i = 0; i < ncell_data; ++i) {
      file << " 1";
    }
    file << std::endl;
    for (int i = 0; i < ncell_data; ++i) {
      file << attribute_names[i] << ", empty" << std::endl;
    }
    for (int i = 0; i < num_cells; ++i) {
      file << i;
      for (int j = 0; j < ncell_data; ++j) {
        if (typeid(*(pc_mesh->get_cell_attribute(attribute_names[j]))) ==
            typeid(mesh::DoubleAttribute)) {
          double value;
          pc_mesh->get_cell_attribute_value(attribute_names[j], i, value);
          file << " " << value;
        } else if (typeid(*(pc_mesh->get_cell_attribute(attribute_names[j]))) ==
                   typeid(mesh::IntAttribute)) {
          int value;
          pc_mesh->get_cell_attribute_value(attribute_names[j], i, value);
          file << " " << value;
        }
      }
      file << std::endl;
    }
  }
}

} // namespace polynomialchaos
} // namespace hiflow
