#ifndef HIFLOW_PC_MESH_H_
#define HIFLOW_PC_MESH_H_

#include <boost/iterator/iterator_facade.hpp>
#include <cassert>
#include <fstream>
#include <iostream>
#include <math.h>
#include <vector>

#include "common/log.h"
#include "common/macros.h"
#include "common/pointers.h"
#include "mesh/attributes.h"

namespace hiflow {
namespace polynomialchaos {

int get_ref_level(unsigned int index);
void calculate_vertices_from_origin_and_extents(
    std::vector< double > &origin, std::vector< double > &extents,
    std::vector< std::vector< double > > &vertices);
// Cell class for the PCMesh. For the indexing the assumption is made that
// 1U = 0...01 in the actual memory. Where 0...0 are an arbitrary number
// n of zeros. The maximum number of cells is then given by 2^(n-1)
class PCCell {
public:
  PCCell();

  void set_ref_type(int type);

  void clean_deletion();

  int get_ref_dim() const { return ref_dim_; }

  // get pointer to subcell
  const PCCell *get_cell(unsigned int index) const;

  // refine subcell
  void ref_cell(unsigned int index, int ref_dim);

  // get coordinates of subcell (needs origin and extents of base cell as input)
  void get_cell_coordinates(unsigned int index, std::vector< double > &origin,
                            std::vector< double > &extents) const;

  // print index, ref type and coords of subcell (needs origin and extents of
  // base cell as input)
  void print_cell(unsigned int index, std::vector< double > origin,
                  std::vector< double > extents) const;

  PCCell *get_left_child() { return left_child_; };
  PCCell *get_right_child() { return right_child_; };

private:
  int ref_dim_; // range from -1 to dim-1
  PCCell *left_child_;
  PCCell *right_child_;
};

class PCDatabase {
public:
  inline PCDatabase(int dim) : dim_(dim){};

  inline virtual ~PCDatabase() {
    int num_cells = this->get_num_base_cells();
    for (int i = 0; i < num_cells; ++i) {
      base_cells_[i].clean_deletion();
    }
  }

  virtual void get_base_cell_coords(int id, std::vector< double > &origin,
                                    std::vector< double > &extents) = 0;

  void get_base_cell_vertices(int id,
                              std::vector< std::vector< double > > &vertices) {
    std::vector< double > origin, extents;
    this->get_base_cell_coords(id, origin, extents);
    calculate_vertices_from_origin_and_extents(origin, extents, vertices);
  }

  inline PCCell *get_base_cell(int id) { return &base_cells_[id]; }

  inline int get_num_base_cells() { return base_cells_.size(); }

  inline int get_dim() { return dim_; }

protected:
  std::vector< PCCell > base_cells_; // id to base cell
  int dim_;
};

typedef BSharedPtr< PCDatabase >::Type PCDatabasePtr;

class PCDatabaseRegular : public PCDatabase {
public:
  PCDatabaseRegular(const std::vector< double >& origin,
                    const std::vector< std::vector< double > >& extents);

  void get_base_cell_coords(int id, std::vector< double > &origin,
                            std::vector< double > &extents);

private:
  // convert id to position vector
  std::vector< int > id_to_pos(int id);
  // inverse of above function
  int pos_to_id(std::vector< int > pos);

  std::vector< double > origin_;
  std::vector< std::vector< double > > extents_;
  // the origin of base_cell_ (a_1, a_2, a_3, ..., a_n) is given by
  // origin[i] = origin_[i] + extents_[i][a_i]
  // its extents are extents[i] = extents_[i][a_i + 1] - extents_[i][a_i]
  // its index is index = \sum_{i=1}^{n} a_i * \prod_{j=1}^{i-1}
  // extents_[j].size()
};

class PCDatabaseIrregular : public PCDatabase {
public:
  PCDatabaseIrregular(const std::vector< std::vector< double > >& origin,
                      const std::vector< std::vector< double > >& extents);

  void get_base_cell_coords(int id, std::vector< double > &origin,
                            std::vector< double > &extents);

private:
  // origin of each cell
  std::vector< std::vector< double > > origins_;
  // extents of each cell
  std::vector< std::vector< double > > extents_;
};

class PCEntity {
public:
  PCEntity() : pcdb_(), base_cell_id_(-1), local_cell_id_(0U) {}

  PCEntity(const PCDatabasePtr& pcdb, int base_cell_id, unsigned int local_cell_id);

  void set_base_cell_id(int base_cell_id) { base_cell_id_ = base_cell_id; }

  int get_base_cell_id() const { return base_cell_id_; }

  void set_local_cell_id(unsigned int local_cell_id) {
    local_cell_id_ = local_cell_id;
  }

  unsigned int get_local_cell_id() const { return local_cell_id_; }

  void get_coordinates(std::vector< double > &origin,
                       std::vector< double > &extents) const;

  void get_vertices(std::vector< std::vector< double > > &vertices) const;

  double get_relative_volume() const;

  int get_last_ref_dir() const;

  // refine the entity in the given directions and return the new cell ids
  std::vector< unsigned int > refine_entity(const std::vector< int > &ref_dims,
                                            int ref_index = 0) const;

  bool has_children() const;
  PCEntity get_child(int child_index) const;
  bool has_parent() const;

  PCEntity get_parent() const;

private:
  PCDatabasePtr pcdb_;
  int base_cell_id_; // base cell id (in pcdb_) of the base cell the entity is
                     // on
  mutable unsigned int local_cell_id_; // local id of the cell in the base cell
};

class PCMesh;
typedef BSharedPtr< PCMesh >::Type PCMeshPtr;
typedef BSharedPtr< const PCMesh >::Type ConstPCMeshPtr;
class CellIter;
class PCMesh {
public:
  PCMesh(const PCDatabasePtr& pcdb, const std::vector< unsigned int >& cell_ids,
         const std::vector< int >& base_cell_offsets);

  // Refines based on a given refinement vector
  PCMeshPtr refine(const std::vector< std::vector< int > > &refinements) const;

  int get_dim() const { return dim_; }

  int num_cells() const { return index_to_id_.size(); }

  // get entity with a given index
  PCEntity get_cell(int index) const;

  // get base_cell_id of a cell with a given index
  int get_base_cell_id(int index) const;

  PCDatabasePtr get_database() const { return pcdb_; };

  // Returns cell index for a given cell_id on the base cell given
  // by the base_cell_id. Return -1 if cell is not in mesh.
  int id_to_index(unsigned int cell_id, int base_cell_id) const;

  const std::vector< unsigned int > &get_index_to_id_map() const {
    return index_to_id_;
  };

  const std::vector< int > &get_offsets() const { return offsets_; };

  // template<typename T>
  // void set_cell_attribute_value(const std::string& name, int index, T value);
  void set_cell_attribute_value(const std::string &name, int index, int value);
  void set_cell_attribute_value(const std::string &name, int index,
                                double value);

  // template<typename T>
  // void get_cell_attribute_value(const std::string& name, int index, T& value)
  // const;
  void get_cell_attribute_value(const std::string &name, int index,
                                int &value) const;
  void get_cell_attribute_value(const std::string &name, int index,
                                double &value) const;

  void set_cell_attribute(const std::string &name,
                          const mesh::AttributePtr& attribute);

  mesh::AttributePtr get_cell_attribute(const std::string &name) const;

  bool has_attribute(const std::string &name) const;

  std::vector< std::string > get_cell_attribute_names() const;

  CellIter begin() const;

  CellIter end() const;

private:
  PCDatabasePtr pcdb_;
  mesh::AttributeTable attributes_;
  std::vector< unsigned int > index_to_id_;
  std::vector< int > offsets_;
  int dim_;
};

class CellIter : public boost::iterator_facade< CellIter, const PCEntity,
                                                boost::forward_traversal_tag > {
public:
  CellIter(const PCMesh *pc_mesh) // returns an Iterator to first mesh cell
      : pc_entity_(pc_mesh->get_database(), 0, 1U), pc_mesh_(pc_mesh),
        curr_index_(0) {}

  CellIter(const PCMesh *pc_mesh, int start_index)
      : pc_entity_(pc_mesh->get_database(), 0, 0U), pc_mesh_(pc_mesh),
        curr_index_(start_index) {}

  CellIter &operator=(const CellIter &other) {
    this->curr_index_ = other.curr_index_;
    this->pc_mesh_ = other.pc_mesh_;
    return *this;
  }

private:
  // provide access for boost::iterator_facade
  friend class boost::iterator_core_access;

  // interface for boost::iterator_facade

  bool equal(const CellIter &other) const {
    return this->curr_index_ == other.curr_index_;
  }

  void increment() { ++curr_index_; }

  const PCEntity &dereference() const {
    pc_entity_.set_base_cell_id(pc_mesh_->get_base_cell_id(curr_index_));
    pc_entity_.set_local_cell_id(pc_mesh_->get_index_to_id_map()[curr_index_]);
    return pc_entity_;
  }

  mutable PCEntity pc_entity_;

  const PCMesh *pc_mesh_;

  // data about current entity
  int curr_index_;
};

// this creates a mesh with the base cells as mesh cells
PCMeshPtr create_mesh_from_database(const PCDatabasePtr& pcdb);

// write out a 1d, 2d or 3d PCMesh with all its attributes (cell data)
void mesh_ucd_writer(const PCMeshPtr& pc_mesh, const std::string& filename);

} // namespace polynomialchaos
} // namespace hiflow

#endif
