// Copyright (C) 2011-2021 Vincent Heuveline
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

#ifndef _PROPERTY_TREE_H_
#define _PROPERTY_TREE_H_

#include "common/log.h"
#include "common/macros.h"
#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <mpi.h>
#include <sstream>
#include <string>
#include <tinyxml2.h>
#include <vector>

namespace hiflow {

/// @brief Describes a tree of properties, i.e a hierarchical
/// collection of key-value pairs. The keys are of type
/// std::string, and the values of the template parameter type
/// ValueType. ValueType must be default-constructible.
/// @author Jonas Latt, Jonas Fietz, Mathias Krause, Tobias Hahn, Jonas Kratzke,
/// Simon Gawlok

class PropertyTree {
public:
  typedef std::map< std::string, PropertyTree * >::const_iterator
      const_iterator;
  typedef std::map< std::string, PropertyTree * >::iterator iterator;

  /// Constructs a new PropertyTree from an XML file
  PropertyTree(const std::string &fName, int master_rank, const MPI_Comm &comm);

  /// Copy constructor
  PropertyTree(const PropertyTree &srcTree);

  /// Constructs a new plane PropertyTree
  PropertyTree(const std::string &name);

  PropertyTree();

  /// Destructor
  ~PropertyTree();

  /// Remove all subtrees
  void clear_tree();

  /// Assignment
  PropertyTree &operator=(const PropertyTree &srcTree);

  /// Return the subtree with given name as const
  PropertyTree const &operator[](const std::string &name) const;

  /// Return the subtree with given name for extensions
  PropertyTree *get_child(const std::string &name);

  /// Standard read function of node value.
  template < typename T > bool read(T &value) const;
  template < typename T > bool read(std::vector< T > &values) const;

  /// Direct retrieval of node value.
  template < typename T > T get() const;
  /// Direct retrieval of node value with the possibility to set an default
  /// value if the value was not found.
  template < typename T > T get(T def) const;

  /// Add a child node.
  void add(const std::string &key);

  /// Add a child node and directly set a value.
  template < typename T > void add(const std::string &key, const T &value);

  /// Merges the tree with another, preferring the other values in case of same
  /// keys.
  void merge(const PropertyTree &tree);

  /// Returns true if the tree contains child with specified key
  bool contains(const std::string &key) const;

  /// Returns true if this tree is empty, i.e. the tree is a leaf.
  bool isEmpty() const;

  /// Returns the number of keys in this level.
  int size() const;

  /// Return the name of the element
  std::string getName() const;
  /// Return the text of the element
  std::string getText() const;

private:
  /// Construct a XML node.
  PropertyTree(tinyxml2::XMLNode *pParent, int master_rank,
               const MPI_Comm &comm);

  /// Construct a new tree node with given name and value.
  PropertyTree(const std::string &name, const std::string &text,
               int master_rank, const MPI_Comm &comm);

  /// Write the tree to a stream in xml format
  friend std::ostream &operator<<(std::ostream &os, const PropertyTree &tree);

  /// Set value of node.
  template < typename T > void set(const T &value);

  /// Add a subtree
  void add(const PropertyTree &tree);

  /// Remove a subtree.
  bool remove(const std::string &key);

  void mainProcessorIni(tinyxml2::XMLNode *pParent, int master_rank,
                        const MPI_Comm &comm);
  void slaveProcessorIni(int master_rank, const MPI_Comm &comm);
  void bCastString(std::string *sendbuf, int master_rank, const MPI_Comm &comm);

  /// Return an iterator for this level at the tree
  const_iterator begin_children() const;
  iterator begin_children();

  /// Return an iterator end element
  const_iterator end_children() const;
  iterator end_children();

  std::string name_;
  std::string text_;
  std::map< std::string, PropertyTree * > children_;
  static PropertyTree notFound;
};

template < typename T > bool PropertyTree::read(T &value) const {
  std::stringstream valueStr(text_);
  T tmp = T();
  if (!(valueStr >> tmp)) {
    if (this->isEmpty()) // Only log errors for leafs of the tree
      LOG_ERROR("Cannot read value from property element " << name_);
    return false;
  }
  value = tmp;
  return true;
}

template <> inline bool PropertyTree::read(bool &value) const {
  std::stringstream valueStr(text_);
  std::string word;
  valueStr >> word;
  // Transform to lower-case, so that "true" and "false" are case-insensitive.
  std::transform(word.begin(), word.end(), word.begin(), ::tolower);
  if (word == "true") {
    value = true;
    return true;
  }
  if (word == "false") {
    value = false;
    return true;
  }
  LOG_ERROR("Cannot read boolean value from XML element " << name_);

  return false;
}

template <> inline bool PropertyTree::read(std::string &value) const {
  if (name_ == "XML node not found") {
    return false;
  }
  value = text_;
  return true;
}

template < typename T >
inline bool PropertyTree::read(std::vector< T > &values) const {
  std::stringstream multiValueStr(text_);
  std::string word;
  std::vector< T > tmp(values);
  while (multiValueStr >> word) {
    std::stringstream valueStr(word);
    T value;
    if (!(valueStr >> value)) {
      if (this->isEmpty()) // Only log errors for leafs of the tree
        LOG_ERROR("Cannot read value array from property element. " << name_);
      return false;
    }
    tmp.push_back(value);
  }
  values.swap(tmp);
  return true;
}

template < typename T > inline T PropertyTree::get() const {
  std::stringstream valueStr(text_);
  T tmp = T();
  if (!(valueStr >> tmp)) {
    LOG_ERROR("Cannot read value from property element. " << name_ << ".");
    quit_program();
  }
  return tmp;
}

template < typename T > inline T PropertyTree::get(T def) const {
  std::stringstream valueStr(text_);
  T tmp = T();
  if (!(valueStr >> tmp)) {
    LOG_INFO("Property tree element not found. Using default", def);
    return def;
  }
  return tmp;
}

template < typename T >
inline void PropertyTree::add(const std::string &key, const T &value) {
  this->add(key);
  children_[key]->set(value);
}

template < typename T > inline void PropertyTree::set(const T &value) {
  std::stringstream valueStr;
  valueStr << value;
  text_ = valueStr.str();
}

} // namespace hiflow

#endif // _PROPERTY_TREE_H_
