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

/** \file
 * Input/Output in XML format -- non-generic code.
 */

#include "common/property_tree.h"
#include <cctype>
#include <iostream>
#include <stack>

/// @author Jonas Latt, Jonas Fietz, Mathias Krause, Tobias Hahn, Jonas Kratzke,
/// Simon Gawlok

namespace hiflow {

PropertyTree PropertyTree::notFound;

PropertyTree::PropertyTree(tinyxml2::XMLNode *pParent, int master_rank,
                           const MPI_Comm &comm) {
  int my_rank = -1;
  MPI_Comm_rank(comm, &my_rank);
  if (my_rank == master_rank) {
    mainProcessorIni(pParent, master_rank, comm);
  } else {
    slaveProcessorIni(master_rank, comm);
  }
}

PropertyTree::PropertyTree(const std::string &fName, int master_rank,
                           const MPI_Comm &comm) {
  tinyxml2::XMLDocument *doc = 0;
  int my_rank = -1;
  MPI_Comm_rank(comm, &my_rank);
  LOG_INFO("Read XML", fName);
  
  if (my_rank == master_rank) {
    doc = new tinyxml2::XMLDocument();
    tinyxml2::XMLError loadOK = doc->LoadFile(fName.c_str());
    if (loadOK != tinyxml2::XML_SUCCESS) {
      LOG_ERROR("Problem processing input XML file " << fName);
      quit_program();
    }
  }

  if (my_rank == master_rank) {
    mainProcessorIni(doc, master_rank, comm);
    delete doc;
  } else {
    slaveProcessorIni(master_rank, comm);
  }
}

void PropertyTree::mainProcessorIni(tinyxml2::XMLNode *pParent, int master_rank,
                                    const MPI_Comm &comm) {
  assert(pParent->ToDocument() != NULL || pParent->ToElement() != NULL);

  if (pParent->ToDocument() != NULL) {
    // ignore the surrounding PARAM-block
    pParent = pParent->FirstChild();
  }

  name_ = pParent->Value();
  bCastString(&name_, master_rank, comm);

  tinyxml2::XMLNode *pChild;

  // -2: Finished, -1: XMLUnknown, 0: XMLElement, 1: XMLText
  int type = -1;
  for (pChild = pParent->FirstChild(); pChild != 0;
       pChild = pChild->NextSibling()) {
    if (pChild->ToElement() != NULL) {
      type = 0;
    } else if (pChild->ToText() != NULL) {
      type = 1;
    } else {
      type = -1;
    }

    MPI_Bcast(static_cast< void * >(&type), 1, MPI_INT, master_rank, comm);
    if (type == 0) {
      PropertyTree *new_child = new PropertyTree(pChild, master_rank, comm);
      children_[new_child->getName()] = new_child;
    } else if (type == 1) {
      text_ = pChild->ToText()->Value();
      bCastString(&text_, master_rank, comm);
    }
  }
  type = -2;
  MPI_Bcast(static_cast< void * >(&type), 1, MPI_INT, master_rank, comm);
}

void PropertyTree::slaveProcessorIni(int master_rank, const MPI_Comm &comm) {
  bCastString(&name_, master_rank, comm);

  int type = -1;
  do {
    MPI_Bcast(static_cast< void * >(&type), 1, MPI_INT, master_rank, comm);
    if (type == 0) {
      PropertyTree *new_child = new PropertyTree(0, master_rank, comm);
      children_[new_child->getName()] = new_child;
    } else if (type == 1) {
      bCastString(&text_, master_rank, comm);
    }
  } while (type != -2);
}

void PropertyTree::bCastString(std::string *sendBuf, int master_rank,
                               const MPI_Comm &comm) {
  int length = static_cast< int >(sendBuf->size());
  MPI_Bcast(static_cast< void * >(&length), 1, MPI_INT, master_rank, comm);
  char *buffer = new char[length + 1];
  int rank = -1;
  MPI_Comm_rank(comm, &rank);
  if (rank == master_rank) {
    std::copy(sendBuf->c_str(), sendBuf->c_str() + length + 1, buffer);
  }
  MPI_Bcast(static_cast< void * >(buffer), length + 1, MPI_CHAR, master_rank,
            comm);
  if (rank != master_rank) {
    *sendBuf = buffer;
  }
  delete[] buffer;
}

PropertyTree::PropertyTree() { name_ = "XML node not found"; }

PropertyTree::PropertyTree(const PropertyTree &srcTree)
    : name_(srcTree.name_), text_(srcTree.text_) {
  if (!srcTree.isEmpty()) {
    PropertyTree::const_iterator src_end = srcTree.end_children();
    for (PropertyTree::const_iterator it = srcTree.begin_children();
         it != src_end; ++it) {
      add(*(it->second));
    }
  }
}

PropertyTree::PropertyTree(const std::string &name) : name_(name) {}

PropertyTree::~PropertyTree() {
  if (!isEmpty()) {
    PropertyTree::iterator end_tree = this->end_children();
    for (PropertyTree::iterator it = this->begin_children(); it != end_tree;
         ++it) {
      delete it->second;
    }
  }
}

void PropertyTree::clear_tree() {
  if (!isEmpty()) {
    PropertyTree::iterator end_tree = this->end_children();
    for (PropertyTree::iterator it = this->begin_children(); it != end_tree;
         ++it) {
      delete it->second;
    }
    children_.clear();
  }
}

PropertyTree &PropertyTree::operator=(const PropertyTree &srcTree) {
  if (this != &srcTree) {
    if (!isEmpty()) {
      PropertyTree::iterator end_tree = this->end_children();
      for (PropertyTree::iterator it = this->begin_children(); it != end_tree;
           ++it) {
        remove(it->first);
      }
    }
    name_ = srcTree.getName();
    text_ = srcTree.getText();

    if (!srcTree.isEmpty()) {
      PropertyTree::const_iterator src_end = srcTree.end_children();
      for (PropertyTree::const_iterator it = srcTree.begin_children();
           it != src_end; ++it) {
        add(*(it->second));
      }
    }
  }

  return *this;
}

PropertyTree const &PropertyTree::operator[](const std::string &name) const {
  std::map< std::string, PropertyTree * >::const_iterator element =
      children_.find(name);
  if (element != children_.end()) {
    return *(element->second);
  }
  LOG_ERROR("Child " << name << " in tree " << this->name_ << " not found!");
  return notFound;
}

PropertyTree *PropertyTree::get_child(const std::string &name) {
  std::map< std::string, PropertyTree * >::const_iterator element =
      children_.find(name);
  if (element != children_.end()) {
    return element->second;
  }
  LOG_ERROR("Child " << name << " in tree " << this->name_ << " not found!");
  return &notFound;
}

std::map< std::string, PropertyTree * >::const_iterator
PropertyTree::begin_children() const {
  return children_.begin();
}

std::map< std::string, PropertyTree * >::iterator
PropertyTree::begin_children() {
  return children_.begin();
}

std::map< std::string, PropertyTree * >::const_iterator
PropertyTree::end_children() const {
  return children_.end();
}

std::map< std::string, PropertyTree * >::iterator PropertyTree::end_children() {
  return children_.end();
}

std::string PropertyTree::getName() const { return name_; }

std::string PropertyTree::getText() const { return text_; }

void PropertyTree::add(const PropertyTree &tree) {
  PropertyTree::iterator has_element = children_.find(tree.getName());
  if (has_element == children_.end()) {
    PropertyTree *newTree = new PropertyTree(tree);
    children_[tree.getName()] = newTree;
  } else {
    has_element->second->merge(tree);
  }
}

void PropertyTree::add(const std::string &key) {
  PropertyTree::iterator has_element = children_.find(key);
  if (has_element == children_.end()) {
    PropertyTree *newTree = new PropertyTree(key);
    children_[key] = newTree;
  }
}

bool PropertyTree::remove(const std::string &key) {
  std::map< std::string, PropertyTree * >::iterator element =
      children_.find(key);
  if (element != children_.end()) {
    delete element->second;
    children_.erase(element);
    return true;
  }
  return false;
}

void PropertyTree::merge(const PropertyTree &tree) {
  if (name_ == tree.getName()) {
    // text_=tree.get<std::string>();
    text_ = tree.getText();
  }
  if (!tree.isEmpty()) {
    for (PropertyTree::const_iterator it = tree.begin_children(),
                                      end = tree.end_children();
         it != end; ++it) {
      add(*(it->second));
    }
  }
}

bool PropertyTree::contains(const std::string &key) const {
  std::map< std::string, PropertyTree * >::const_iterator element =
      children_.find(key);
  return !(element == children_.end());
}

bool PropertyTree::isEmpty() const {
  switch (children_.size()) {
  case 0:
    return true;
  default:
    return false;
  }
}

int PropertyTree::size() const { return static_cast< int >(children_.size()); }

std::ostream &operator<<(std::ostream &os, const PropertyTree &tree) {
  os << "<" << tree.getName() << ">" << std::endl;
  if (!tree.getText().empty()) {
    os << tree.getText() << std::endl;
  }
  if (!tree.isEmpty()) {
    PropertyTree::const_iterator end_tree = tree.end_children();
    for (PropertyTree::const_iterator it = tree.begin_children();
         it != end_tree; ++it) {
      os << *(it->second);
    }
  }
  os << "</" << tree.getName() << ">" << std::endl;
  return os;
}

} // namespace hiflow
