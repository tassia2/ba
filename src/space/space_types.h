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

#ifndef __SPACE_TYPES_H_
#define __SPACE_TYPES_H_
#include "common/pointers.h"
#include "common/vector_algebra_descriptor.h"

/// \author Philipp Gerstner

namespace hiflow {
  template <class DataType, int DIM> class VectorSpace;

  template <class DataType, int DIM> 
  using VectorSpaceSPtr = SharedPtr< VectorSpace<DataType, DIM> >;  

  template <class DataType, int DIM> 
  using CVectorSpaceSPtr = SharedPtr< const VectorSpace<DataType, DIM> >;  

  template <class DataType, int DIM> 
  using VectorSpaceUPtr = UniquePtr< VectorSpace<DataType, DIM> >; 
  
} // namespace hiflow

#endif
