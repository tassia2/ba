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

#ifndef HIFLOW_DATA_TOOLS_H
#define HIFLOW_DATA_TOOLS_H

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <iomanip> 
#include <vector>
#include <math.h> 
#include <memory>

/// @brief T

/// @author Philipp Gerstner

namespace {

} // namespace

namespace hiflow {

template < class T >
inline void Axpy(std::vector< T > &out, 
                 const std::vector< T > &in,
                 const T alpha) 
{
  assert (out.size() == in.size());
  for (int i=0, e_i = in.size(); i != e_i; ++i)
  {
    out[i] += alpha * in[i];
  }
}

template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

template <class DataType, size_t AlignSize>
DataType* allocate_aligned (size_t nb_elements, DataType* & delete_ptr) 
{
  const size_t overhead = AlignSize / sizeof(DataType) + 1;
  delete_ptr = new DataType[nb_elements+overhead];
  void* tmp_ptr = delete_ptr;
  
  size_t space = nb_elements * sizeof(DataType) + AlignSize;
  void* tmp3 = std::align(AlignSize, nb_elements*sizeof(DataType), tmp_ptr, space);

  //std::cout  << space << "  " << tmp << " " << tmp_ptr << " " << tmp3 << " length " <<  N*N * sizeof(DataType) << std::endl;
  assert (tmp3 != nullptr);
               
  return (DataType*) tmp3;
}

template <typename Enumeration>
auto as_integer(Enumeration const value)
    -> typename std::underlying_type<Enumeration>::type
{
    return static_cast<typename std::underlying_type<Enumeration>::type>(value);
}

template<class DataType>
void number_range (DataType first, DataType inc, size_t length, std::vector<DataType>& result)
{
  result.resize(length, static_cast<DataType>(0));
  for (size_t l=0; l<length; ++l)
  {
    result[l] = first + static_cast<DataType>(l) * inc;
  }
}

/// @brief Function to get the corresponding permutation for sorting of
/// values vector. Permutation vector is allocated inside the function
/// @param[in] values Vector of values to get the sorting permutation for
/// @param[out] v Permutation vector

template < class T, class IndexType = int >
void compute_sort_permutation_stable(const std::vector< T > &values, 
                                     std::vector< IndexType > &v) 
{
  const size_t size = values.size();
  v.clear();
  v.resize(size);
  for (size_t i = 0; i != size; ++i) 
  {
    v[i] = static_cast< IndexType >(i);
  }

  auto comp = [values] (IndexType a, IndexType b) { return values[a] < values[b]; };

  std::stable_sort(v.begin(), v.end(), comp);
}

template < class T, class IndexType = int >
void compute_sort_permutation_quick(const std::vector< T > &values, 
                                     std::vector< IndexType > &v) 
{
  const size_t size = values.size();
  v.clear();
  v.resize(size);
  for (size_t i = 0; i != size; ++i) 
  {
    v[i] = static_cast< IndexType >(i);
  }

  auto comp = [values] (IndexType a, IndexType b) { return values[a] < values[b]; };

  std::sort(v.begin(), v.end(), comp);
}

template < class T, class IndexType = int >
void compute_sort_permutation_stable(const std::vector< T > &values,
                                     std::vector< T > & sorted_values,
                                     std::vector< IndexType > &v) 
{
  const size_t size = values.size();
  v.clear();
  v.resize(size);
  for (size_t i = 0; i != size; ++i) 
  {
    v[i] = static_cast< IndexType >(i);
  }

  auto comp = [values] (IndexType a, IndexType b) { return values[a] < values[b]; };

  std::stable_sort(v.begin(), v.end(), comp);
  
  sorted_values.resize(size);
  for (size_t i = 0; i != size; ++i) 
  {
    sorted_values[i] = values[v[i]];
  }
}

template < class T, class IndexType = int >
void compute_sort_permutation_quick(const std::vector< T > &values,
                                     std::vector< T > & sorted_values,
                                     std::vector< IndexType > &v) 
{
  const size_t size = values.size();
  v.clear();
  v.resize(size);
  for (size_t i = 0; i != size; ++i) 
  {
    v[i] = static_cast< IndexType >(i);
  }

  auto comp = [values] (IndexType a, IndexType b) { return values[a] < values[b]; };

  std::sort(v.begin(), v.end(), comp);
  
  sorted_values.resize(size);
  for (size_t i = 0; i != size; ++i) 
  {
    sorted_values[i] = values[v[i]];
  }
}


template < class T, class IndexType = int >
void compute_sort_permutation_stable_2d(const std::vector< T > &row_values,
                           const std::vector< T > &col_values, 
                           std::vector< IndexType > &v) 
{
  const size_t size = row_values.size();
  
  assert (size == col_values.size());

  v.clear();
  v.resize(size);
  for (size_t i = 0; i != size; ++i) 
  {
    v[i] = static_cast< IndexType >(i);
  }

  auto comp = [row_values, col_values] (IndexType a, IndexType b) 
  {
    if (row_values[a] < row_values[b])
    {
      return true;
    }
    else if (row_values[a] == row_values[b])
    {
      return col_values[a] < col_values[b]; 
    }
    return false;
  };

  std::stable_sort(v.begin(), v.end(), comp);
}

/// sort and remove duplicate entries in vector values
// -> store result in reduced_values 
// the map V is defined such that
// values[i] = reduced_values[V[i]]
// v and v_inv are temporary objects

template < class ValueType, class IndexType = int >
void sort_and_reduce(const std::vector< ValueType > &values,
                     std::vector< ValueType >& reduced_values,
                     std::vector< IndexType > &v,
                     std::vector< IndexType > &v_inv,
                     std::vector< IndexType > &V) 
{
  const size_t size = values.size();
  if (size == 0)
  {
    reduced_values.clear();
    v.clear();
    return;
  }

  // sort the values 
  // -> with permutationv such that
  // reduced_values[i] = values[v[i]]
  compute_sort_permutation_stable(values, reduced_values, v);

  // inverse mapping : v_inv[ v[i]] = i
  // -> v[ v_inv[i] ] = i
  // -> v[v_inv[i]] = v_inv[v[i]]
  // -> values [i] = values [v_inv[v[i]]] = values [v[v_inv[i]]] = reduced_values[v_inv[i]] 

  v_inv.clear();
  v_inv.resize(size,-1);
  for (size_t i = 0; i != size; ++i)
  {
    v_inv[v[i]] = i;
  }

  // remove duplicates
  // see implementation of std::unique
  auto first = reduced_values.begin();
  const auto last = reduced_values.end();

  // goal:
  // values[i] = (new) reduced_values[V[i]]

  // (old) reduced_values[i] = (new) reduced_values[v_r[i]]
  
  // set V[i] := v_r[v_inv[i]]
  // then: (new) reduced_values[V[i]] = (new) reduced_values[v_r[v_inv[i]]] 
  //                                  = (old) reduced_values[v_inv[i]]
  //                                  = values[i]

  // use v as v_r
  v.clear();
  v.resize(size, -1);

  auto result = first;
  IndexType pos = 0;
  IndexType ctr = 0;
  
  //std::cout << " size " << size << std::endl;
  //std::cout << string_from_range(reduced_values.begin(), reduced_values.end()) << std::endl;

  v[0] = 0;
  while (++first != last) 
  { /*
    std::cout << "ctr " << ctr << " first " << std::distance(reduced_values.begin(), first)  
                               << " result " << std::distance(reduced_values.begin(), result) 
                               << " || *first " << *first
                               << " *result " << *result << std::endl;*/
    if (!(*result == *first) )  
    {
      if (++result != first)
      {
        *result = std::move(*first);
      }
      pos++;
      //std::cout << "pos " << pos << " new result " << *result << std::endl;
    }
    v[++ctr] = pos;
  }

  //std::cout << "v " << string_from_range(v.begin(), v.end()) << std::endl;

  V.clear();
  V.resize(size, -1);
  for (size_t i = 0; i != size; ++i)
  {
    V[i] = v[v_inv[i]];
  }
  
  reduced_values.erase( ++result, reduced_values.end());
  /*
  std::cout << "reduced values " << string_from_range(reduced_values.begin(), reduced_values.end()) << std::endl;

  for (size_t i = 0; i != size; ++i)
  {
    std::cout << "i " << i << " V[i] " << V[i] << " val[i] " << values[i] << " red[v[i]] " << reduced_values[V[i]] << std::endl;
  }
  */
  assert (ctr == size-1);
  assert (reduced_values.size() == pos+1);
}

template<class DataType> 
void sort_and_erase_duplicates (std::vector<DataType>& vec)
{
  std::sort( vec.begin(), vec.end() );
  vec.erase( unique( vec.begin(), vec.end() ), vec.end() );
}

template<class DataType>
void log_2d_array(const std::vector< std::vector< DataType> >& vals,
                  std::ostream &s, 
                  int precision = 6) 
{
  s << std::setprecision(precision);
  for (size_t l=0; l<vals.size(); ++l)
  {
    for (size_t k=0; k<vals[l].size(); ++k)
    {
      s << vals[l][k] << " ";
    }
    s << "\n";
  }
}  
  
template <typename T, typename S>
void set_to_value(const size_t size, T val, std::vector<S> & array)
{
  array.resize(size, val);
  std::fill(array.begin(), array.end(), static_cast<S>(val));
}

template <typename T>
void set_to_zero(const size_t size, std::vector<T> & array)
{
  set_to_value(size, static_cast<T>(0), array);
}

template <typename T>
void set_to_unitvec_i(const size_t size, size_t i, std::vector<T> & array)
{
  assert (i >= 0);
  assert (i < size);

  set_to_zero(size, array);
  array[i] = static_cast<T>(1);
}

template <class DataType>
void double_2_datatype (const std::vector<double>& in_array, 
                        std::vector<DataType>& out_array)
{
  const size_t size = in_array.size();
  out_array.resize(size);
  
  for (size_t i=0; i<size; ++i)
  {
    out_array[i] = static_cast<DataType>(in_array[i]);
  }
}

template <class DataType, class T, int DIM, class VectorType>
void interlaced_coord_to_points (const std::vector<DataType>& inter_coords,   
                                 std::vector< VectorType >& points)
{
  const size_t num_vert = static_cast<size_t> (inter_coords.size() / DIM);
  points.resize(num_vert);
  
  for (size_t i=0; i<num_vert; ++i)
  {
    for (size_t d=0; d<DIM; ++d)
    {
      points[i].set(d, static_cast<T>(inter_coords[i * DIM + d]));
    }
  }
}

template <class DataType, class T, int DIM, class VectorType>
void points_to_interlaced_coord(const std::vector<VectorType> &points,
                                std::vector<DataType> &inter_coords)
{
  const size_t nverts = points.size();
  inter_coords.resize(nverts * DIM);

  for (size_t i = 0; i < nverts; ++i) {
    for (size_t d = 0; d < DIM; ++d) {
      inter_coords[i * DIM + d] = static_cast<T>(points[i][d]);
    }
  }
}

template <typename T, class VectorType>
void flatten_2d_vec_array(const std::vector< std::vector< VectorType > >& input, 
                          std::vector<T> & output, 
                          std::vector<size_t>& sizes) 
{
  if (input.size() == 0)
  {
    output.clear();
    return;
  }

  sizes.clear();
  size_t total_size = 0;
  
  for (size_t d=0; d<input.size(); ++d)
  {
    sizes.push_back(input[d].size());
    total_size += input[d].size();
  }
  
  const int DIM = input[0][0].size();
  output.clear();
  output.resize(total_size*DIM);
  
  size_t offset = 0;
  for (size_t d=0; d<input.size(); ++d)
  {
    for (size_t i=0; i<sizes[d]; ++i)
    {
      for (size_t l=0; l<DIM; ++l)
      {
        output[offset+l] = input[d][i][l];
      }
      offset += DIM;
    }
  }
}

template <typename T>
void flatten_2d_array(const std::vector< std::vector<T> >& input, 
                      std::vector<T> & output, 
                      std::vector<size_t>& sizes) 
{
  if (input.size() == 0)
  {
    output.clear();
    return;
  }

  sizes.clear();
  size_t total_size = 0;
  
  for (size_t d=0; d<input.size(); ++d)
  {
    sizes.push_back(input[d].size());
    total_size += input[d].size();
  }
  
  output.clear();
  output.resize(total_size);
  
  size_t offset = 0;
  for (size_t d=0; d<input.size(); ++d)
  {
    for (size_t i=0; i<sizes[d]; ++i)
    {
      output[offset+i] = input[d][i];
    }
    offset += sizes[d];
  }
}

template <typename T>
void expand_to_2d_array(const std::vector<T> & input,
                        const std::vector<size_t>& sizes,
                        std::vector< std::vector<T> >& output)
{
  if (input.size() == 0)
  {
    output.clear();
    return;
  }
  
  const size_t n = sizes.size();
  if (output.size() != n)
  {
    output.resize(n);
  }
  
  size_t offset = 0;
  for (size_t d=0; d<n; ++d)
  {
    const size_t m = sizes[d];
    if (output[d].size() != m)
    {
      output[d].resize(m);
    }
    
    for (size_t i=0; i<m; ++i)
    {
      output[d][i] = input[offset+i];
    }
    offset += m;
  }
}

template <typename T, int DIM, class VectorType>
void expand_to_2d_vec_array(const std::vector<T> & input,
                            const std::vector<size_t>& sizes,
                            std::vector< std::vector<VectorType > >& output)
{
  if (input.size() == 0)
  {
    output.clear();
    return;
  }
  
  const size_t n = sizes.size();
  if (output.size() != n)
  {
    output.resize(n);
  }
  
  size_t offset = 0;
  for (size_t d=0; d<n; ++d)
  {
    const size_t m = sizes[d];
    if (output[d].size() != m)
    {
      output[d].resize(m);
    }
    
    for (size_t i=0; i<m; ++i)
    {
      for (size_t l=0; l<DIM; ++l)
      {
        output[d][i].set(l, input[offset+l]);
      }
      offset += DIM; 
    }
  }
}

template <typename T>
bool vectors_are_equal(const std::vector<T>& arg1,
                       const std::vector<T>& arg2,
                       T eps)
{
  if (arg1.size() != arg2.size())
  {
    return false;
  }
  if (arg1.size() == 0)
  {
    return true;
  }
  
  const size_t len = arg1.size();
  auto it1 = arg1.begin();
  auto it2 = arg2.begin();
   
  for (size_t i=0; i!=len; ++i)
  {
    if (std::abs(*it1 - *it2) > eps * (*it1))
    {
      return false;
    }
    ++it1;
    ++it2;
  }
  return true;
}

template <typename T, class VectorType>
bool vectors_are_equal(const std::vector<VectorType >& arg1,
                       const std::vector<VectorType >& arg2,
                       T eps)
{
  if (arg1.size() != arg2.size())
  {
    return false;
  }
  if (arg1.size() == 0)
  {
    return true;
  }
  
  const size_t len = arg1.size();
  auto it1 = arg1.begin();
  auto it2 = arg2.begin();
   
  for (size_t i=0; i!=len; ++i)
  {
    if (it1->neq(*it2, eps))
    {
      return false;
    }
    ++it1;
    ++it2;
  }
  return true;
}

template <typename T>
bool contains_nan (const std::vector<T>& arg)
{
  for (size_t i=0, e=arg.size(); i!=e; ++i)
  {
    if (std::isnan(arg[i]))
    {
      return true;
    }
  }
  return false;
}

template <typename T>
bool contains_nan (const T* vals, int size)
{
  for (size_t i=0; i!=size; ++i)
  {
    if (std::isnan(vals[i]))
    {
      return true;
    }
  }
  return false;
}

template <typename T>
bool contains_value (const T& val, const std::vector<T>& arg)
{
  for (size_t i=0, e=arg.size(); i!=e; ++i)
  {
    if (val == arg[i])
    {
      return true;
    }
  }
  return false;
}

template <typename T>
T norm1 (const std::vector<T>& arg)
{
  T res = 0;
  for (size_t i=0, e=arg.size(); i!=e; ++i)
  {
    res += std::abs(arg[i]);
  }
  return res;
}

template<class T>
void create_unique_objects_mapping (const std::vector<T*>& in_objects,
                                    std::vector< int >& ind_2_unique,
                                    std::vector< std::vector< int > >& unique_2_ind,
                                    std::vector< T* >& unique_objects)
{
  int num_obj = in_objects.size();
  ind_2_unique.clear();
  ind_2_unique.resize(num_obj, -1);

  unique_objects.clear();
  unique_2_ind.clear();

  for (int l=0; l!=num_obj; ++l)
  {
    T* l_obj = in_objects[l];
    const int num_unique = unique_objects.size();
    bool found_obj = false;
    int found_ind = -1;
    for (int k=0; k!=num_unique; ++k)
    {
      if (*(unique_objects[k]) == *l_obj)
      {
        found_obj = true;
        found_ind = k;
        break;
      }
    }
    if (found_obj)
    {
      ind_2_unique[l] = found_ind;
    }
    else
    {
      ind_2_unique[l] = unique_objects.size();
      unique_objects.push_back(l_obj);
    }
  }

  unique_2_ind.resize(unique_objects.size());
  for (int l=0; l!=num_obj; ++l)
  {
    unique_2_ind[ind_2_unique[l]].push_back(l);
  }  
}

} // namespace hiflow

#endif /* _DATA_TOOLS_H_ */
