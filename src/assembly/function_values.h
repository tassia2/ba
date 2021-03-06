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

/// \author Staffan Ronnas

#ifndef HIFLOW_ASSEMBLY_FUNCTION_VALUES_H
#define HIFLOW_ASSEMBLY_FUNCTION_VALUES_H

#include <algorithm>
#include <cassert>
#include <numeric>
#include <vector>

namespace hiflow {

/// Function concept
///
/// A class that models the Function concept must export the
/// following in its public interface:
///
/// - void Function::operator()(int i,
///                             const argument_type& in,
///                             value_type& out) const;
///   -> Evaluation of the function for argument with index i and value in.
///   After return of the function, out must contain the value of the function
///   for argument (i, in) .
///

/// \brief Evaluation and storage of function values
///
/// \details A FunctionValues object facilitates the evaluation of
/// a function for a set of arguments. It also provides storage
/// for the evaluated function values, making it useful to cache
/// precomputed function values.

template < class T > class FunctionValues {
public:
  /// \brief Type of function values. Must be default constructible.
  typedef T value_type;

  /// \brief Evaluates a function object at a set of argument
  /// values and stores the function values internally.
  ///
  /// \details The function is quite flexible with respect to
  /// the types of its two arguments. The precise requirements
  /// on these types is detailed next.
  ///
  /// The Arguments type must provide the function size(),
  /// returning the number of argument values, and an
  /// implementation of operator[] that takes an integer
  /// argument i and returns the i:th argument value. Both
  /// functions must be const-qualified. One can use for
  /// instance a std::vector<X> or FunctionValues<X>, where X is
  /// the type of the argument values provided to the function.
  ///
  /// The Function type only has to provide operator() with the
  /// following signature (or compatible):
  ///
  /// void operator()(int i, const arg_type& arg, value_type& function_value)
  /// const
  ///
  /// This function will be called once with each argument in
  /// the arguments container. In this call, i contains the
  /// index of the current argument in the container, and arg
  /// the argument value. The function must return the function
  /// value F(arg) in the reference parameter function_value.
  ///
  /// Instead of using a class that implements operator() in
  /// this way, it is also possible (with recent compilers) to
  /// use a pointer to a free function with the same signature.
  ///
  /// \param arguments  input values to the function object
  /// \param function   function object
  template < class Function >
  inline void compute_wo_arg(size_t num_q, const Function &function);
  
  template < class Arguments, class Function >
  inline void compute(const Arguments &arguments, const Function &function);

  template < class ArgumentsTypeA, class ArgumentsTypeB, class Function >
  inline void compute(const ArgumentsTypeA &arguments1, 
                      const ArgumentsTypeB &arguments2,
                      const Function &function);
              
  template < class ArgumentsTypeA, class ArgumentsTypeB, class ArgumentsTypeC, class Function >
  inline void compute(const ArgumentsTypeA &arguments1, 
                      const ArgumentsTypeB &arguments2,
                      const ArgumentsTypeC &arguments3,
                      const Function &function);
                      
  template < class ArgumentsTypeA, class ArgumentsTypeB, class ArgumentsTypeC, class ArgumentsTypeD, 
             class Function >
  inline void compute(const ArgumentsTypeA &arguments1, 
                      const ArgumentsTypeB &arguments2,
                      const ArgumentsTypeC &arguments3,
                      const ArgumentsTypeD &arguments4,
                      const Function &function);
                      
  template < class ArgumentsTypeA, class ArgumentsTypeB, class ArgumentsTypeC, class ArgumentsTypeD, 
             class ArgumentsTypeE, class Function >
  inline void compute(const ArgumentsTypeA &arguments1, 
                      const ArgumentsTypeB &arguments2,
                      const ArgumentsTypeC &arguments3,
                      const ArgumentsTypeD &arguments4,
                      const ArgumentsTypeE &arguments5,
                      const Function &function);
        
  template < class ArgumentsTypeA, class ArgumentsTypeB, class ArgumentsTypeC, class ArgumentsTypeD, 
             class ArgumentsTypeE, class ArgumentsTypeF, class Function >
  inline void compute(const ArgumentsTypeA &arguments1, 
                      const ArgumentsTypeB &arguments2,
                      const ArgumentsTypeC &arguments3,
                      const ArgumentsTypeD &arguments4,
                      const ArgumentsTypeE &arguments5,
                      const ArgumentsTypeF &arguments6,
                      const Function &function);

  /// \brief Access to the function value corresponding to
  /// argument i from the last call to compute().
  ///
  /// \param i  Index of the function value
  /// \return   The function value F(arg[i])
  inline const value_type &operator[](int i) const;
  inline value_type &operator[](int i);
  
  inline void set (int i, const value_type& val);

  /// \brief Number of function values computed in the last call to compute().
  inline int size() const;

  /// \brief Deletes all stored function values.
  void clear();

  /// \brief Fills the values with zeros
  /// \param size desired size of array
  void zeros(int size);

private:
  std::vector< value_type > values_;
};

template < class T >
template < class Function >
inline void FunctionValues< T >::compute_wo_arg(size_t num_q, const Function &function) 
{
  //values_.clear();
  if (values_.size() != num_q)
  {
    values_.resize(num_q);
  }
  
  for (size_t i = 0, i_e = num_q; i < i_e; ++i) 
  {
    function(i, values_[i]);
  }
}

template < class T >
template < class Arguments, class Function >
inline void FunctionValues< T >::compute(const Arguments &arguments, const Function &function) 
{
  //values_.clear();
  if (values_.size() != arguments.size())
  {
    values_.resize(arguments.size());
  }
  
  for (size_t i = 0, i_e = arguments.size(); i < i_e; ++i) 
  {
    function(i, arguments[i], values_[i]);
  }
}

template < class T >
template < class ArgumentsTypeA, class ArgumentsTypeB, class Function >
inline void FunctionValues< T >::compute(const ArgumentsTypeA &arguments1,
                                         const ArgumentsTypeB &arguments2,
                                         const Function &function) {

  assert (arguments1.size() == arguments2.size());

  //values_.clear();
  if (values_.size() != arguments1.size())
  {
    values_.resize(arguments1.size());
  }

  for (size_t i = 0, i_e = arguments1.size(); i < i_e; ++i) {
    function(i, arguments1[i], arguments2[i], values_[i]);
  }
}

template < class T >
template < class ArgumentsTypeA, class ArgumentsTypeB, class ArgumentsTypeC, class Function >
inline void FunctionValues< T >::compute(const ArgumentsTypeA &arguments1,
                                         const ArgumentsTypeB &arguments2,
                                         const ArgumentsTypeC &arguments3,
                                         const Function &function) {

  assert (arguments1.size() == arguments2.size());
  assert (arguments1.size() == arguments3.size());

  //values_.clear();
  if (values_.size() != arguments1.size())
  {
    values_.resize(arguments1.size());
  }

  for (size_t i = 0, i_e = arguments1.size(); i < i_e; ++i) {
    function(i, arguments1[i], arguments2[i], arguments3[i], values_[i]);
  }
}

template < class T >
template < class ArgumentsTypeA, class ArgumentsTypeB, class ArgumentsTypeC, class ArgumentsTypeD, class Function >
inline void FunctionValues< T >::compute(const ArgumentsTypeA &arguments1,
                                         const ArgumentsTypeB &arguments2,
                                         const ArgumentsTypeC &arguments3,
                                         const ArgumentsTypeD &arguments4,
                                         const Function &function) {

  assert (arguments1.size() == arguments2.size());
  assert (arguments1.size() == arguments3.size());
  assert (arguments1.size() == arguments4.size());

  //values_.clear();
  if (values_.size() != arguments1.size())
  {
    values_.resize(arguments1.size());
  }

  for (size_t i = 0, i_e = arguments1.size(); i < i_e; ++i) {
    function(i, arguments1[i], arguments2[i], arguments3[i], arguments4[i], values_[i]);
  }
}

template < class T >
template < class ArgumentsTypeA, class ArgumentsTypeB, class ArgumentsTypeC, class ArgumentsTypeD, class ArgumentsTypeE, class Function >
inline void FunctionValues< T >::compute(const ArgumentsTypeA &arguments1,
                                         const ArgumentsTypeB &arguments2,
                                         const ArgumentsTypeC &arguments3,
                                         const ArgumentsTypeD &arguments4,
                                         const ArgumentsTypeE &arguments5,
                                         const Function &function) {

  assert (arguments1.size() == arguments2.size());
  assert (arguments1.size() == arguments3.size());
  assert (arguments1.size() == arguments4.size());
  assert (arguments1.size() == arguments5.size());

  //values_.clear();
  if (values_.size() != arguments1.size())
  {
    values_.resize(arguments1.size());
  }

  for (size_t i = 0, i_e = arguments1.size(); i < i_e; ++i) {
    function(i, arguments1[i], arguments2[i], arguments3[i], arguments4[i], arguments5[i], values_[i]);
  }
}

template < class T >
template < class ArgumentsTypeA, class ArgumentsTypeB, class ArgumentsTypeC, class ArgumentsTypeD, class ArgumentsTypeE, class ArgumentsTypeF, class Function >
inline void FunctionValues< T >::compute(const ArgumentsTypeA &arguments1,
                                         const ArgumentsTypeB &arguments2,
                                         const ArgumentsTypeC &arguments3,
                                         const ArgumentsTypeD &arguments4,
                                         const ArgumentsTypeE &arguments5,
                                         const ArgumentsTypeF &arguments6,
                                         const Function &function) {

  assert (arguments1.size() == arguments2.size());
  assert (arguments1.size() == arguments3.size());
  assert (arguments1.size() == arguments4.size());
  assert (arguments1.size() == arguments5.size());
  assert (arguments1.size() == arguments6.size());
  
  //values_.clear();
  if (values_.size() != arguments1.size())
  {
    values_.resize(arguments1.size());
  }

  for (size_t i = 0, i_e = arguments1.size(); i < i_e; ++i) {
    function(i, arguments1[i], arguments2[i], arguments3[i], arguments4[i], arguments5[i], arguments6[i], values_[i]);
  }
}

template < class T >
inline const typename FunctionValues< T >::value_type &FunctionValues< T >::operator[](int i) const {
  assert(i >= 0);
  assert(i < values_.size());
  return values_[i];
}

template < class T >
inline typename FunctionValues< T >::value_type &FunctionValues< T >::operator[](int i) {
  assert(i >= 0);
  assert(i < values_.size());
  return values_[i];
}

template < class T >
inline void FunctionValues< T >::set(int i, const value_type& val)
{
  assert(i >= 0);
  assert(i < values_.size());
  values_[i] = val;
}

template < class T > 
inline int FunctionValues< T >::size() const 
{
  return values_.size();
}

template < class T > 
void FunctionValues< T >::clear() 
{ 
  values_.clear(); 
}

template < class T > 
void FunctionValues< T >::zeros(int size) 
{
  values_.clear();
  values_.resize(size);
}
} // namespace hiflow

#endif /* _FUNCTION_H_ */
