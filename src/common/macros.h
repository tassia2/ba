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

#ifndef _macros_h_
#define _macros_h_

#include <iostream>
#include <stdlib.h>

/// @author Staffan Ronnas

/// The macro here() prints the filename, line number and a message (can be
/// left empty, so it is still possible to call here()) to the standard output
/// stream.

#define here(message)                                                          \
  do {                                                                         \
    std::cout << "-> here() called in " << __FILE__ << " line " << __LINE__    \
              << ". " << #message << std::endl;                                \
  } while (0);

/// The macro interminable_assert(test) checks 'test' and breaks if false, else
/// nothing happens. This is comparable to the assert of macro 'assert()' from
/// assert.h of common C/C++ libraries. The major difference is, that it doesn't
/// depend on the symbol NDEBUG. This means this macro is alway active.

#define interminable_assert(test)                                              \
  do {                                                                         \
    if (!(test)) {                                                             \
      std::cerr << "Error: " << __FILE__ << ", " << __LINE__                   \
                << ": Assertion '" << #test << "' failed!" << std::endl;       \
      exit(-1);                                                                \
    }                                                                          \
  } while (0);

// #ifndef NDEBUG  
# define eq_assert(lhs, rhs)                                                   \
  do {                                                                         \
    if (!(lhs == rhs)) {                                                       \
      std::cerr << "Error: " << __FILE__ << ", " << __LINE__                   \
                << ": Assertion '" << lhs << " == " << rhs                     \
                << "' failed!" << std::endl;                                   \
      exit(-1);                                                                \
    }                                                                          \
  } while (0);
// #else
// # define eq_assert(lhs, rhs)                                                   \
// #endif

// #ifndef NDEBUG  
# define l_assert(lhs, rhs)                                                   \
  do {                                                                         \
    if (!(lhs < rhs)) {                                                       \
      std::cerr << "Error: " << __FILE__ << ", " << __LINE__                   \
                << ": Assertion '" << lhs << " < " << rhs                     \
                << "' failed!" << std::endl;                                   \
      exit(-1);                                                                \
    }                                                                          \
  } while (0);
// #else
// # define eq_assert(lhs, rhs)                                                   \
// #endif
 
/// This macro writes the actual file and line to the error stream and
/// terminates the program.

#define quit_program()                                                         \
  do {                                                                         \
    std::cerr << "Information: Program was terminated. " << __FILE__ << ", "   \
              << __LINE__ << "." << std::endl;                                 \
    exit(-1);                                                                  \
  } while (0);

#define not_implemented()                                                         \
  do {                                                                         \
    std::cerr << "Called routine is not implemented for class specialization " << __FILE__ << ", "   \
              << __LINE__ << "." << std::endl;                                 \
    exit(-1);                                                                  \
  } while (0);


#ifdef GCC_COMPILER
# define PRAGMA_LOOP_VEC 
#else
# ifdef CLANG_COMPILER 
#   define PRAGMA_LOOP_VEC _Pragma("clang loop vectorize(enable)") 
# else 
#   ifdef INTEL_COMPILER
#     define PRAGMA_LOOP_VEC _Pragma("vector always") 
#   else
#     define PRAGMA_LOOP_VEC 
#   endif
# endif
#endif

// check "test_has_member.cpp" for a usage example

/// Defines a "has_member_member_name" class template
///
/// This template can be used to check if its "T" argument
/// has a data or function member called "member_name"
#define define_has_member(member_name)                                         \
    template <typename T>                                                      \
    class has_member_##member_name                                             \
    {                                                                          \
        typedef char yes_type;                                                 \
        typedef long no_type;                                                  \
        template <typename U> static yes_type test(decltype(&U::member_name)); \
        template <typename U> static no_type  test(...);                       \
    public:                                                                    \
        static constexpr bool value = sizeof(test<T>(0)) == sizeof(yes_type);  \
    }

/// Shorthand for testing if "class_" has a member called "member_name"
///
/// @note "define_has_member(member_name)" must be used
///       before calling "has_member(class_, member_name)"
#define has_member(class_, member_name)  has_member_##member_name<class_>::value

// TODO
// _Pragma("GCC ivdep") 
#endif