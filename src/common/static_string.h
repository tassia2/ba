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

#ifndef HIFLOW_STATIC_STRING_H
#define HIFLOW_STATIC_STRING_H

#include <cassert>
#include <iostream>
#include <string>     
#include <iomanip> 
#include <array>
#include <charconv>
#include <system_error>

/// @author Philipp Gerstner
namespace hiflow {

/// @brief class implemnting string functionalities, however with compile-time determined capacity

template <int N> 
class StaticString 
{
public:

  StaticString() = default;

  StaticString(const char * c)
  {
    const size_t size_c = std::char_traits<char>::length(c);
    //std::cout << size_c << " : " << c << "|" <<  std::endl;
    assert (size_c <= N);

    for (int i=0; i!=size_c; ++i)
    {
      this->buffer_[i] = c[i];
    }
    this->size_ = size_c;
  }

  StaticString(const StaticString<N>& other) = default;
  
  StaticString& operator= (const StaticString<N>& other) = default;

  ~StaticString() = default;


  inline void clear() 
  {
    this->size_ = 0;
  }

  inline size_t size() const 
  {
    return this->size_;
  }

  inline char& operator[] (int i) 
  {
    assert (i < N);
    return this->buffer_[i];
  }
  
  inline const char& operator[] (int i) const 
  {
    assert (i < N);
    return this->buffer_[i];
  }

  void append (const char * c)
  {
    const size_t size_c = std::char_traits<char>::length(c);
    assert (size_ + size_c <= N);

    for (int i=0; i!=size_c; ++i)
    {
      buffer_[size_++] = c[i];
    }
  }

  template <int M>
  void append (const StaticString<M>& A)  
  {
    assert(A.size() + size_ <= N);

    for (size_t i = 0; i != A.size_; ++i)
    {
      buffer_[size_++] = A.buffer_[i];
    }
  }

  template <typename T>
  inline void append_value (const T& val)  
  {
    auto [ptr, ec] = std::to_chars(buffer_.data() + size_, buffer_.data() + N, val);
    size_ = ptr - buffer_.data();
    assert (ec == std::errc());
    assert (size_ <= N);    
  }
  
  template <int M>
  inline StaticString<N>& operator+ (const StaticString<M>& A)  
  {
    this->append(A);
    return *this;
  }

  inline StaticString<N>& operator+ (const char * c)
  {
    this->append(c);
    return *this;
  }

  /*
  template <typename T>
  inline StaticString<N>& operator+ (const T& val)  
  {
    this->append(val);
    return *this;
  }*/

  StaticString<N>& operator= (const char * c)
  {
    const size_t size_c = std::char_traits<char>::length(c);
    assert (size_c <= N);

    for (int i=0; i!=size_c; ++i)
    {
      this->buffer_[i] = c[i];
    }
    this->size_ = size_c;
    return *this;
  }

  template <int M>
  StaticString<N>& operator= (const StaticString<M>& other) 
  {
    assert (other.size_ <= N);
    this->size_ = other.size_;

    for (size_t i = 0; i != size_; ++i)
    {
      buffer_[i] = other.buffer_[i];
    }
    return *this;
  }

  bool operator== (const char * c) const
  {
    const size_t size_c = std::char_traits<char>::length(c);
    if (this->size_ != size_c)
    {
      return false;
    }
    for (size_t i = 0; i != size_; ++i)
    {
      if (buffer_[i] != c[i])
      {
        return false;
      }
    }
    return true;
  }

  template <int M>
  bool operator== (const StaticString<M>& other) const 
  {
    if (this->size_ != other.size_)
    {
      return false;
    }
    for (size_t i = 0; i != size_; ++i)
    {
      if (buffer_[i] != other.buffer_[i])
      {
        return false;
      }
    }
    return true;
  }

  inline bool operator!= (const char * c) const
  {
    return (!(*this == c));
  }

  template <int M>
  inline bool operator!= (const StaticString<M>& other) const 
  {
    return (!(*this == other));
  }

private:

  std::array<char, N> buffer_;
  size_t size_ = 0;
};


template<int M>
std::ostream& operator<< (std::ostream& stream, const StaticString<M>& string) 
{
  for (int i=0; i != string.size(); ++i)
  {
    stream << " " << string[i];
  }
  return stream;
}

} // namespace hiflow

#endif 
