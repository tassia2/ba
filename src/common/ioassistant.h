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

#ifndef HIFLOW_IOASSISTANT_H_
#define HIFLOW_IOASSISTANT_H_

/// \file ioassistant.h
///
/// \author Michael Schick

#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace hiflow {

template < class DataType >
void write_2d_array(std::string filename, 
                    std::string delim, 
                    bool append,
                    int precision,
                    int rank,
                    const std::vector< std::vector<DataType> >& values)
{
  // file output
  if (rank == 0) 
  {
    std::string path = filename;
    std::ofstream out;
    if (append)
    {
      out.open(path.c_str(), std::ios::out | std::ios::app);
    }
    else
    {
      out.open(path.c_str(), std::ios::out );
    }
    if (out.is_open()) 
    {
      out.precision(precision);
      out << std::scientific;
      for (int i=0; i != values.size(); ++i)
      {
        for (int l=0; l != values[i].size()-1; ++l)
        {
          out << values[i][l] << delim; 
        }
        out << values[i][values[i].size()-1] << "\n";
      }
      out.close();
    }
  }
}


class IOAssistant {
public:
  IOAssistant() {}

  template < class DataType >
  inline void read(const std::string &filename,
                   std::vector< DataType > &result);
                   
  template < class DataType >
  inline void read_array (std::string filename, 
                          std::string seperator, 
                          int max_lines, 
                          std::vector< std::vector< DataType > > &data);
                              
  template < class DataType >
  inline void read_binary(const std::string &filename,
                          std::vector< DataType > &result);
                          
  template < class DataType >
  inline void write(const std::string &filename,
                    const std::vector< DataType > &input) const;
  
  template < class DataType >
  inline void write_binary(const std::string &filename,
                           const std::vector< DataType > &input) const;
};

/// INLINE FUNCTIONS

template < class DataType >
void IOAssistant::read(const std::string &filename,
                       std::vector< DataType > &result) {
  std::string strline; // Current line

  // Open file

  std::ifstream file((filename + ".txt").c_str());
  if (!file) {
    std::cerr << "\n Can't open file : " << filename + ".txt" << std::endl;
    exit(1);
  }

  // First line

  std::getline(file, strline);
  std::istringstream istl1(strline);

  int n;
  istl1 >> n;

  assert(n > 0);

  result.resize(n, 0.0);

  // Full vector

  for (int k = 0; k < n; ++k) {
    std::getline(file, strline);
    std::istringstream istlc(strline);

    istlc >> result[k];

  } // for(int k=0; ...
}

template < class DataType >
void IOAssistant::read_array (std::string filename, 
                              std::string seperator, 
                              int max_lines, 
                              std::vector< std::vector< DataType > > &data)
{
  std::ifstream in;
  in.open(filename.c_str());

  if (in.is_open()) 
  {
    const std::string &delim = seperator;

    std::string line;
    std::string strnum;
    int counter = 0;
    while (std::getline(in, line)) 
    {
      if (max_lines >= 0 && counter >= max_lines) 
      {
        break;
      }
      data.push_back(std::vector< DataType >());

      for (std::string::const_iterator i = line.begin(); i != line.end(); ++i) 
      {
        // If i is not a delim, then append it to strnum
        if (delim.find(*i) == std::string::npos) 
        {
          strnum += *i;
          if (i + 1 != line.end()) // If it's the last char, do not continue
            continue;
        }

        // if strnum is still empty, it means the previous char is also a
        // delim (several delims appear together). Ignore this char.
        if (strnum.empty())
          continue;

        // If we reach here, we got a number. Convert it to DataType.
        DataType number;

        std::istringstream(strnum) >> number;
        data.back().push_back(number);

        strnum.clear();
      }
      counter++;
    }
    in.close();
  }
}

template < class DataType >
void IOAssistant::read_binary(const std::string &filename,
                              std::vector< DataType > &result) {
  // Open file

  std::ifstream file;

  file.open((filename + ".bin").c_str(), std::ios::in | std::ios::binary);

  if (!file) {
    std::cerr << "\n Can't open file : " << filename + ".bin" << std::endl;
    exit(1);
  }

  int n;

  file.read((char *)&n, sizeof(int));

  assert(n > 0);

  // Reinit

  result.resize(n, 0.0);

  // Full vector

  file.read((char *)&(result.front()), sizeof(DataType) * n);

  file.close();
}

/// write

template < class DataType >
void IOAssistant::write(const std::string &filename,
                        const std::vector< DataType > &input) const {
  // Open the file

  std::ofstream file((filename + ".txt").c_str(), std::ios::out);

  if (!file) {
    std::cerr << "\n Can't open file : " << filename + ".txt" << std::endl;
    exit(1);
  }

  // Set output format

  file.setf(std::ios::scientific, std::ios::floatfield);
  file.precision(18);

  // First line

  file << input.size() << std::endl;

  // Loop on the coordinates

  for (int k = 0; k < input.size(); ++k) {
    file << input[k] << std::endl;
  }
}

template < class DataType >
void IOAssistant::write_binary(const std::string &filename,
                               const std::vector< DataType > &input) const {
  // Open the file

  std::ofstream file;
  file.open((filename + ".bin").c_str(), std::ios::out | std::ios::binary);

  if (!file) {
    std::cerr << "\n Can't open file : " << filename + ".bin" << std::endl;
    exit(1);
  }

  // First line

  int mysize = input.size();
  file.write((char *)&mysize, sizeof(int));

  // all the coordinates

  file.write((char *)&(input.front()), sizeof(DataType) * input.size());

  file.close();
}

} // namespace hiflow
#endif
