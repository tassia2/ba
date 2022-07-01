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

/// \author Thomas Gengenbach, Staffan Ronnas

#ifndef _TEST_H_
#define _TEST_H_

#include <cstdlib>

static bool CONSOLE_OUTPUT_ACTIVE = true;
#define CONSOLE_OUTPUT(rank, x)                                                \
  {                                                                            \
    if (CONSOLE_OUTPUT_ACTIVE && rank == 0) {                                  \
      std::cout << x << "\n";                                                  \
    }                                                                          \
  }

#define CONSOLE_OUTPUT_PAR(rank, x)                                            \
  {                                                                            \
      std::cout <<  "[" << std::setw(3) << rank << "] : "<< x << "\n";         \
  }

#endif
