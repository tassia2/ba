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

/// @author Bernd Doser, HITS gGmbH
/// @date 2015-12-04

#include "linear_algebra/petsc_environment.h"
#include "petsc.h"

namespace hiflow {
namespace la {

PETScEnvironment::~PETScEnvironment() { finalize(); }

void PETScEnvironment::initialize(int argc, char **argv) {
  if (initialized_)
    return;
  PetscErrorCode ierr = PetscInitialize(&argc, &argv, NULL, NULL);
  initialized_ = true;
}

void PETScEnvironment::initialize() {
  if (initialized_)
    return;
  PetscErrorCode ierr = PetscInitializeNoArguments();
  initialized_ = true;
}

void PETScEnvironment::finalize() {
  if (!initialized_)
    return;
  PetscErrorCode ierr = PetscFinalize();
  initialized_ = false;
}

bool PETScEnvironment::initialized_ = false;

} // namespace la
} // namespace hiflow
