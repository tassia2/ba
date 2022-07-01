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

/// \author Martin Baumann

#include "horner.h"

#include "polynomials/lagrangecoeff1.h"
#include "polynomials/lagrangecoeff10.h"
#include "polynomials/lagrangecoeff11.h"
#include "polynomials/lagrangecoeff12.h"
#include "polynomials/lagrangecoeff13.h"
#include "polynomials/lagrangecoeff14.h"
#include "polynomials/lagrangecoeff15.h"
#include "polynomials/lagrangecoeff16.h"
#include "polynomials/lagrangecoeff17.h"
#include "polynomials/lagrangecoeff18.h"
#include "polynomials/lagrangecoeff19.h"
#include "polynomials/lagrangecoeff2.h"
#include "polynomials/lagrangecoeff20.h"
#include "polynomials/lagrangecoeff3.h"
#include "polynomials/lagrangecoeff4.h"
#include "polynomials/lagrangecoeff5.h"
#include "polynomials/lagrangecoeff6.h"
#include "polynomials/lagrangecoeff7.h"
#include "polynomials/lagrangecoeff8.h"
#include "polynomials/lagrangecoeff9.h"

#include "polynomials/lagrangepolynomial.h"

using namespace std;

#define USE_LOW_DEG_OPTIM

namespace hiflow {

/// Constructor

template < class T > LagrangePolynomial< T >::LagrangePolynomial() { reinit(); }

/// reinit

template < class T > void LagrangePolynomial< T >::reinit() {
  // Initialize pre computed polynomials

  int nlag = 20;

  _lag_coeff.reserve(nlag);

  _lag_coeff.push_back(&_lagrange_coeff1);
  _lag_coeff.push_back(&_lagrange_coeff2);
  _lag_coeff.push_back(&_lagrange_coeff3);
  _lag_coeff.push_back(&_lagrange_coeff4);
  _lag_coeff.push_back(&_lagrange_coeff5);
  _lag_coeff.push_back(&_lagrange_coeff6);
  _lag_coeff.push_back(&_lagrange_coeff7);
  _lag_coeff.push_back(&_lagrange_coeff8);
  _lag_coeff.push_back(&_lagrange_coeff9);
  _lag_coeff.push_back(&_lagrange_coeff10);
  _lag_coeff.push_back(&_lagrange_coeff11);
  _lag_coeff.push_back(&_lagrange_coeff12);
  _lag_coeff.push_back(&_lagrange_coeff13);
  _lag_coeff.push_back(&_lagrange_coeff14);
  _lag_coeff.push_back(&_lagrange_coeff15);
  _lag_coeff.push_back(&_lagrange_coeff16);
  _lag_coeff.push_back(&_lagrange_coeff17);
  _lag_coeff.push_back(&_lagrange_coeff18);
  _lag_coeff.push_back(&_lagrange_coeff19);
  _lag_coeff.push_back(&_lagrange_coeff20);
}

/// poly ...

template < class T >
T LagrangePolynomial< T >::poly(int deg, int i, const T &x) const {
  
#ifdef USE_LOW_DEG_OPTIM
  switch (deg)
  {
    case 0:
      return 1.;
      break;
    case 1:
      switch (i)
      {
        case 0: 
          return 1. - x;
        case 1: 
          return x;
      }
      break; 
    case 2:
      switch (i)
      {
        case 0: 
          return 2. * (x-0.5) * (x-1.);
        case 1: 
          return -4. * x * (x-1.);
        case 2:
          return 2. * x* (x - 0.5);
      }
      break;
  }
#else 
  if (deg == 0)
    return 1.;
#endif

  assert((i >= 0) && (i <= deg));
  // assert((x>=0.) && (x<=1. ));

  if (deg <= _lag_coeff.size()) {
    return _horner(deg, _lag_coeff[deg - 1]->poly(i), x);
  }
  return _lagrange_polynomial(deg, i, x);
}

template < class T >
T LagrangePolynomial< T >::poly_x(int deg, int i, const T &x) const {
  
#ifdef USE_LOW_DEG_OPTIM
  switch (deg)
  {
    case 0:
      return 0.;
      break;
    case 1:
      switch (i)
      {
        case 0: 
          return -1.;
        case 1: 
          return 1.;
      }
      break; 
    case 2:
      switch (i)
      {
        case 0: 
          return 4. * x - 3.;
        case 1: 
          return -8. * x + 4;
        case 2:
          return 4. * x - 1.;
      }
      break;
  }
#else 
  if (deg == 0)
    return 0.;
#endif

  assert((i >= 0) && (i <= deg));
  // assert((x>=0.) && (x<=1. ));

  if (deg <= _lag_coeff.size()) {
    return _horner(deg, _lag_coeff[deg - 1]->poly_x(i), x);
  }
  return _lagrange_polynomial_x(deg, i, x);
}

template < class T >
T LagrangePolynomial< T >::poly_xx(int deg, int i, const T &x) const {
  
#ifdef USE_LOW_DEG_OPTIM
  switch (deg)
  {
    case 0:
      return 0.;
      break;
    case 1:
      return 0.;
      break; 
    case 2:
      switch (i)
      {
        case 0: 
          return 4.;
        case 1: 
          return -8.;
        case 2:
          return 4.;
      }
      break;
  }
#else 
  if (deg <= 1)
    return 0.;
#endif

  assert((i >= 0) && (i <= deg));
  // assert((x>=0.) && (x<=1. ));

  if (deg <= _lag_coeff.size()) {
    return _horner(deg, _lag_coeff[deg - 1]->poly_xx(i), x);
  }
  return _lagrange_polynomial_xx(deg, i, x);
}

/// template instanciation

template class LagrangePolynomial< double >;
template class LagrangePolynomial< float >;
} // namespace hiflow
