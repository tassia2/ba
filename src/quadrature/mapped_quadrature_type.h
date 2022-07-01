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

#ifndef _MAPPED_QUADRATURE_TYPE_H_
#define _MAPPED_QUADRATURE_TYPE_H_

#include "common/vector_algebra_descriptor.h"
#include "quadrature/quadraturetype.h"
#include "dof/dof_fem_types.h"

#include <vector>

namespace hiflow {

template < class DataType > class Quadrature;
template < class DataType > class QuadratureMapping;

template < class DataType >
class MappedQuadratureType : public QuadratureType< DataType > {
public:
  MappedQuadratureType(const Quadrature< DataType > &base_quadrature,
                       const QuadratureMapping< DataType > &mapping);

  MappedQuadratureType< DataType > *clone() const;

protected:
  using QuadratureType< DataType >::x_;
  using QuadratureType< DataType >::y_;
  using QuadratureType< DataType >::z_;
  using QuadratureType< DataType >::w_;
  using QuadratureType< DataType >::index_field_;
};

template < class DataType > class QuadratureMapping {
public:
  virtual ~QuadratureMapping() {}
  virtual void
  map_quadrature_data(const Quadrature< DataType > &base_quadrature,
                      std::vector< DataType > &target_x,
                      std::vector< DataType > &target_y,
                      std::vector< DataType > &target_z,
                      std::vector< DataType > &target_w) const = 0;
};

template < class DataType >
class QuadrilateralQuadratureMapping : public QuadratureMapping< DataType > {
public:
  QuadrilateralQuadratureMapping(int facet_number);

  virtual void
  map_quadrature_data(const Quadrature< DataType > &base_quadrature,
                      std::vector< DataType > &target_x,
                      std::vector< DataType > &target_y,
                      std::vector< DataType > &target_z,
                      std::vector< DataType > &target_w) const;

private:
  int facet_number_;
};

template < class DataType >
class TriangleQuadratureMapping : public QuadratureMapping< DataType > {
public:
  TriangleQuadratureMapping(int facet_number);

  virtual void
  map_quadrature_data(const Quadrature< DataType > &base_quadrature,
                      std::vector< DataType > &target_x,
                      std::vector< DataType > &target_y,
                      std::vector< DataType > &target_z,
                      std::vector< DataType > &target_w) const;

private:
  int facet_number_;
};

template < class DataType >
class HexahedralQuadratureMapping : public QuadratureMapping< DataType > {
public:
  HexahedralQuadratureMapping(int facet_number);

  virtual void
  map_quadrature_data(const Quadrature< DataType > &base_quadrature,
                      std::vector< DataType > &target_x,
                      std::vector< DataType > &target_y,
                      std::vector< DataType > &target_z,
                      std::vector< DataType > &target_w) const;

private:
  int facet_number_;
};

template < class DataType >
class TetrahedralQuadratureMapping : public QuadratureMapping< DataType > {
public:
  TetrahedralQuadratureMapping(int facet_number);

  virtual void
  map_quadrature_data(const Quadrature< DataType > &base_quadrature,
                      std::vector< DataType > &target_x,
                      std::vector< DataType > &target_y,
                      std::vector< DataType > &target_z,
                      std::vector< DataType > &target_w) const;

private:
  int facet_number_;
};

template < class DataType >
class PyramidQuadratureMapping : public QuadratureMapping< DataType > {
public:
  PyramidQuadratureMapping(int facet_number);

  virtual void
  map_quadrature_data(const Quadrature< DataType > &base_quadrature,
                      std::vector< DataType > &target_x,
                      std::vector< DataType > &target_y,
                      std::vector< DataType > &target_z,
                      std::vector< DataType > &target_w) const;

private:
  int facet_number_;
};

template < class DataType, int DIM >
class SubCellQuadratureMapping : public QuadratureMapping< DataType > {
public:
  using Coord = typename StaticLA<DIM, DIM, DataType>::ColVectorType;

  SubCellQuadratureMapping(doffem::CCellTrafoSPtr<DataType, DIM> c_trafo_1,
                           doffem::CCellTrafoSPtr<DataType, DIM> c_trafo_2);

  virtual void
  map_quadrature_data(const Quadrature< DataType > &base_quadrature,
                      std::vector< DataType > &target_x,
                      std::vector< DataType > &target_y,
                      std::vector< DataType > &target_z,
                      std::vector< DataType > &target_w) const;

private:
  doffem::CCellTrafoSPtr<DataType, DIM> c_trafo_1_;
  doffem::CCellTrafoSPtr<DataType, DIM> c_trafo_2_;
};

template < class DataType, int RDIM, int PDIM >
class SubEntityQuadratureMapping : public QuadratureMapping< DataType > {
public:
  using RCoord = typename StaticLA<PDIM, RDIM, DataType>::RowVectorType;
  using PCoord = typename StaticLA<PDIM, RDIM, DataType>::ColVectorType;

  SubEntityQuadratureMapping(doffem::CSurfaceTrafoSPtr<DataType, RDIM, PDIM> s_trafo,
                             doffem::CCellTrafoSPtr<DataType, PDIM> c_trafo);

  virtual void
  map_quadrature_data(const Quadrature< DataType > &base_quadrature,
                      std::vector< DataType > &target_x,
                      std::vector< DataType > &target_y,
                      std::vector< DataType > &target_z,
                      std::vector< DataType > &target_w) const;

private:
  doffem::CSurfaceTrafoSPtr<DataType, RDIM, PDIM> s_trafo_;
  doffem::CCellTrafoSPtr<DataType, PDIM> c_trafo_;
};

} // namespace hiflow

#endif /* _MAPPED_QUADRATURE_TYPE_H_ */
