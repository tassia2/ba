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

#ifndef __DOF_FEM_TYPES_H_
#define __DOF_FEM_TYPES_H_
#include "common/pointers.h"

/// \author Michael Schick<br>Martin Baumann<br>Philipp Gerstner

namespace hiflow {
namespace doffem {
  template <class DataType, int DIM> class CellTransformation;
  template <class DataType, int DIM> class FETransformation;
  template <class DataType, int RDIM, int PDIM> class SurfaceTransformation;
  template <class DataType, int DIM> class RefCell;
  template <class DataType, int DIM> class DofContainer;
  template <class DataType, int DIM> class DofContainerLagrange;
  template <class DataType, int DIM> class DofContainerRTBDM;
  template <class DataType, int DIM> class AnsatzSpace;
  template <class DataType, int DIM> class AnsatzSpaceSum;
  template <class DataType, int DIM> class RefElement;

  template <class DataType, int DIM> 
  using CellTrafoSPtr = SharedPtr< CellTransformation<DataType, DIM> >;
  
  template <class DataType, int DIM> 
  using CCellTrafoSPtr = SharedPtr< const CellTransformation<DataType, DIM> >;

  template <class DataType, int DIM> 
  using FETrafoSPtr = SharedPtr< FETransformation<DataType, DIM> >;

  template <class DataType, int DIM> 
  using CFETrafoSPtr = SharedPtr< const FETransformation<DataType, DIM> >;

  template <class DataType, int DIM> 
  using FETrafoUPtr = UniquePtr< FETransformation<DataType, DIM> >;

  template <class DataType, int RDIM, int PDIM> 
  using SurfaceTrafoSPtr = SharedPtr< SurfaceTransformation<DataType, RDIM, PDIM> >;
  
  template <class DataType, int RDIM, int PDIM> 
  using CSurfaceTrafoSPtr = SharedPtr< const SurfaceTransformation<DataType, RDIM, PDIM> >;
      
  template <class DataType, int DIM> 
  using RefCellSPtr = SharedPtr< RefCell<DataType, DIM> >;

  template <class DataType, int DIM> 
  using CRefCellSPtr = SharedPtr< const RefCell<DataType, DIM> >;  

  template <class DataType, int DIM> 
  using DofContainerSPtr = SharedPtr< DofContainer<DataType, DIM> >;  

  template <class DataType, int DIM> 
  using CDofContainerSPtr = SharedPtr< const DofContainer<DataType, DIM> >;  

  template <class DataType, int DIM> 
  using DofContainerUPtr = UniquePtr< DofContainer<DataType, DIM> >;

  template <class DataType, int DIM> 
  using DofContainerLagSPtr = SharedPtr< DofContainerLagrange<DataType, DIM> >;  

  template <class DataType, int DIM> 
  using CDofContainerLagSPtr = SharedPtr< const DofContainerLagrange<DataType, DIM> >;  

  template <class DataType, int DIM> 
  using DofContainerLagUPtr = UniquePtr< DofContainerLagrange<DataType, DIM> >;

  template <class DataType, int DIM> 
  using DofContainerRtBdmSPtr = SharedPtr< DofContainerRTBDM<DataType, DIM> >;  

  template <class DataType, int DIM> 
  using CDofContainerRtBdmSPtr = SharedPtr< const DofContainerRTBDM<DataType, DIM> >;  

  template <class DataType, int DIM> 
  using DofContainerRtBdmUPtr = UniquePtr< DofContainerRTBDM<DataType, DIM> >;

  template <class DataType, int DIM> 
  using AnsatzSpaceSPtr = SharedPtr< AnsatzSpace<DataType, DIM> >;  

  template <class DataType, int DIM> 
  using CAnsatzSpaceSPtr = SharedPtr< const AnsatzSpace<DataType, DIM> >;  

  template <class DataType, int DIM> 
  using AnsatzSpaceUPtr = UniquePtr< AnsatzSpace<DataType, DIM> >;  

  template <class DataType, int DIM> 
  using AnsatzSpaceSumSPtr = SharedPtr< AnsatzSpaceSum<DataType, DIM> >;  

  template <class DataType, int DIM> 
  using CAnsatzSpaceSumSPtr = SharedPtr< const AnsatzSpaceSum<DataType, DIM> >;  

  template <class DataType, int DIM> 
  using AnsatzSpaceSumUPtr = UniquePtr< AnsatzSpaceSum<DataType, DIM> >;  

  template <class DataType, int DIM> 
  using RefElementSPtr = SharedPtr< RefElement<DataType, DIM> >;  

  template <class DataType, int DIM> 
  using CRefElementSPtr = SharedPtr< const RefElement<DataType, DIM> >;  

  template <class DataType, int DIM> 
  using RefElementUPtr = UniquePtr< RefElement<DataType, DIM> >; 

  template <class DataType, int DIM> 
  using RefElementWPtr = WeakPtr< RefElement<DataType, DIM> >; 

  template <class DataType, int DIM> 
  using CRefElementWPtr = WeakPtr< const RefElement<DataType, DIM> >; 

  // dof numbering within cell
  typedef int cDofId;

  // dof numbering within local subdomain
  typedef int lDofId;

  // dof numbering within complete mesh
  // TODO: cmake option for long int
  typedef int gDofId;
  
  // deprecated 
  typedef int DofID;

  typedef int FETypeID;
  
  enum class RefCellType
  {
    NOT_SET = 0,
    LINE_STD = 1,
    TRI_STD = 2,
    QUAD_STD = 3,
    TET_STD = 4,
    HEX_STD = 5,
    PYR_STD = 6,
  };

  // Type of dof collection defining a FE on reference cell
  enum class DofContainerType
  {
    NOT_SET = 0,
    LAGRANGE = 1,
    BDM = 2,
    RT = 3
  };
  
  enum class DofFunctionalType 
  {
    NOT_SET = 0,
    POINT_EVAL = 1,
    POINT_NORMAL_EVAL = 2,
    POINT_TANGENT_EVAL = 3,
    CELL_MOMENT = 4,
    FACET_MOMENT = 5
  };
  
  enum class DofConstraint
  {
    INTERIOR = 0,
    FACET = 1,
    EDGE = 2,
    VERTEX = 3  
  };

  enum class DofPosition
  {
    INTERIOR = 0,
    FACET = 1,
    EDGE = 2,
    VERTEX = 3,  
  };
  
  /// Enumeration of different DoF ordering strategies. DOF_ORDERING::HIFLOW_CLASSIC refers to
  /// the DoF numbering as always done in HiFlow3. The other two options allow
  /// permutations of the classic numbering by means of the Cuthill-McKee and the
  /// King method, respectively.

  enum class DOF_ORDERING 
  { 
    HIFLOW_CLASSIC, 
    CUTHILL_MCKEE, 
    KING 
  };

  enum class AnsatzSpaceType 
  {
    NOT_SET = 0,
    P_LAGRANGE = 1,
    Q_LAGRANGE = 2,
    SKEW_P_AUG = 3,
    P_AUG = 4,
    RT = 5,
    SUM = 6,
    BDM = 7,
    TRANSFORMED = 8
  };
  
  // FE type on reference cell
  enum class FEType 
  {
    NOT_SET = 0,
    CUSTOM = 1,
    LAGRANGE = 2,
    BDM = 3,
    RT = 4,
    LAGRANGE_VECTOR = 5,
    LAGRANGE_P = 6
  };
 
  enum class FEConformity
  {
    NONE = -1,
    L2 = 0,
    HDIV = 1,
    HCURL = 2,
    H1 = 3
  };

  enum class FETransformationType
  {
    STD = 0,
    PIOLA = 1
  };
} // namespace doffem
} // namespace hiflow

#endif
