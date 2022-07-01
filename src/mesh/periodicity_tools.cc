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

#include "mesh/periodicity_tools.h"
#include "mesh.h"
#include <cmath>

namespace hiflow {
namespace mesh {

const double PERIOD_TOL = 1.0e-13;

template < class DataType >
std::vector< std::vector< PeriodicBoundaryType > >
get_periodicity_type(const std::vector< DataType > &coordinates,
                     const GDim gdim,
                     const std::vector< MasterSlave > &period) {
  int num_points = coordinates.size() / gdim;
  int num_per = period.size();
  assert(num_per < gdim);

  std::vector< std::vector< PeriodicBoundaryType > > temp(num_points);
  for (int i = 0; i < num_points; ++i) {
    temp[i].resize(num_per, NO_PERIODIC_BDY);
  }

  for (int k = 0; k < num_per; k++) {
    int period_dir = period[k].index();

    assert(period_dir >= 0);
    assert(period_dir < gdim);

    for (int i = 0; i < num_points; ++i) {
      if (std::abs(coordinates[i * gdim + period_dir] - period[k].slave()) <
          PERIOD_TOL) {
        temp[i][k] = SLAVE_PERIODIC_BDY;
      }
      if (std::abs(coordinates[i * gdim + period_dir] - period[k].master()) <
          PERIOD_TOL) {
        temp[i][k] = MASTER_PERIODIC_BDY;
      }
    }
  }

  return temp;
}

template std::vector< std::vector< PeriodicBoundaryType > >
get_periodicity_type(const std::vector< double > &coordinates, const GDim gdim,
                     const std::vector< MasterSlave > &period);
template std::vector< std::vector< PeriodicBoundaryType > >
get_periodicity_type(const std::vector< float > &coordinates, const GDim gdim,
                     const std::vector< MasterSlave > &period);

template < class DataType, int DIM >
void get_periodicity_type(const Vec<DIM, DataType> &point,
                          const std::vector< MasterSlave > &period,
                          std::vector< PeriodicBoundaryType >& types) 
{
  const int num_per = period.size();
  assert(num_per < DIM);

  types.clear();
  types.resize(num_per, NO_PERIODIC_BDY);

  //LOG_INFO("point", point);
  
  for (int k = 0; k < num_per; k++) 
  {
    const int period_dir = period[k].index();

    assert(period_dir >= 0);
    assert(period_dir < DIM);
    
    /*
    LOG_INFO("period", k << " master " << period[k].master() << " slave " 
                         << period[k].slave() << " diff2slave = " << std::abs(point[period_dir] - period[k].slave())
                         << " diff2master = " << std::abs(point[period_dir] - period[k].master()));
    */
    if (std::abs(point[period_dir] - period[k].slave()) < PERIOD_TOL) 
    {
      types[k] = SLAVE_PERIODIC_BDY;
    }
    if (std::abs(point[period_dir] - period[k].master()) < PERIOD_TOL) 
    {
      types[k] = MASTER_PERIODIC_BDY;
    }
  }
}

template void get_periodicity_type <double, 1> (const Vec<1, double> &, const std::vector< MasterSlave > &,
                                                std::vector< PeriodicBoundaryType >& types);
template void get_periodicity_type <double, 2> (const Vec<2, double> &, const std::vector< MasterSlave > &,
                                                std::vector< PeriodicBoundaryType >& types);
template void get_periodicity_type <double, 3> (const Vec<3, double> &, const std::vector< MasterSlave > &,
                                                std::vector< PeriodicBoundaryType >& types);
template void get_periodicity_type <float, 1> (const Vec<1, float> &, const std::vector< MasterSlave > &,
                                                std::vector< PeriodicBoundaryType >& types);
template void get_periodicity_type <float, 2> (const Vec<2, float> &, const std::vector< MasterSlave > &,
                                                std::vector< PeriodicBoundaryType >& types);
template void get_periodicity_type <float, 3> (const Vec<3, float> &, const std::vector< MasterSlave > &,
                                                std::vector< PeriodicBoundaryType >& types);
                          
template < class DataType >
std::vector< DataType > periodify_slave_to_master(
    const std::vector< DataType > &coordinates, const GDim gdim,
    const std::vector< MasterSlave > &period, int which_period) {
  int num_points = coordinates.size() / gdim;
  std::vector< DataType > temp(coordinates);

  int num_per = period.size();

  if (which_period == -1) {
    for (int k = 0; k < num_per; k++) {
      for (int i = 0; i < num_points; ++i) {
        if (std::abs(temp[i * gdim + period[k].index()] - period[k].slave()) <
            PERIOD_TOL) {
          temp[i * gdim + period[k].index()] = period[k].master();
        }
      }
    }
  } else {
    assert(which_period >= 0);
    assert(which_period < num_per);

    for (int i = 0; i < num_points; ++i) {
      if (std::abs(temp[i * gdim + period[which_period].index()] -
                   period[which_period].slave()) < PERIOD_TOL) {
        temp[i * gdim + period[which_period].index()] =
            period[which_period].master();
      }
    }
  }

  return temp;
}

template std::vector< double > periodify_slave_to_master(
    const std::vector< double > &coordinates, const GDim gdim,
    const std::vector< MasterSlave > &period, int which_period);
template std::vector< float > periodify_slave_to_master(
    const std::vector< float > &coordinates, const GDim gdim,
    const std::vector< MasterSlave > &period, int which_period);

template < class DataType, int DIM >
void periodify_slave_to_master(const Vec<DIM, DataType> &point,
                               const std::vector< MasterSlave > &period, 
                               const int which_period,
                               Vec<DIM, DataType> &out_point) 
{
  out_point = point;
  const int num_per = period.size();

  if (which_period == -1) 
  {
    for (int k = 0; k < num_per; k++) 
    {
      if (std::abs(out_point[period[k].index()] - period[k].slave()) < PERIOD_TOL) 
      {
        out_point.set(period[k].index(), period[k].master());
      }
    }
  }
  else 
  {
    assert(which_period >= 0);
    assert(which_period < num_per);

    if (std::abs(out_point[period[which_period].index()] - period[which_period].slave()) < PERIOD_TOL) 
    {
      out_point.set(period[which_period].index(), period[which_period].master());
    }
  }
}

template void periodify_slave_to_master <double, 1> (const Vec< 1, double > &, const std::vector< MasterSlave > &, 
                                                     const int, Vec< 1, double > &); 
template void periodify_slave_to_master <double, 2> (const Vec< 2, double > &, const std::vector< MasterSlave > &, 
                                                     const int, Vec< 2, double > &); 
template void periodify_slave_to_master <double, 3> (const Vec< 3, double > &, const std::vector< MasterSlave > &, 
                                                     const int, Vec< 3, double > &); 
template void periodify_slave_to_master <float, 1> (const Vec< 1, float > &, const std::vector< MasterSlave > &, 
                                                     const int, Vec< 1, float > &);                                
template void periodify_slave_to_master <float, 2> (const Vec< 2, float > &, const std::vector< MasterSlave > &, 
                                                     const int, Vec< 2, float > &);   
template void periodify_slave_to_master <float, 3> (const Vec< 3, float > &, const std::vector< MasterSlave > &, 
                                                     const int, Vec< 3, float > &);   
                                                     
template < class DataType >
std::vector< DataType > periodify_master_to_slave(
    const std::vector< DataType > &coordinates, const GDim gdim,
    const std::vector< MasterSlave > &period, int which_period) {
  int num_points = coordinates.size() / gdim;
  std::vector< DataType > temp(coordinates);

  int num_per = period.size();

  if (which_period == -1) {
    for (int k = 0; k < num_per; k++) {
      for (int i = 0; i < num_points; ++i) {
        if (std::abs(temp[i * gdim + period[k].index()] - period[k].master()) <
            PERIOD_TOL) {
          temp[i * gdim + period[k].index()] = period[k].slave();
        }
      }
    }
  } else {
    assert(which_period >= 0);
    assert(which_period < num_per);

    for (int i = 0; i < num_points; ++i) {
      if (std::abs(temp[i * gdim + period[which_period].index()] -
                   period[which_period].master()) < PERIOD_TOL) {
        temp[i * gdim + period[which_period].index()] =
            period[which_period].slave();
      }
    }
  }

  return temp;
}

template std::vector< double > periodify_master_to_slave(
    const std::vector< double > &coordinates, const GDim gdim,
    const std::vector< MasterSlave > &period, int which_period);
template std::vector< float > periodify_master_to_slave(
    const std::vector< float > &coordinates, const GDim gdim,
    const std::vector< MasterSlave > &period, int which_period);

template < class DataType, int DIM >
void periodify_master_to_slave(const Vec<DIM, DataType> &point, 
                               const std::vector< MasterSlave > &period, 
                               int which_period,
                               Vec<DIM, DataType> &out_point) 
{
  out_point = point;
  int num_per = period.size();

  if (which_period == -1) 
  {
    for (int k = 0; k < num_per; k++) 
    {
      if (std::abs(out_point[period[k].index()] - period[k].master()) < PERIOD_TOL) 
      {
        out_point.set(period[k].index(), period[k].slave());
      }
    }
  }
  else 
  {
    assert(which_period >= 0);
    assert(which_period < num_per);

    if (std::abs(out_point[period[which_period].index()] - period[which_period].master()) < PERIOD_TOL) 
    {
      out_point.set(period[which_period].index(), period[which_period].slave());
    }
  }
}

template void periodify_master_to_slave <double, 1> (const Vec< 1, double > &, const std::vector< MasterSlave > &, 
                                                     const int, Vec< 1, double > &); 
template void periodify_master_to_slave <double, 2> (const Vec< 2, double > &, const std::vector< MasterSlave > &, 
                                                     const int, Vec< 2, double > &); 
template void periodify_master_to_slave <double, 3> (const Vec< 3, double > &, const std::vector< MasterSlave > &, 
                                                     const int, Vec< 3, double > &); 
template void periodify_master_to_slave <float, 1> (const Vec< 1, float > &, const std::vector< MasterSlave > &, 
                                                     const int, Vec< 1, float > &);                                
template void periodify_master_to_slave <float, 2> (const Vec< 2, float > &, const std::vector< MasterSlave > &, 
                                                     const int, Vec< 2, float > &);   
template void periodify_master_to_slave <float, 3> (const Vec< 3, float > &, const std::vector< MasterSlave > &, 
                                                     const int, Vec< 3, float > &);   


template < class DataType >
std::vector< DataType > unperiodify(const std::vector< DataType > &coordinates,
                                    const GDim gdim,
                                    const std::vector< MasterSlave > &period) 
{
  int num_points = coordinates.size() / gdim;
  std::vector< DataType > temp(coordinates);

  int num_per = period.size();
  for (int k = 0; k < num_per; k++) {
    bool slave_boundary = false;
    DataType boundary_band = period[k].h();

    for (int i = 0; i < num_points; ++i) {
      if (std::abs(temp[i * gdim + period[k].index()] - period[k].slave()) <
          boundary_band) {
        slave_boundary = true;
        break;
      }
    }

    // modify coordinates if necessary
    if (slave_boundary) {
      for (int i = 0; i < num_points; ++i) {
        if (std::abs(temp[i * gdim + period[k].index()] - period[k].master()) <
            PERIOD_TOL) {
          temp[i * gdim + period[k].index()] = period[k].slave();
        }
      }
    }
  }

  return temp;
}

template std::vector< double >
unperiodify(const std::vector< double > &coordinates, const GDim gdim,
            const std::vector< MasterSlave > &period);
template std::vector< float >
unperiodify(const std::vector< float > &coordinates, const GDim gdim,
            const std::vector< MasterSlave > &period);

template < class DataType >
std::vector< std::vector< DataType > >
unperiodify(const std::vector< std::vector< DataType > > &coordinates,
            const GDim gdim, const std::vector< MasterSlave > &period) {
  int num_points = coordinates.size();
  std::vector< std::vector< DataType > > temp(coordinates);

  int num_per = period.size();
  for (int k = 0; k < num_per; k++) {
    bool slave_boundary = false;
    DataType boundary_band = period[k].h();

    for (int i = 0; i < num_points; ++i) {
      assert(temp[i].size() == gdim);

      if (std::abs(temp[i][period[k].index()] - period[k].slave()) <
          boundary_band) {
        slave_boundary = true;
        break;
      }
    }

    // modify coordinates if necessary
    if (slave_boundary) {
      for (int i = 0; i < num_points; ++i) {
        if (std::abs(temp[i][period[k].index()] - period[k].master()) < PERIOD_TOL) {
          temp[i][period[k].index()] = period[k].slave();
        }
      }
    }
  }

  return temp;
}

template std::vector< std::vector< double > >
unperiodify(const std::vector< std::vector< double > > &coordinates,
            const GDim gdim, const std::vector< MasterSlave > &period);

template std::vector< std::vector< float > >
unperiodify(const std::vector< std::vector< float > > &coordinates,
            const GDim gdim, const std::vector< MasterSlave > &period);

} // namespace mesh
} // namespace hiflow
