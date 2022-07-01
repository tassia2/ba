#define BOOST_TEST_MODULE cell_transformation

#include <boost/test/unit_test.hpp>
#include <boost/test/included/unit_test.hpp>

#include "hiflow.h"

using namespace hiflow;

template<class DataType, int RDIM, int PDIM>
struct TrafoProperties {
  std::vector<double> coords;
  Mat<PDIM, RDIM, DataType> J0;
  Mat<RDIM, RDIM, DataType> H[PDIM];
};

template<int DIM>
bool vec_equal_eps(Vec<DIM, double> v1, Vec<DIM, double> v2, double eps)
{
  for (int i = 0; i < DIM; ++i) {
    if (std::abs(v1[i] - v2[i]) >= eps)
      return false;
  }
  return true;
}

template<int M, int N, class DataType>
bool mat_equal_eps(Mat<M, N, DataType> m1, Mat<M, N, DataType> m2, DataType eps)
{
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      if (std::abs(m2(i, j) - m1(i, j)) >= eps)
        return false;
    }
  }
  return true;
}

template<int RDIM, int PDIM>
void init_linear_line_properties(struct TrafoProperties<double, RDIM, PDIM>& props)
{
  if (PDIM == 1) {
    props.coords = { 6.3, 2.1 };
    double jvals[] = { -4.2 };
    props.J0 = Mat<PDIM, 1, double>(jvals);
  } else if (PDIM == 2) {
    props.coords = {
      -6.7, 2.1,
      -7.5, -4.6,
    };
    double jvals[] = {
      -0.8,
      -6.7, 
    };
    props.J0 = Mat<PDIM, 1, double>(jvals);
  } else if (PDIM == 3) {
    props.coords = {
      -1.6, -4.1, -9.8,
      5.3, -3.8, 8.0,
    };
    double jvals[] = {
      6.9, 
      0.3, 
      17.8,
    };
    props.J0 = Mat<PDIM, 1, double>(jvals);
  }
}
template<int RDIM, int PDIM>
void init_linear_triangle_properties(struct TrafoProperties<double, RDIM, PDIM>& props)
{
  if (PDIM == 2) {
    props.coords = {
      4.6, -1.8,
      -4.2, 9.3,
      0.6, -6.9,
    };
    const double jvals[] = {
      -8.8, -4.0,
      11.1, -5.1,
    };
    props.J0 = Mat<PDIM, 2, double>(jvals);
  } else if (PDIM == 3) {
    props.coords = {
      -6.1, 2.5, -9.5,
      -5.4, -9.7, 1.0,
      -4.5, -4.7, -2.5,
    };
    const double jvals[] = {
      0.7, 1.6, 
      -12.2, -7.2, 
      10.5, 7.0, 
    };
    props.J0 = Mat<PDIM, 2, double>(jvals);
  }
}
template<int DIM>
void init_aligned_quad_transformation(struct TrafoProperties<double, DIM, DIM>& props)
{
  props.coords = {
    4.8, 3.8,
    -6.3, 3.1,
    -8.7, -8.1,
    2.4, -7.4,
  };
  const double jvals[] = {
    -11.1, -2.4,
    -0.7, -11.2,
  };
  props.J0 = Mat<DIM, DIM, double>(jvals);
}

template<int RDIM, int PDIM>
void init_bilinear_quad_transformation(struct TrafoProperties<double, RDIM, PDIM>& props)
{
  if (PDIM == 2) {
    props.coords = {
      0.1, 3.3,
      8.7, -9.7,
      7.0, 5.1,
      -4.9, 2.3,
    };
    const double jvals[] = {
      8.6, -5.0,
      -13.0, -1.0,
    };
    props.J0 = Mat<PDIM, 2, double>(jvals);
  } else if (PDIM == 3) {
    props.coords = {
      -6.9, -8.8, 4.2,
      -4.9, 3.3, 6.8,
      7.3, -7.8, 8.0,
      5.3, 4.3, -6.1,
    };
    const double jvals[] = {
      2.0, 12.2, 
      12.1, 13.1,  
      2.6, -10.3, 
    };
    props.J0 = Mat<PDIM, 2, double>(jvals);
  }
}
template<int DIM>
void init_linear_tetrahedron_transformation(struct TrafoProperties<double, DIM, DIM>& props)
{
  props.coords = {
   -3.6, -0.5, -3.2,
   -1.5, 7.0, 5.9,
   -5.9, -4.5, 7.7,
   -7.3, 7.7, -6.9,
  };
  const double jvals[] = {
    2.1, -2.3, -3.7,
    7.5, -4.0, 8.2,
    9.1, 10.9, -3.7,
  };
  props.J0 = Mat<DIM, DIM, double>(jvals);
}
template<int DIM>
void init_linear_pyramid_transformation(struct TrafoProperties<double, DIM, DIM>& props)
{
  props.coords = {
    0.6, 1.9, 5.3,
    -9.9, -9.6, -0.7,
    -6.8, -3.0, -13.9,
    3.7, 8.5, -7.9,
    -0.8, -1.6, -2.7,
  };
  const double jvals[] = {
    -10.5, 3.1, 2.3,
    -11.5, 6.6, -1.05,
    -6.0, -13.2, 1.6,
  };
  props.J0 = Mat<DIM, DIM, double>(jvals);
}
template<int DIM>
void init_aligned_hexahedron_transformation(struct TrafoProperties<double, DIM, DIM>& props)
{
  props.coords = {
    -1.0, -1.9, -1.0,
    1.0, -0.9, -1.0,
    0.0, 1.1, 0.0,
    -2.0, 0.1, 0.0,
    -2.0, -0.9, 1.0,
    0.0, 0.1, 1.0,
    -1.0, 2.1, 2.0,
    -3.0, 1.1, 2.0,
  };
  const double jvals[] = {
    2.0, -1.0, -1.0,
    1.0, 2.0, 1.0,
    0.0, 1.0, 2.0,
  };
  props.J0 = Mat<DIM, DIM, double>(jvals);
}
template<int DIM>
void init_trilinear_hexahedron_transformation(struct TrafoProperties<double, DIM, DIM>& props)
{
  props.coords = {
    3.0, 3.0, 0.0,
    -2.88, 2.4, 0.5,
    -2.88, -2.4, 0.5,
    3.0, -3.0, 0.0,
    1.8, 1.8, 5.0,
    -1.92, 1.44, 4.5,
    -1.92, -1.44, 4.5,
    1.8, -1.8, 5.0,
  };
  const double jvals[] = {
    -5.88, 0.0, -1.2,
    -0.6, -6.0, -1.2,
    0.5, 0.0, 5.0,
  };
  props.J0 = Mat<DIM, DIM, double>(jvals);
}

template<int DIM>
void test_transformation(CellTrafoSPtr<double, DIM> trafo,
    struct TrafoProperties<double, DIM, DIM> props)
{
  const double TOL = trafo->get_ref_cell()->eps();

  Vec<DIM, double> x;
  x.Zeros();

  Mat<DIM, DIM, double> J0;
  trafo->J(x, J0);
  bool is_equal = mat_equal_eps<DIM, DIM, double> (J0, props.J0, TOL);
  BOOST_TEST(is_equal );

  BOOST_TEST(std::abs(trafo->detJ(x) - det(props.J0)) < TOL);

  std::vector<Vec<DIM, double>> refcoords = trafo->get_reference_coordinates();
  std::vector<Vec<DIM, double>> physcoords = trafo->get_coordinates();
  for (int i = 0; i < refcoords.size(); ++i) {
    Vec<DIM, double> ptmp, refcoord, physcoord;

    trafo->transform(refcoords[i], ptmp);
    BOOST_TEST(trafo->inverse(ptmp, refcoord));

    BOOST_TEST(trafo->inverse(physcoords[i], ptmp));
    trafo->transform(ptmp, physcoord);
    
    BOOST_TEST(vec_equal_eps<DIM>(refcoord, refcoords[i], TOL));
    BOOST_TEST(vec_equal_eps<DIM>(physcoord, physcoords[i], TOL));
  }
}

template<int RDIM, int PDIM>
void test_surface_transformation(SurfaceTrafoSPtr<double, RDIM, PDIM> trafo,
    struct TrafoProperties<double, RDIM, PDIM> props)
{
  const double TOL = trafo->get_ref_cell()->eps();

  Vec<RDIM, double> x;
  x.Zeros();

  Mat<PDIM, RDIM, double> J0;
  trafo->J(x, J0);
  bool is_equal = mat_equal_eps<PDIM, RDIM, double> (J0, props.J0, TOL);
  BOOST_TEST(is_equal );


  std::vector<Vec<RDIM, double>> refcoords = trafo->get_reference_coordinates();
  std::vector<Vec<PDIM, double>> physcoords = trafo->get_coordinates();
  for (int i = 0; i < refcoords.size(); ++i) {
    Vec<PDIM, double> ptmp, physcoord;
    Vec<RDIM, double> refcoord, ptmr;

    trafo->transform(refcoords[i], ptmp);
    BOOST_TEST(trafo->inverse(ptmp, refcoord));

    BOOST_TEST(trafo->inverse(physcoords[i], ptmr));
    trafo->transform(ptmr, physcoord);
    
    BOOST_TEST(vec_equal_eps<RDIM>(refcoord, refcoords[i], TOL));
    BOOST_TEST(vec_equal_eps<PDIM>(physcoord, physcoords[i], TOL));
  }
}


template<int DIM>
void test_linear_line_transformation()
{
  doffem::CRefCellSPtr<double, DIM> cell(new doffem::RefCellLineStd<double, DIM>());
  doffem::CellTrafoSPtr<double, DIM> trafo(
      new doffem::LinearLineTransformation<double, DIM>(cell));

  struct TrafoProperties<double, DIM, DIM> props;
  init_linear_line_properties<DIM>(props);
  trafo->reinit(props.coords);

  test_transformation<DIM>(trafo, props);
}

template<int DIM>
void test_linear_line_surface_transformation()
{
  doffem::CRefCellSPtr<double, 1> cell(new doffem::RefCellLineStd<double, 1>());
  doffem::SurfaceTrafoSPtr<double, 1, DIM> trafo(
      new doffem::LinearLineSurfaceTransformation<double, 1, DIM>(cell));

  struct TrafoProperties<double, 1, DIM> props;
  init_linear_line_properties<1, DIM>(props);
  trafo->reinit(props.coords);

  test_surface_transformation<1,DIM>(trafo, props);
}


template<int DIM>
void test_linear_triangle_transformation()
{
  doffem::CRefCellSPtr<double, DIM> cell(new doffem::RefCellTriStd<double, DIM>());
  doffem::CellTrafoSPtr<double, DIM> trafo(
      new doffem::LinearTriangleTransformation<double, DIM>(cell));

  struct TrafoProperties<double, DIM, DIM> props;
  init_linear_triangle_properties<DIM>(props);
  trafo->reinit(props.coords);

  test_transformation<DIM>(trafo, props);
}

template<int DIM>
void test_linear_triangle_surface_transformation()
{
  doffem::CRefCellSPtr<double, 2> cell(new doffem::RefCellTriStd<double, 2>());
  doffem::SurfaceTrafoSPtr<double, 2, DIM> trafo(
      new doffem::LinearTriangleSurfaceTransformation<double, 2, DIM>(cell));

  struct TrafoProperties<double, 2, DIM> props;
  init_linear_triangle_properties<2, DIM>(props);
  trafo->reinit(props.coords);

  test_surface_transformation<2,DIM>(trafo, props);
}

template<int DIM>
void test_bilinear_quad_surface_transformation()
{
  doffem::CRefCellSPtr<double, 2> cell(new doffem::RefCellQuadStd<double, 2>());
  doffem::SurfaceTrafoSPtr<double, 2, DIM> trafo(
      new doffem::BiLinearQuadSurfaceTransformation<double, 2, DIM>(cell));

  struct TrafoProperties<double, 2, DIM> props;
  init_bilinear_quad_transformation<2, DIM>(props);
  trafo->reinit(props.coords);

  test_surface_transformation<2,DIM>(trafo, props);
}

template<int DIM>
void test_aligned_quad_transformation()
{
  doffem::CRefCellSPtr<double, DIM> cell(new doffem::RefCellQuadStd<double, DIM>());
  doffem::CellTrafoSPtr<double, DIM> trafo(
      new doffem::LinearQuadTransformation<double, DIM>(cell));

  struct TrafoProperties<double, DIM, DIM> props;
  init_aligned_quad_transformation<DIM>(props);
  trafo->reinit(props.coords);

  test_transformation<DIM>(trafo, props);
}

template<int DIM>
void test_bilinear_quad_transformation()
{
  doffem::CRefCellSPtr<double, DIM> cell(new doffem::RefCellQuadStd<double, DIM>());
  doffem::CellTrafoSPtr<double, DIM> trafo(
      new doffem::BiLinearQuadTransformation<double, DIM>(cell));
  
  struct TrafoProperties<double, DIM, DIM> props;
  init_bilinear_quad_transformation<DIM>(props);
  trafo->reinit(props.coords);

  test_transformation<DIM>(trafo, props);
}

template<int DIM>
void test_linear_tetrahedron_transformation()
{
  doffem::CRefCellSPtr<double, DIM> cell(new doffem::RefCellTetStd<double, DIM>());
  doffem::CellTrafoSPtr<double, DIM> trafo(
      new doffem::LinearTetrahedronTransformation<double, DIM>(cell));
  
  struct TrafoProperties<double, DIM, DIM> props;
  init_linear_tetrahedron_transformation<DIM>(props);
  trafo->reinit(props.coords);

  test_transformation<DIM>(trafo, props);
}

template<int DIM>
void test_linear_pyramid_transformation()
{
  doffem::CRefCellSPtr<double, DIM> cell(new doffem::RefCellPyrStd<double, DIM>());
  doffem::CellTrafoSPtr<double, DIM> trafo(
      new doffem::LinearPyramidTransformation<double, DIM>(cell));
  
  struct TrafoProperties<double, DIM, DIM> props;
  init_linear_pyramid_transformation<DIM>(props);
  trafo->reinit(props.coords);

  test_transformation<DIM>(trafo, props);
}

template<int DIM>
void test_aligned_hexahedron_transformation()
{
  doffem::CRefCellSPtr<double, DIM> cell(new doffem::RefCellHexStd<double, DIM>());
  doffem::CellTrafoSPtr<double, DIM> trafo(
      new doffem::LinearHexahedronTransformation<double, DIM>(cell));

  struct TrafoProperties<double, DIM, DIM> props;
  init_aligned_hexahedron_transformation<DIM>(props);
  trafo->reinit(props.coords);

  test_transformation<DIM>(trafo, props);
}

template<int DIM>
void test_trilinear_hexahedron_transformation()
{
  doffem::CRefCellSPtr<double, DIM> cell(new doffem::RefCellHexStd<double, DIM>());
  doffem::CellTrafoSPtr<double, DIM> trafo(
      new doffem::TriLinearHexahedronTransformation<double, DIM>(cell));

  struct TrafoProperties<double, DIM, DIM> props;
  init_trilinear_hexahedron_transformation<DIM>(props);
  trafo->reinit(props.coords);

  test_transformation<DIM>(trafo, props);
}
BOOST_AUTO_TEST_CASE(cell_transformation)
{
  std::cout << "Test Linear Line" << std::endl;
  test_linear_line_transformation<1>();

  std::cout << "Test Surface Linear Line" << std::endl;
  test_linear_line_surface_transformation<2>();
  test_linear_line_surface_transformation<3>();

  std::cout << "Test Linear Triangle" << std::endl;
  test_linear_triangle_transformation<2>();

  std::cout << "Test Surface Linear Triangle" << std::endl;
  test_linear_triangle_surface_transformation<3>();

  std::cout << "Test Aligned Quad" << std::endl;
  test_aligned_quad_transformation<2>();

  std::cout << "Test BiLinear Quad" << std::endl;
  test_bilinear_quad_transformation<2>();

  std::cout << "Test Surface BiLinear Quad" << std::endl;
  test_bilinear_quad_surface_transformation<3>();

  std::cout << "Test Linear Tet" << std::endl;
  test_linear_tetrahedron_transformation<3>();

  std::cout << "Test Linear Pyramid" << std::endl;
  test_linear_pyramid_transformation<3>();

  std::cout << "Test Aligned Hex" << std::endl;
  test_aligned_hexahedron_transformation<3>();

  std::cout << "Test TriLinear Hex" << std::endl;
  test_trilinear_hexahedron_transformation<3>();
}
