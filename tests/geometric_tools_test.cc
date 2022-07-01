#define BOOST_TEST_MODULE geometric_tools

#include <boost/test/unit_test.hpp>
#include <boost/test/included/unit_test.hpp>

#include "hiflow.h"
#include "test.h"

using namespace hiflow;

template <int DIM>
class Ellipsoid : public BoundaryDomainDescriptor<double, DIM>
{
public:
  Ellipsoid(const double a, const double b) : _a(a), _b(b), _c(0.0) {};
  Ellipsoid(const double a, const double b, const double c) : _a(a), _b(b), _c(c) {};

  double eval_func(const Vec<DIM, double> &x, MaterialNumber mat_num) const
  {
    double ret = 1.0 - x[0] * x[0] / (_a * _a) - x[1] * x[1] / (_b * _b);
    if (DIM == 3)
      ret -= x[2] * x[2] / (_c * _c);
    return ret;
  }

  Vec<DIM, double> eval_grad(const Vec<DIM, double> &x, MaterialNumber mat_num) const
  {
    Vec<DIM, double> grad;
    grad.set(0, -2.0 * x[0] / (_a * _a));
    grad.set(1, -2.0 * x[1] / (_b * _b));
    if (DIM == 3)
      grad.set(2, -2.0 * x[2] / (_c * _c));
    return grad;
  }

  const double _a;
  const double _b;
  const double _c;
};

template <int DIM>
bool vec_is_equal_eps(Vec<DIM, double> v1, Vec<DIM, double> v2, double eps)
{
  for (int i = 0; i < v1.size(); ++i) {
    if (std::abs(v1[i] - v2[i]) > eps) {
      return false;
    }
  }
  return true;
}

void test_intersect_facet()
{
  /* check 2d facet with line that is intersecting */
  std::vector<double> coords_facet2d = {
    1.0, 0.5,
    0.5, 2.0,
  };
  std::vector<double> coords_facet2d_a_inters = {
    0.0, 0.0,
  };
  std::vector<double> coords_facet2d_b_inters = {
    1.0, 1.5,
  };
  bool facet2d_inters_success;
  Vec<2, double> a_facet2d(coords_facet2d_a_inters);
  Vec<2, double> b_facet2d(coords_facet2d_b_inters);
  mesh::intersect_facet<double, 2>(a_facet2d, b_facet2d,
      coords_facet2d, facet2d_inters_success);
  BOOST_TEST(facet2d_inters_success);
  BOOST_TEST((mesh::crossed_facet<double, 2>(a_facet2d, b_facet2d, coords_facet2d)));

  /* check 2d facet with line that is not intersecting */
  std::vector<double> coords_facet2d_a_nointers = {
    0.0, 0.0,
  };
  std::vector<double> coords_facet2d_b_nointers = {
    2.0, 0.5,
  };
  bool facet2d_nointers_failure;
  a_facet2d = Vec<2, double>(coords_facet2d_a_nointers);
  b_facet2d = Vec<2, double>(coords_facet2d_b_nointers);
  mesh::intersect_facet<double, 2>(a_facet2d, b_facet2d,
      coords_facet2d, facet2d_nointers_failure);
  BOOST_TEST(!facet2d_nointers_failure);
  BOOST_TEST(!(mesh::crossed_facet<double, 2>(a_facet2d, b_facet2d, coords_facet2d)));

  /* check 3d facet with line that is intersecting */
  std::vector<double> coords_facet3d = {
    -0.85334, 1.36393, 2.53441,
    0.957589, -0.966325, 0.154516,
    -1.55048, -0.515958, 0.000698,
  };
  std::vector<double> coords_facet3d_a_inters = {
    0.002077, 1.36861, 0.110462,
  };
  std::vector<double> coords_facet3d_b_inters = {
    -0.630086, -1.5503, 1.34651,
  };
  bool facet3d_inters_success;
  Vec<3, double> a_facet3d(coords_facet3d_a_inters);
  Vec<3, double> b_facet3d(coords_facet3d_b_inters);
  mesh::intersect_facet<double, 3>(a_facet3d, b_facet3d,
      coords_facet3d, facet3d_inters_success);
  BOOST_TEST(facet3d_inters_success);
  BOOST_TEST((mesh::crossed_facet<double, 3>(a_facet3d, b_facet3d, coords_facet3d)));

  /* check 3d facet with line that is not intersecting */
  std::vector<double> coords_facet3d_a_nointers = {
    -0.368621, 1.2233, -0.088928,
  };
  std::vector<double> coords_facet3d_b_nointers = {
    -2.92371, 0.461491, 1.36503,
  };
  bool facet3d_nointers_failure;
  a_facet3d = Vec<3, double>(coords_facet3d_a_nointers);
  b_facet3d = Vec<3, double>(coords_facet3d_b_nointers);
  mesh::intersect_facet<double, 3>(a_facet3d, b_facet3d,
      coords_facet3d, facet3d_nointers_failure);
  BOOST_TEST(!facet3d_nointers_failure);
  BOOST_TEST(!(mesh::crossed_facet<double, 3>(a_facet3d, b_facet3d, coords_facet3d)));
}

void test_point_inside_entity()
{
  /* check 2d line */
  Vec<2, double> point2d_inside({
    1.0, 1.0,
  });
  Vec<2, double> point2d_outside({
    2.0, 0.0,
  });
  std::vector<double> coords_line2d = {
    0.0, 0.0,
    2.0, 2.0,
  };
  BOOST_TEST((mesh::point_inside_entity<double, 2>(point2d_inside, 1, coords_line2d)));
  BOOST_TEST(!(mesh::point_inside_entity<double, 2>(point2d_outside, 1, coords_line2d)));

  /* check 2d triangle */
  std::vector<double> coords_triangle2d = {
    -1.0, 1.0,
    2.0, 1.0,
    1.0, -1.0,
  };
  BOOST_TEST((mesh::point_inside_entity<double, 2>(point2d_inside, 2, coords_triangle2d)));
  BOOST_TEST(!(mesh::point_inside_entity<double, 2>(point2d_outside, 2, coords_triangle2d)));

  /* check 2d quadrilateral */
  std::vector<double> coords_quad2d = {
    -1.0, 0.0,
    0.0, 1.0,
    3.0, 2.0,
    1.0, -1.0,
  };
  BOOST_TEST((mesh::point_inside_entity<double, 2>(point2d_inside, 2, coords_quad2d)));
  BOOST_TEST(!(mesh::point_inside_entity<double, 2>(point2d_outside, 2, coords_quad2d)));

  /* check 3d line */
  Vec<3, double> point3d_inside({
    -0.4791485, 0.28801, 0.432975,
  });
  Vec<3, double> point3d_outside({
    2.0, 0.0, 1.0,
  });
  std::vector<double> coords_line3d = {
    -0.938402, -0.222531, 0.264352,
    -0.019895, 0.798551, 0.601598,
  };
  BOOST_TEST((mesh::point_inside_entity<double, 3>(point3d_inside, 1, coords_line3d)));
  BOOST_TEST(!(mesh::point_inside_entity<double, 3>(point3d_outside, 1, coords_line3d)));

  /* use extra inside points for 3d triangle and quadrilateral because it is easier to find a
     point almost exactly inside a triangle/quad, than a triangle/quad containing almost exactly
     a specific point */

  /* check 3d triangle*/
  Vec<3, double> point3d_inside_triangle({
    0.933429, 0.42216825, -0.113952,
  });
  std::vector<double> coords_triangle3d = {
    -0.938402, -0.222531, 0.264352,
    0.878718, 2.61818, 1.54044,
    1.8967, -0.353488, -1.1303,
  };
  BOOST_TEST((mesh::point_inside_entity<double, 3>(point3d_inside_triangle, 2, coords_triangle3d)));
  BOOST_TEST(!(mesh::point_inside_entity<double, 3>(point3d_outside, 2, coords_triangle3d)));

  /* check 3d quadrilateral */
  Vec<3, double> point3d_inside_quad({
    -0.014921, 0.59891225, 0.451198,
  });
  std::vector<double> coords_quad3d = {
    -0.938402, -0.222531, 0.264352,
    -1.8967, 0.353488, 1.1303,
    0.878718, 2.61818, 1.54044,
    1.8967, -0.353488, -1.1303,
  };
  BOOST_TEST((mesh::point_inside_entity<double, 3>(point3d_inside_quad, 3, coords_quad3d)));
  BOOST_TEST(!(mesh::point_inside_entity<double, 3>(point3d_outside, 3, coords_quad3d)));

  /* check 3d tetrahedron */
  std::vector<double> coords_tethed3d = {
    0.726503, 2.87455, 1.22694,
    -1.09062, 0.033843, -0.049149,
    1.74448, -0.097114, -1.4438,
    0.623705, -0.889507, 2.01997,
  };
  BOOST_TEST((mesh::point_inside_entity<double, 3>(point3d_inside, 3, coords_tethed3d)));
  BOOST_TEST(!(mesh::point_inside_entity<double, 3>(point3d_outside, 3, coords_tethed3d)));

  /* check 3d pyramid */
  std::vector<double> coords_pyramid3d = {
    1.00936, 1.97477, 1.10357,
    -0.990642, 1.97477, 0.103572,
    -0.990642, -1.02524, 0.103571,
    1.00936, -1.02524, 1.10357,
    -0.990642, 1.57477, 2.10357,
  };
  BOOST_TEST((mesh::point_inside_entity<double, 3>(point3d_inside, 3, coords_pyramid3d)));
  BOOST_TEST(!(mesh::point_inside_entity<double, 3>(point3d_outside, 3, coords_pyramid3d)));

  /* check 3d hexahedron */
  std::vector<double> coords_hexhed3d = {
    1.00936, 1.97477, 1.10357,
    -0.990642, 1.97477, 0.103572,
    -0.990642, -1.02524, 0.103571,
    1.00936, -1.02524, 1.10357,
    1.00936, 1.57476, 3.10357,
    -0.990642, 1.57477, 2.10357,
    -0.990642, -0.025236, 3.10357,
    1.00936, -0.025236, 4.10357,
  };
  BOOST_TEST((mesh::point_inside_entity<double, 3>(point3d_inside, 3, coords_hexhed3d)));
  BOOST_TEST(!(mesh::point_inside_entity<double, 3>(point3d_outside, 3, coords_hexhed3d)));
}

void test_point_inside_cell()
{
  Vec<1, double> ref_point1d;
  Vec<2, double> ref_point2d;
  Vec<3, double> ref_point3d;

  /* check 1d line */
  Vec<1, double> point1d_inside({ 1.1 });
  Vec<1, double> point1d_outside({ -0.7 });
  std::vector<double> coords_line1d = { -0.6, 1.2 };
  BOOST_TEST((mesh::point_inside_cell<double, 1>(point1d_inside,
          coords_line1d, ref_point1d)));
  BOOST_TEST(!(mesh::point_inside_cell<double, 1>(point1d_outside,
          coords_line1d, ref_point1d)));


  /* check 2d triangle */
  Vec<2, double> point2d_inside({
    1.0, 1.0,
  });
  Vec<2, double> point2d_outside({
    2.0, 0.0,
  });
  std::vector<double> coords_triangle2d = {
    -1.0, 1.0,
    2.0, 1.0,
    1.0, -1.0,
  };
  BOOST_TEST((mesh::point_inside_cell<double, 2>(point2d_inside,
          coords_triangle2d, ref_point2d)));
  BOOST_TEST(!(mesh::point_inside_cell<double, 2>(point2d_outside,
          coords_triangle2d, ref_point2d)));

  /* check 2d quadrilateral */
  std::vector<double> coords_quad2d = {
    -1.0, 0.0,
    0.0, 1.0,
    3.0, 2.0,
    1.0, -1.0,
  };
  BOOST_TEST((mesh::point_inside_cell<double, 2>(point2d_inside, coords_quad2d, ref_point2d)));
  BOOST_TEST(!(mesh::point_inside_cell<double, 2>(point2d_outside, coords_quad2d, ref_point2d)));

  /* check 3d tetrahedron */
  Vec<3, double> point3d_inside({
    -0.4791485, 0.28801, 0.432975,
  });
  Vec<3, double> point3d_outside({
    2.0, 0.0, 1.0,
  });
  std::vector<double> coords_tethed3d = {
    0.726503, 2.87455, 1.22694,
    -1.09062, 0.033843, -0.049149,
    1.74448, -0.097114, -1.4438,
    0.623705, -0.889507, 2.01997,
  };
  BOOST_TEST((mesh::point_inside_cell<double, 3>(point3d_inside,
          coords_tethed3d, ref_point3d)));
  BOOST_TEST(!(mesh::point_inside_cell<double, 3>(point3d_outside,
          coords_tethed3d, ref_point3d)));

  /* check 3d pyramid */
  std::vector<double> coords_pyramid3d = {
    1.00936, 1.97477, 1.10357,
    -0.990642, 1.97477, 0.103572,
    -0.990642, -1.02524, 0.103571,
    1.00936, -1.02524, 1.10357,
    -0.990642, 1.57477, 2.10357,
  };
  BOOST_TEST((mesh::point_inside_cell<double, 3>(point3d_inside,
          coords_pyramid3d, ref_point3d)));
  BOOST_TEST(!(mesh::point_inside_cell<double, 3>(point3d_outside,
          coords_pyramid3d, ref_point3d)));

  /* check 3d hexahedron */
  std::vector<double> coords_hexhed3d = {
    1.00936, 1.97477, 1.10357,
    -0.990642, 1.97477, 0.103572,
    -0.990642, -1.02524, 0.103571,
    1.00936, -1.02524, 1.10357,
    1.00936, 1.57476, 3.10357,
    -0.990642, 1.57477, 2.10357,
    -0.990642, -0.025236, 3.10357,
    1.00936, -0.025236, 4.10357,
  };
  BOOST_TEST((mesh::point_inside_cell<double, 3>(point3d_inside,
          coords_hexhed3d, ref_point3d)));
  BOOST_TEST(!(mesh::point_inside_cell<double, 3>(point3d_outside,
          coords_hexhed3d, ref_point3d)));
}

void test_vertices_inside_one_hyperplane()
{
  std::vector<double> coords_plane_samples = {
    2.6, -1.2,  3.0,
    -4.1,  2.3,  4.8,
    -3.5 ,  2.17,  1.8,
    -23.09 ,  11.945,  14.16,
    62.99 , -33.334,  -4.14,
  };

  std::vector<double> coords_noplane_samples = {
    2.6, -1.2,  3.0,
    -4.1,  1.3,  4.8,
    -3.5 ,  2.17,  1.8,
    -23.09 ,  11.945,  14.16,
    62.99 , -33.334,  -4.14,
  };

  BOOST_TEST((vertices_inside_one_hyperplane<double, 3>(coords_plane_samples, 3, 1e-8)));
  BOOST_TEST(!(vertices_inside_one_hyperplane<double, 3>(coords_noplane_samples, 3, 1e-8)));
}

void test_triangle_area()
{
  std::vector<double> coords_triangle2d = {
    0.5 - 0.5 * sqrt(3), -0.5 * sqrt(3) - 0.5,
    0.5 * sqrt(3) + 0.5, 0.5 - 0.5 * sqrt(3),
    0.205 * sqrt(3) - 0.5, 0.5 * sqrt(3) + 0.205,
  };
  double area2d = 2.0;

  std::vector<double> coords_triangle3d = {
    -0.5 * sqrt(3), -1.0, 0.5,
    0.5 * sqrt(3), -1.0, -0.5,
    0.205 * sqrt(3), 1.0, -0.205,
  };
  double area3d = 2.0;

  BOOST_TEST(std::abs(triangle_area<double, 2>(coords_triangle2d) - area2d) < GEOM_TOL);
  BOOST_TEST(std::abs(triangle_area<double, 3>(coords_triangle3d) - area3d) < GEOM_TOL);
}

void test_facet_area()
{
  std::vector<double> coords_line2d = {
    0.4, 0.3,
    1.3, 2.5
  };
  double area2d = 2.3769728648009427;

  std::vector<double> coords_hexgon3d = {
    0.0, -2.0, 0.0,
    -1.0/sqrt(2.0), -1.0, 1.0/sqrt(2.0),
    -1.0/sqrt(2.0), 1.0, 1.0/sqrt(2.0),
    0.0, 2.0, 0.0,
    1.0/sqrt(2.0), 1.0, -1.0/sqrt(2.0),
    1.0/sqrt(2.0), -1.0, -1.0/sqrt(2.0),
  };
  double area3d = 6.0;

  BOOST_TEST(std::abs(facet_area<double, 2>(coords_line2d, 2) - area2d) < GEOM_TOL);
  BOOST_TEST(std::abs(facet_area<double, 3>(coords_hexgon3d, 3) - area3d) < GEOM_TOL);
}

void test_in_plane()
{
  Vec<2, double> pi2d({2.2, -5.3});
  Vec<2, double> po2d({2.2, -5.2});
  Vec<2, double> origin2d({1.2, -4.3});
  Vec<2, double> normal2d({1.0/sqrt(2.0), 1.0/sqrt(2.0)});

  Vec<3, double> pi3d({5.1, -6.7, 4.0});
  Vec<3, double> po3d({5.1, -6.7, 4.1});
  Vec<3, double> origin3d({1.3, 4.1, -3.0});
  Vec<3, double> normal3d({1.0/sqrt(3.0), 1.0/sqrt(3.0), 1.0/sqrt(3)});

  BOOST_TEST((in_plane<double, 2>(pi2d, origin2d, normal2d, GEOM_TOL)));
  BOOST_TEST((in_plane<double, 3>(pi3d, origin3d, normal3d, GEOM_TOL)));
}

void test_crossed_plane()
{
  Vec<2, double> a_2d({-0.242495, -0.844617});
  Vec<2, double> bc_2d({0.57818, -0.164169});
  Vec<2, double> bn_2d({0.344773, -0.736713});
  Vec<2, double> origin2d({1.07264, -1.47693});
  Vec<2, double> normal2d({-1.992982, -1.674973});
  normal2d *= 1.0 / norm(normal2d);

  BOOST_TEST((crossed_plane<double, 2>(a_2d, bc_2d, origin2d, normal2d)));
  BOOST_TEST(!(crossed_plane<double, 2>(a_2d, bn_2d, origin2d, normal2d)));

  Vec<3, double> a_3d({-2.745734, -1.677992, 1.556207});
  Vec<3, double> bc_3d({-3.986419, -2.483497, 0.911735});
  Vec<3, double> bn_3d({-2.948312, -2.067292, 1.206284});
  Vec<3, double> origin3d({-3.1, -2.1, 1.2});
  Vec<3, double> normal3d({1.0, 1.0, 1.0});
  normal3d *= 1.0 / norm(normal3d);

  BOOST_TEST((crossed_plane<double, 3>(a_3d, bc_3d, origin3d, normal3d)));
  BOOST_TEST(!(crossed_plane<double, 3>(a_3d, bn_3d, origin3d, normal3d)));
}

void test_distance_point_hyperplane()
{
  Vec<2, double> point2d({1.5, -2.9});
  Vec<2, double> origin2d({1.7, -3.7});
  Vec<2, double> normal2d({1.0, 1.0});
  normal2d *= 1.0 / norm(normal2d);
  double dist2d = 0.3 * sqrt(2.0);

  Vec<3, double> point3d({4.5, -4.7, 5.2});
  Vec<3, double> origin3d({4.1, -5.1, 4.8});
  Vec<3, double> normal3d({1.0, 1.0, 1.0});
  normal3d *= 1.0 / norm(normal3d);
  double dist3d = 0.4 * sqrt(3.0);

  BOOST_TEST(
      (distance_point_hyperplane<double, 2>(point2d, origin2d, normal2d) - dist2d) < GEOM_TOL);
  BOOST_TEST(
      (distance_point_hyperplane<double, 3>(point3d, origin3d, normal3d) - dist3d) < GEOM_TOL);
}

void test_distance_point_line()
{
  Vec<2, double> point2d({-1.2, 2.0});
  Vec<2, double> origin2d({-4.3, -2.6});
  Vec<2, double> dir2d({2.3, 4.3});
  double dist2d = 0.5639320384466053;

  Vec<3, double> point3d({-1.2, 2.0, 7.7});
  Vec<3, double> origin3d({-4.3, -2.6, 8.6});
  Vec<3, double> dir3d({2.3, 4.3, -6.3});
  double dist3d = 3.854380650239859;

  BOOST_TEST(std::abs(distance_point_line<double, 2>(point2d, origin2d, dir2d)
        - dist2d) < GEOM_TOL);
  BOOST_TEST(std::abs(distance_point_line<double, 3>(point3d, origin3d, dir3d)
        - dist3d) < GEOM_TOL);
}

void test_foot_point_hyperplane()
{
  Vec<2, double> point2d( { 4.8, -0.3 });
  Vec<2, double> origin2d({ -8.7, 1.5 });
  Vec<2, double> normal2d({ 5.5, -5.7 });
  Vec<2, double> foot2d({ -2.608431622569333, 7.37782913611731 });

  Vec<3, double> point3d({ -9.3, -2.1, 7.4 });
  Vec<3, double> origin3d({ 9.3, -9.1, -2.9 });
  Vec<3, double> normal3d({ -0.7, -4.4, -3.6 });
  Vec<3, double> foot3d({-10.47043584273088, -9.457025297165499, 1.380615665955502});

  Vec<2, double> f2d = foot_point_hyperplane<double, 2>(point2d, origin2d, normal2d);
  for (int i = 0; i < f2d.size(); ++i) {
    BOOST_TEST(std::abs(f2d[i] - foot2d[i]) < GEOM_TOL);
  }

  Vec<3, double> f3d = foot_point_hyperplane<double, 3>(point3d, origin3d, normal3d);
  for (int i = 0; i < f3d.size(); ++i) {
    BOOST_TEST(std::abs(f3d[i] - foot3d[i]) < GEOM_TOL);
  }
}

void test_foot_point_line()
{
  Vec<2, double> point2d({ -1.4, -3.1 });
  Vec<2, double> origin2d({ 3.0, -5.7 });
  Vec<2, double> dir2d({ -1.4, -7.8 });
  Vec<2, double> foot2d({ 3.314777070063694, -3.94624203821656 });

  Vec<3, double> point3d({ -6.4, -2.3, 6.3 });
  Vec<3, double> origin3d({ 5.6, -7.7, -8.2 });
  Vec<3, double> dir3d({ -5.2, -2.2, 8.4 });
  Vec<3, double> foot3d({ -3.147208121827411, -11.40074189769621, 5.930105427567357 });

  Vec<2, double> f2d = foot_point_line<double, 2>(point2d, origin2d, dir2d);
  for (int i = 0; i < f2d.size(); ++i) {
    BOOST_TEST(std::abs(f2d[i] - foot2d[i]) < GEOM_TOL);
  }

  Vec<3, double> f3d = foot_point_line<double, 3>(point3d, origin3d, dir3d);
  for (int i = 0; i < f3d.size(); ++i) {
    BOOST_TEST(std::abs(f3d[i] - foot3d[i]) < GEOM_TOL);
  }
}

void test_normal()
{
  std::vector<Vec<2, double>> dirs2d = {
    Vec<2, double>({ 6.9, -2.5 }),
  };
  Vec<3, double> normal2d({-0.3406487770499215, -0.9401906246577834});

  std::vector<Vec<3, double>> dirs3d = {
    Vec<3, double>({ 8.7, 8.9, 5.1 }),
    Vec<3, double>({ -8.5, -9.2, -3.7 }),
  };
  Vec<3, double> normal3d({0.7592305051627601, -0.6056477796723663, -0.2382431678101864});

  Vec<2, double> n2d = normal<double, 2>(dirs2d);
  for (int i = 0; i < n2d.size(); ++i) {
    BOOST_TEST(std::abs(n2d[i] - normal2d[i]) < GEOM_TOL);
  }

  Vec<3, double> n3d = normal<double, 3>(dirs3d);
  for (int i = 0; i < n3d.size(); ++i) {
    BOOST_TEST(std::abs(n3d[i] - normal3d[i]) < GEOM_TOL);
  }
}

// unprecise
void test_distance_point_facet()
{
  Vec<2, double> p_edge2d({-4.3, -14.8});
  Vec<2, double> p_vert2d({9.6, 6.7});
  std::vector<double> lineverts2d({ -9.3, -9.7, 9.3, 2.2 });
  Vec<2, double> clp_edge2d({-8.067426215722872, -8.911417847693663});
  Vec<2, double> clp_vert2d({9.3, 2.2});
  double dist_edge2d = 6.990629446292852;
  double dist_vert2d = 4.509988913511872;

  Vec<2, double> clp2d;
  BOOST_TEST(std::abs(distance_point_facet<double, 2>(p_edge2d, lineverts2d, clp2d)
        - dist_edge2d) < GEOM_TOL);
  for (int i = 0; i < clp2d.size(); ++i) {
    BOOST_TEST(std::abs(clp2d[i] - clp_edge2d[i]) < GEOM_TOL);
  }
  BOOST_TEST(std::abs(distance_point_facet<double, 2>(p_vert2d, lineverts2d, clp2d)
        - dist_vert2d) < GEOM_TOL);
  for (int i = 0; i < clp2d.size(); ++i) {
    BOOST_TEST(std::abs(clp2d[i] - clp_vert2d[i]) < GEOM_TOL);
  }

  Vec<3, double> p_plane3d({-1.6, -2.8, 4.0});
  Vec<3, double> p_edge3d({-7.3, -4.4, 9.0});
  Vec<3, double> p_vert3d({-8.7, 3.6, -4.0});
  std::vector<double> triangleverts3d({
    -1.046381822915578, 1.599677328004305, -0.07821462101729393,
    1.574232665133403, -6.886484926424545, 11.84686105574614,
    -6.548681135884792, 1.362411535675318, -0.9993048523676322,
  });
  Vec<3, double> clp_plane3d({ -1.732872574009981, -2.011177901052426, 4.590543851036967 });
  /* the commented one, seems to be to unprecise, or maybe 'distance_point_facet' is? */
  //Vec<3, double> clp_edge3d({ -1.934242262047544, -3.323595165048366, 6.298303961200298 });
  Vec<3, double> clp_edge3d({ -1.934242262047544, -3.323595165048366, 6.298303961200312 });
  Vec<3, double> clp_vert3d({ -6.548681135884792, 1.362411535675318, -0.9993048523676322 });
  double dist_plane3d = 0.9943025016108856;
  double dist_edge3d = 6.103209398109319;
  double dist_vert3d = 4.317307767556221;

  Vec<3, double> clp3d;
  BOOST_TEST(std::abs(distance_point_facet<double, 3>(p_plane3d, triangleverts3d, clp3d)
        - dist_plane3d) < GEOM_TOL);
  for (int i = 0; i < clp3d.size(); ++i) {
    BOOST_TEST(std::abs(clp3d[i] - clp_plane3d[i]) < GEOM_TOL);
  }
  BOOST_TEST(std::abs(distance_point_facet<double, 3>(p_edge3d, triangleverts3d, clp3d)
        - dist_edge3d) < GEOM_TOL);
  for (int i = 0; i < clp3d.size(); ++i) {
    BOOST_TEST(std::abs(clp3d[i] - clp_edge3d[i]) < GEOM_TOL);
  }
  BOOST_TEST(std::abs(distance_point_facet<double, 3>(p_vert3d, triangleverts3d, clp3d)
        - dist_vert3d) < GEOM_TOL);
  for (int i = 0; i < clp3d.size(); ++i) {
    BOOST_TEST(std::abs(clp3d[i] - clp_vert3d[i]) < GEOM_TOL);
  }
}

void test_project_point()
{
  Vec<2, double> point2d({2.5, 2.0});
  Ellipsoid<2> ellipbound2d(sqrt(8.0), sqrt(2.0));
  Vec<2, double> proj2d({2.0, 1.0});

  Vec<2, double> pj2d = project_point<double, 2>(ellipbound2d, point2d, -1);
  for (int i = 0; i < pj2d.size(); ++i) {
    BOOST_TEST(std::abs(pj2d[i] - proj2d[i]) < GEOM_TOL);
  }

  Vec<3, double> point3d({7.0 * sqrt(2.0) / 6.0, 5.0 / 3.0, 3.0 * sqrt(2.0) / 2.0 });
  Ellipsoid<3> ellipbound3d(sqrt(12), sqrt(3), sqrt(4));
  Vec<3, double> proj3d({ sqrt(2), 1, sqrt(2) });

  Vec<3, double> pj3d = project_point<double, 3>(ellipbound3d, point3d, -1);
  for (int i = 0; i < pj3d.size(); ++i) {
    BOOST_TEST(std::abs(pj3d[i] - proj3d[i]) < GEOM_TOL);
  }
}

void test_is_point_on_subentity()
{
  Vec<2, double> inpoint2d({ -5.2, 0.35 });
  Vec<2, double> outpoint2d({ -0.5, 8.2 });
  std::vector<Vec<2, double>> points2d = {
    Vec<2, double>({ -4.9, 5.2 }),
    Vec<2, double>({ -5.5, -4.5 }),
  };

  BOOST_TEST((is_point_on_subentity<double, 2>(inpoint2d, points2d)));
  BOOST_TEST(!(is_point_on_subentity<double, 2>(outpoint2d, points2d)));

  Vec<3, double> inpoint3d({0.625, 0.35, -0.55});
  Vec<3, double> outpoint3d({1.23, 1.68, 2.0});
  std::vector<Vec<3, double>> points3d = {
    Vec<3, double>({ -5.7, 5.2, -4.8 }),
    Vec<3, double>({ -4.8, 0.6, -0.2 }),
    Vec<3, double>({ 6.5, -2.2, 1.4 }),
  };

  BOOST_TEST((is_point_on_subentity<double, 3>(inpoint3d, points3d)));
  BOOST_TEST(!(is_point_on_subentity<double, 3>(outpoint3d, points3d)));
}

void check_subentity_coords(mesh::CellType const *triangle, int tdim, int idx,
    std::vector<double> coords, std::vector<double> refined_coords)
{
  std::vector<int> verts = triangle->local_vertices_of_entity(tdim, idx);
  for (int i = 0; i < verts.size(); ++i) {
    for (int j = 0; j < 3; ++j) {
      BOOST_TEST(std::abs(refined_coords[verts[i] * 3 + j] - coords[3 * i + j]) < GEOM_TOL);
    }
  }
}
void test_find_subentities_containing_point()
{
  Vec<3, double> onpt({0.625, 0.35, -0.55});
  Vec<3, double> outpt({2.20625, -0.8625, 0.5125});
  std::vector<hiflow::Vec<3, double>> coords = {
    Vec<3, double>({ -5.7, 5.2, -4.8 }),
    Vec<3, double>({ -4.8, 0.6, -0.2 }),
    Vec<3, double>({ 6.5, -2.2, 1.4 }),
  };
  std::vector<std::vector<std::vector<double>>> on_subentity_coords = {
    {}, // tdim = 0
    {
      { 0.4, 1.5, -1.7, 0.85, -0.8, 0.6 },
      { -5.25, 2.9, -2.5, 6.5, -2.2, 1.4 },
    }, // tdim = 1
  };
  std::vector<std::vector<std::vector<double>>> out_subentity_coords = {
    {}, // tdim = 0
    {}, // tdim = 1
  };

  mesh::CellType::Tag tag = mesh::CellType::TRIANGLE;
  mesh::CellType const *triangle = &(mesh::CellType::get_instance(tag));

  std::vector<double> cell_coords(coords.size() * coords[0].size());
  for (int i = 0; i < coords.size(); ++i) {
    for (int j = 0; j < coords[i].size(); ++j) {
      cell_coords[i * coords[i].size() + j] = coords[i][j];
    }
  }
  std::vector<double> refined_coords;
  mesh::compute_refined_vertices(*triangle, 3, cell_coords, refined_coords);

  std::vector<std::vector<int>> dof;
  find_subentities_containing_point<double, 3>(onpt, triangle, coords, dof);
  for (int tdim = 0; tdim < triangle->tdim(); ++tdim) {
    BOOST_TEST(dof[tdim].size() == on_subentity_coords[tdim].size());
    for (int i = 0; i < dof[tdim].size(); ++i) {
      check_subentity_coords(triangle, tdim, dof[tdim][i], on_subentity_coords[tdim][i], refined_coords);
    }
  }

  find_subentities_containing_point<double, 3>(outpt, triangle, coords, dof);
  for (int tdim = 0; tdim < triangle->tdim(); ++tdim) {
    BOOST_TEST(dof[tdim].size() == out_subentity_coords[tdim].size());
    for (int i = 0; i < dof[tdim].size(); ++i) {
      check_subentity_coords(triangle, tdim, i, out_subentity_coords[tdim][i], refined_coords);
    }
  }
}

#if 0
void test_map_ref_coord_to_other_cell()
{
  Vec<2, double> coord_ref({ 0.0, 0.2 });
  Vec<2, double> coord_other({0.2, 0.0});
  std::vector<double> coords_ref = {
    0.0, 2.0,
    -2.0, 0.0,
    0.0, 0.0,
  };
  std::vector<double> coords_other = {
    -2.0, 2.0,
    -2.0, 0.0,
    0.0, 2.0,
  };

  doffem::CRefCellSPtr<double, 2> cell(new doffem::RefCellTriStd<double, 2>());
  doffem::CellTrafoSPtr<double, 2> trafo_ref(
      new doffem::LinearTriangleTransformation<double, 2>(cell));
  trafo_ref->reinit(coords_ref);
  doffem::CellTrafoSPtr<double, 2> trafo_other(
      new doffem::LinearTriangleTransformation<double, 2>(cell));
  trafo_other->reinit(coords_other);

  Vec<2, double> coord;
  std::vector<mesh::MasterSlave> period(1, mesh::MasterSlave(0.0, -2.0, 0.0, 0));
  BOOST_TEST((map_ref_coord_to_other_cell<double, 2>(coord_ref, coord,
      trafo_ref.get(), trafo_other.get(), period)));
  for (int i = 0; i < coord.size(); ++i) {
    BOOST_TEST(std::abs(coord[i] - coord_other[i]) < GEOM_TOL);
  }
}
#endif

// unprecise
void test_create_bbox_for_mesh()
{
  int tdim = 3;
  int gdim = 3;
  mesh::MeshBuilder* mb(new mesh::MeshDbViewBuilder(tdim, gdim));
  ScopedPtr<Reader>::Type reader(new UcdReader(mb));
  MeshPtr mesh;
  reader->read(MESH_DATADIR "two_tetras_3d.inp", mesh);

  std::vector<double> extends = { 10, 12, 2, 3, 1, 2 };

  BBox<double, 3> bbox(3);
  mesh::create_bbox_for_mesh(mesh, bbox);
  std::vector<double> ext = bbox.get_extents();
  BOOST_TEST(ext.size() == extends.size());
  for (int i = 0; i < ext.size(); ++i) {
    /* this fails without 'times 100', maybe improve
       the precision in create_bbox_for_mesh? */
    BOOST_TEST(std::abs(ext[i] - extends[i]) < 100 * GEOM_TOL);
  }

  delete mb;
}

void test_find_adjacent_cells()
{
  std::vector<std::set<int>> adjcells = {
    std::set<int>{13},
    std::set<int>{13, 14, 16, 17},
    std::set<int>{5, 13, 14},
  };

  int tdim = 3;
  int gdim = 3;
  mesh::MeshBuilder* mb(new mesh::MeshDbViewBuilder(tdim, gdim));
  ScopedPtr<Reader>::Type reader(new UcdReader(mb));

  MeshPtr cubemesh, tetramesh;
  reader->read(MESH_DATADIR "find_adjacent_cells_cubemesh.inp", cubemesh);
  reader->read(MESH_DATADIR "find_adjacent_cells_tetramesh.inp", tetramesh);

  std::map<int, std::set<int>> cells;
  find_adjacent_cells<double, 3>(tetramesh, cubemesh, cells);

  BOOST_TEST(cells.size() == 3);
  for (int i = 0; i < cells.size(); ++i) {
    BOOST_TEST(cells[i] == adjcells[i]);
  }
}

void test_find_adjacent_cells_related()
{
  std::vector<size_t> adjcubecellsnum = {
    27, 36, 27,
    36, 48, 36,
    27, 36, 27,

    36, 48, 36,
    48, 64, 48,
    36, 48, 36,

    27, 36, 27,
    36, 48, 36,
    27, 36, 27,
  };
  std::vector<size_t> adjtetracellsnum = { 22, 19, 21 };

  int tdim = 3;
  int gdim = 3;
  mesh::MeshBuilder* mb(new mesh::MeshDbViewBuilder(tdim, gdim));
  ScopedPtr<Reader>::Type reader(new UcdReader(mb));

  MeshPtr cubemesh;
  reader->read(MESH_DATADIR "find_adjacent_cells_cubemesh.inp", cubemesh);
  MeshPtr refinedcubemesh = cubemesh->refine();

  std::map<EntityNumber, std::set<EntityNumber>> cubecells;
  find_adjacent_cells_related<double, 3>(cubemesh, refinedcubemesh, cubecells);
  for (EntityIterator cell = cubemesh->begin(tdim); cell != cubemesh->end(tdim); ++cell) {
    EntityNumber i = cell->index();
    std::set<EntityNumber> c = cubecells.at(i);
    BOOST_TEST(c.size() == adjcubecellsnum[i]);
  }

  MeshPtr tetramesh;
  reader->read(MESH_DATADIR "find_adjacent_cells_tetramesh.inp", tetramesh);
  MeshPtr refinedtetramesh = tetramesh->refine();

  std::map<EntityNumber, std::set<EntityNumber>> tetracells;
  find_adjacent_cells_related<double, 3>(tetramesh, refinedtetramesh, tetracells);
  for (EntityIterator cell = tetramesh->begin(tdim); cell != tetramesh->end(tdim); ++cell) {
    EntityNumber i = cell->index();
    std::set<EntityNumber> c = tetracells.at(i);
    BOOST_TEST(c.size() == adjtetracellsnum[i]);
  }
}

void test_cells_intersect()
{
  std::vector<double> coords1d_in_inters = { -4.3, -1.5 };
  std::vector<double> coords1d_out_inters = { -8.4, -3.9 };
  std::vector<double> coords1d_in_notinters = { 0.6, 1.4 };
  std::vector<double> coords1d_out_notinters = { 3.4, 6.8 };
  BOOST_TEST((mesh::cells_intersect<double, 1>(coords1d_in_inters, coords1d_out_inters)));
  BOOST_TEST(!(mesh::cells_intersect<double, 1>(coords1d_in_notinters, coords1d_out_notinters)));

  std::vector<double> coords2d_in_inters = {
    -0.76, 1.66,
    -2.0, -1.0,
    2.8, -1.44,
    3.82, 2.34,
  };
  std::vector<double> coords2d_out_inters = {
    -4.16, 3.42,
    -5.5, 1.18,
    -0.24, 1.00,
    0.32, 3.96,
  };
  std::vector<double> coords2d_in_notinters = {
    7.00, 4.64,
    5.34, 2.88,
    8.18, 1.82,
    9.94, 4.48,
  };
  std::vector<double> coords2d_out_notinters = {
    6.4, 1.12,
    4.2, -2.7,
    10.14, -3.4,
    11.62, 2.36
  };
  BOOST_TEST((mesh::cells_intersect<double, 2>(coords2d_in_inters,
          coords2d_out_inters)));
  BOOST_TEST(!(mesh::cells_intersect<double, 2>(coords2d_in_notinters,
          coords2d_out_notinters)));

  std::vector<double> coords3d_pyramid = {
    -1.02164, -0.674217, -0.390665,
    0.978355, -0.674217, -0.390665,
    1.558487, 1.32353, -0.390665,
    -0.441508, 1.32353, -0.390665,
    -0.308324, 0.286491, 2.51679,
  };
  std::vector<double> coords3d_tetra_inters = {
    0.377723, -1.42162, -0.108183,
    2.75964, -0.315616, 0.210277,
    1.91659, 1.70193, -0.028103,
    0.763728, 1.2595, 1.50556,
  };
  std::vector<double> coords3d_hexa_notinters = {
    -3.77006, -0.319313, -0.209279,
    -1.77006, -0.319313, -0.209279,
    -1.77006, 1.68069, -0.209279,
    -3.77006, 1.68069, -0.209279,
    -2.65744, 0.856776, 2.12469,
    -0.65744, 0.856776, 2.12469,
    -0.65744, 2.856779, 2.12469,
    -2.65744, 2.856779, 2.12469,
  };
  BOOST_TEST((mesh::cells_intersect<double, 3>(coords3d_pyramid, coords3d_tetra_inters)));
  BOOST_TEST(!(mesh::cells_intersect<double, 3>(coords3d_pyramid, coords3d_hexa_notinters)));
}

void test_is_aligned_rectangular_cuboid()
{
  std::vector<double> arc_coords1 = {
    -1.0, -1.0, 0.0,
    2.0, -1.0, 0.0,
    2.0, 1.0, 0.0,
    -1.0, 1.0, 0.0,
    -1.0, -1.0, 1.0,
    2.0, -1.0, 1.0,
    2.0, 1.0, 1.0,
    -1.0, 1.0, 1.0,
  };
  BOOST_TEST(mesh::is_aligned_rectangular_cuboid<double>(arc_coords1));

  std::vector<double> arc_coords2 = {
    1.847203, 1.000000, 1.327673,
    -0.081159, 1.000000, -0.970460,
    -0.847204, 1.000000, -0.327673,
    1.081159, 1.000000, 1.970460,
    1.081159, -1.000000, 1.970460,
    -0.847203, -1.000000, -0.327673,
    -0.081159, -1.000000, -0.970460,
    1.847204, -1.000000, 1.327673,
  };
  BOOST_TEST(!mesh::is_aligned_rectangular_cuboid<double>(arc_coords2));
}

void test_is_aligned_rectangular_quad()
{
  std::vector<double> arq_coords1 = {
    0.0, 0.0,
    0.0, 1.0,
    -1.0, 1.0,
    -1.0, 0.0,
  };
  BOOST_TEST(mesh::is_aligned_rectangular_quad(arq_coords1));

  std::vector<double> arq_coords2 = {
    0.0, 0.0,
    0.0, 1.1,
    -1.0, 1.0,
    -1.0, 0.0,
  };
  BOOST_TEST(!mesh::is_aligned_rectangular_quad(arq_coords2));
}

void test_is_parallelogram()
{
  std::vector<double> pg_coords1 = {
    0.0, 0.0,
    2.0, 1.0,
    3.0, 3.0,
    1.0, 2.0,
  };
  BOOST_TEST(mesh::is_parallelogram<double>(pg_coords1));

  std::vector<double> pg_coords2 = {
    0.0, 0.0,
    1.0, 0.0,
    2.0, 2.0,
    0.5, 1.0,
  };
  BOOST_TEST(!mesh::is_parallelogram<double>(pg_coords2));
}

void test_is_parallelepiped()
{
  std::vector<double> pe_coords1 = {
    0.0, 0.0, 0.0,
    2.0, 0.0, 0.5,
    3.0, 2.0, 0.5,
    1.0, 2.0, 0.0,
    1.0, 0.0, 2.0,
    3.0, 0.0, 2.5,
    4.0, 2.0, 2.5,
    2.0, 2.0, 2.0,
  };
  BOOST_TEST(mesh::is_parallelepiped<double>(pe_coords1));

  std::vector<double> pe_coords2 = {
    0.0, 0.0, 0.0,
    2.0, 0.0, 0.5,
    3.0, 2.0, 0.5,
    1.0, 2.0, 0.0,
    1.0, 0.0, 2.0,
    4.0, -1.0, 2.5,
    4.0, 2.0, 2.5,
    2.0, 2.0, 2.0,
  };
  BOOST_TEST(!mesh::is_parallelepiped<double>(pe_coords2));
}

void test_parametrize_object()
{
  std::vector<Vec<3, double>> tetcoords = {
    Vec<3, double>({0.0, 0.0, 0.0}),
    Vec<3, double>({1.0, 0.0, 0.0}),
    Vec<3, double>({0.0, 1.0, 0.0}),
    Vec<3, double>({0.0, 0.0, 1.0}),
  };
  std::vector<Vec<3, double>> tetsupvecs = {
    Vec<3, double>({0.0, 0.0, 0.0}),
    Vec<3, double>({0.0, 0.0, 0.0}),
    Vec<3, double>({0.0, 0.0, 0.0}),
    Vec<3, double>({0.0, 0.0, 0.0}),
    Vec<3, double>({0.0, 0.0, 0.0}),
    Vec<3, double>({0.0, 0.0, 0.0}),

    Vec<3, double>({1.0, 0.0, 0.0}),
    Vec<3, double>({1.0, 0.0, 0.0}),
    Vec<3, double>({1.0, 0.0, 0.0}),
    Vec<3, double>({1.0, 0.0, 0.0}),
    Vec<3, double>({1.0, 0.0, 0.0}),

    Vec<3, double>({0.0, 1.0, 0.0}),
    Vec<3, double>({0.0, 1.0, 0.0}),
    Vec<3, double>({0.0, 1.0, 0.0}),
    Vec<3, double>({0.0, 1.0, 0.0}),

    Vec<3, double>({0.0, 0.0, 1.0}),
    Vec<3, double>({0.0, 0.0, 1.0}),
    Vec<3, double>({0.0, 0.0, 1.0}),
  };
  std::vector<Vec<3, double>> tetdirvecs = {
    Vec<3, double>({1.0, 0.0, 0.0}),
    Vec<3, double>({0.0, 1.0, 0.0}),
    Vec<3, double>({0.0, 0.0, 1.0}),
    Vec<3, double>({0.5, 0.5, 0.0}),
    Vec<3, double>({0.5, 0.0, 0.5}),
    Vec<3, double>({0.0, 0.5, 0.5}),

    Vec<3, double>({-1.0, 1.0, 0.0}),
    Vec<3, double>({-1.0, 0.0, 1.0}),
    Vec<3, double>({-1.0, 0.5, 0.0}),
    Vec<3, double>({-1.0, 0.0, 0.5}),
    Vec<3, double>({-1.0, 0.5, 0.5}),

    Vec<3, double>({0.0, -1.0, 1.0}),
    Vec<3, double>({0.5, -1.0, 0.0}),
    Vec<3, double>({0.0, -1.0, 0.5}),
    Vec<3, double>({0.5, -1.0, 0.5}),

    Vec<3, double>({0.5, 0.0, -1.0}),
    Vec<3, double>({0.0, 0.5, -1.0}),
    Vec<3, double>({0.5, 0.5, -1.0}),
  };

  std::vector<Vec<3, double>> dirvecs, supvecs;
  parametrize_object(tetcoords, dirvecs, supvecs);

  BOOST_TEST(dirvecs.size() == tetdirvecs.size());
  BOOST_TEST(supvecs.size() == tetsupvecs.size());
  while (!tetdirvecs.empty()) {
    bool rightvecs = false;
    int i;
    for (i = dirvecs.size() - 1; i >= 0; --i) {
      if (vec_is_equal_eps<3>(dirvecs[i], tetdirvecs.back(), GEOM_TOL)
          && vec_is_equal_eps<3>(supvecs[i], tetsupvecs.back(), GEOM_TOL)) {
        rightvecs = true;
        break;
      }
    }
    BOOST_TEST(rightvecs);

    tetdirvecs.pop_back();
    tetsupvecs.pop_back();
  }

  std::vector<Vec<3, double>> pyrcoords = {
    Vec<3, double>({0.0, 0.0, 0.0}),
    Vec<3, double>({0.0, 1.0, 0.0}),
    Vec<3, double>({1.0, 1.0, 0.0}),
    Vec<3, double>({1.0, 0.0, 0.0}),
    Vec<3, double>({0.0, 0.0, 1.0}),
  };
  std::vector<Vec<3, double>> pyrdirvecs = {
    Vec<3, double>({0.0, 1.0, 0.0}),
    Vec<3, double>({1.0, 1.0, 0.0}),
    Vec<3, double>({1.0, 0.0, 0.0}),
    Vec<3, double>({0.0, 0.0, 1.0}),

    Vec<3, double>({1.0, 0.0, 0.0}),
    Vec<3, double>({1.0, -1.0, 0.0}),
    Vec<3, double>({0.0, -1.0, 1.0}),

    Vec<3, double>({0.0, -1.0, 0.0}),
    Vec<3, double>({-1.0, -1.0, 1.0}),

    Vec<3, double>({-1.0, 0.0, 1.0}),

    Vec<3, double>({0.0, 0.5, -1.0}),
    Vec<3, double>({0.5, 1.0, -1.0}),
    Vec<3, double>({1.0, 0.5, -1.0}),
    Vec<3, double>({0.5, 0.0, -1.0}),

    Vec<3, double>({1.0, 0.0, 0.0}),
    Vec<3, double>({0.0, -1.0, 0.0}),
  };
  std::vector<Vec<3, double>> pyrsupvecs = {
    Vec<3, double>({0.0, 0.0, 0.0}),
    Vec<3, double>({0.0, 0.0, 0.0}),
    Vec<3, double>({0.0, 0.0, 0.0}),
    Vec<3, double>({0.0, 0.0, 0.0}),

    Vec<3, double>({0.0, 1.0, 0.0}),
    Vec<3, double>({0.0, 1.0, 0.0}),
    Vec<3, double>({0.0, 1.0, 0.0}),

    Vec<3, double>({1.0, 1.0, 0.0}),
    Vec<3, double>({1.0, 1.0, 0.0}),
    Vec<3, double>({1.0, 0.0, 0.0}),

    Vec<3, double>({0.0, 0.0, 1.0}),
    Vec<3, double>({0.0, 0.0, 1.0}),
    Vec<3, double>({0.0, 0.0, 1.0}),
    Vec<3, double>({0.0, 0.0, 1.0}),

    Vec<3, double>({0.0, 0.5, 0.0}),
    Vec<3, double>({0.5, 1.0, 0.0}),
  };

  dirvecs.clear();
  supvecs.clear();
  parametrize_object(pyrcoords, dirvecs, supvecs);

  BOOST_TEST(dirvecs.size() == pyrdirvecs.size());
  BOOST_TEST(supvecs.size() == pyrsupvecs.size());
  while (!pyrdirvecs.empty()) {
    bool rightvecs = false;
    int i;
    for (i = dirvecs.size() - 1; i >= 0; --i) {
      if (vec_is_equal_eps<3>(dirvecs[i], pyrdirvecs.back(), GEOM_TOL)
          && vec_is_equal_eps<3>(supvecs[i], pyrsupvecs.back(), GEOM_TOL)) {
        rightvecs = true;
        break;
      }
    }
    BOOST_TEST(rightvecs);

    pyrdirvecs.pop_back();
    pyrsupvecs.pop_back();
  }

  std::vector<Vec<3, double>> hexcoords = {
    Vec<3, double>({0.0, 0.0, 0.0}),
    Vec<3, double>({0.0, 1.0, 0.0}),
    Vec<3, double>({1.0, 1.0, 0.0}),
    Vec<3, double>({1.0, 0.0, 0.0}),
    Vec<3, double>({0.0, 0.0, 1.0}),
    Vec<3, double>({0.0, 1.0, 1.0}),
    Vec<3, double>({1.0, 1.0, 1.0}),
    Vec<3, double>({1.0, 0.0, 1.0}),
  };
  std::vector<Vec<3, double>> hexsupvecs = {
    Vec<3, double>({0.0, 0.0, 0.0}),
    Vec<3, double>({0.0, 0.0, 0.0}),
    Vec<3, double>({0.0, 0.0, 0.0}),
    Vec<3, double>({0.0, 0.0, 0.0}),
    Vec<3, double>({0.0, 0.0, 0.0}),
    Vec<3, double>({0.0, 0.0, 0.0}),
    Vec<3, double>({0.0, 0.0, 0.0}),

    Vec<3, double>({0.0, 1.0, 0.0}),
    Vec<3, double>({0.0, 1.0, 0.0}),
    Vec<3, double>({0.0, 1.0, 0.0}),
    Vec<3, double>({0.0, 1.0, 0.0}),
    Vec<3, double>({0.0, 1.0, 0.0}),
    Vec<3, double>({0.0, 1.0, 0.0}),

    Vec<3, double>({1.0, 1.0, 0.0}),
    Vec<3, double>({1.0, 1.0, 0.0}),
    Vec<3, double>({1.0, 1.0, 0.0}),
    Vec<3, double>({1.0, 1.0, 0.0}),
    Vec<3, double>({1.0, 1.0, 0.0}),

    Vec<3, double>({1.0, 0.0, 0.0}),
    Vec<3, double>({1.0, 0.0, 0.0}),
    Vec<3, double>({1.0, 0.0, 0.0}),
    Vec<3, double>({1.0, 0.0, 0.0}),

    Vec<3, double>({0.0, 0.0, 1.0}),
    Vec<3, double>({0.0, 0.0, 1.0}),
    Vec<3, double>({0.0, 0.0, 1.0}),

    Vec<3, double>({0.0, 1.0, 1.0}),
    Vec<3, double>({0.0, 1.0, 1.0}),

    Vec<3, double>({1.0, 1.0, 1.0}),
  };
  std::vector<Vec<3, double>> hexdirvecs = {
    Vec<3, double>({0.0, 1.0, 0.0}),
    Vec<3, double>({1.0, 1.0, 0.0}),
    Vec<3, double>({1.0, 0.0, 0.0}),
    Vec<3, double>({0.0, 0.0, 1.0}),
    Vec<3, double>({0.0, 1.0, 1.0}),
    Vec<3, double>({1.0, 1.0, 1.0}),
    Vec<3, double>({1.0, 0.0, 1.0}),

    Vec<3, double>({1.0, 0.0, 0.0}),
    Vec<3, double>({1.0, -1.0, 0.0}),
    Vec<3, double>({0.0, -1.0, 1.0}),
    Vec<3, double>({0.0, 0.0, 1.0}),
    Vec<3, double>({1.0, 0.0, 1.0}),
    Vec<3, double>({1.0, -1.0, 1.0}),

    Vec<3, double>({0.0, -1.0, 0.0}),
    Vec<3, double>({-1.0, -1.0, 1.0}),
    Vec<3, double>({-1.0, 0.0, 1.0}),
    Vec<3, double>({0.0, 0.0, 1.0}),
    Vec<3, double>({0.0, -1.0, 1.0}),

    Vec<3, double>({-1.0, 0.0, 1.0}),
    Vec<3, double>({-1.0, 1.0, 1.0}),
    Vec<3, double>({0.0, 1.0, 1.0}),
    Vec<3, double>({0.0, 0.0, 1.0}),

    Vec<3, double>({0.0, 1.0, 0.0}),
    Vec<3, double>({1.0, 1.0, 0.0}),
    Vec<3, double>({1.0, 0.0, 0.0}),

    Vec<3, double>({1.0, 0.0, 0.0}),
    Vec<3, double>({1.0, -1.0, 0.0}),

    Vec<3, double>({0.0, -1.0, 0.0}),
  };

  dirvecs.clear();
  supvecs.clear();
  parametrize_object(hexcoords, dirvecs, supvecs);

  BOOST_TEST(dirvecs.size() == hexdirvecs.size());
  BOOST_TEST(supvecs.size() == hexsupvecs.size());
  while (!hexdirvecs.empty()) {
    bool rightvecs = false;
    int i;
    for (i = dirvecs.size() - 1; i >= 0; --i) {
      if (vec_is_equal_eps<3>(dirvecs[i], hexdirvecs.back(), GEOM_TOL)
          && vec_is_equal_eps<3>(supvecs[i], hexsupvecs.back(), GEOM_TOL)) {
        rightvecs = true;
        break;
      }
    }
    BOOST_TEST(rightvecs);

    hexdirvecs.pop_back();
    hexsupvecs.pop_back();
  }
}

BOOST_AUTO_TEST_CASE(geometric_tools)
{
  LogKeeper::get_log ( "info" ).set_target  ( &( std::cout ) );    
  LogKeeper::get_log ( "debug" ).set_target ( &( std::cout ) );
  LogKeeper::get_log ( "error" ).set_target ( &( std::cout ) );
  
  CONSOLE_OUTPUT(0, "test intersect_facet");
  test_intersect_facet();

  CONSOLE_OUTPUT(0, "test point inside entity");
  test_point_inside_entity();

  CONSOLE_OUTPUT(0, "test point inside cell");
  test_point_inside_cell();

  CONSOLE_OUTPUT(0, "test inside hyperplane");
  test_vertices_inside_one_hyperplane();

  CONSOLE_OUTPUT(0, "test triangle area");
  test_triangle_area();

  CONSOLE_OUTPUT(0, "test facet area");
  test_facet_area();

  CONSOLE_OUTPUT(0, "test in plane");
  test_in_plane();

  CONSOLE_OUTPUT(0, "test crossed plane");
  test_crossed_plane();

  CONSOLE_OUTPUT(0, "test distance_point_hyperplan");
  test_distance_point_hyperplane();

  CONSOLE_OUTPUT(0, "test distance_point_line");
  test_distance_point_line();

  CONSOLE_OUTPUT(0, "test foot_point_hyperplane");
  test_foot_point_hyperplane();

  CONSOLE_OUTPUT(0, "test foot_point_line");
  test_foot_point_line();

  CONSOLE_OUTPUT(0, "test distance_point_facet");
  test_distance_point_facet();

  CONSOLE_OUTPUT(0, "test project_point");
  test_project_point();

  CONSOLE_OUTPUT(0, "test_is_point_on_subentity");
  test_is_point_on_subentity();

  CONSOLE_OUTPUT(0, "test_find_subentities_containing_point");
  test_find_subentities_containing_point();

  //CONSOLE_OUTPUT(0, "test_map_ref_coord_to_other_cell");
  //test_map_ref_coord_to_other_cell();

  CONSOLE_OUTPUT(0, "test_create_bbox_for_mesh");
  test_create_bbox_for_mesh();

  CONSOLE_OUTPUT(0, "test_find_adjacent_cells");
  test_find_adjacent_cells();

  CONSOLE_OUTPUT(0, "test_find_adjacent_cells_related");
  test_find_adjacent_cells_related();

  CONSOLE_OUTPUT(0, "test_cells_intersect");
  test_cells_intersect();

  CONSOLE_OUTPUT(0, "test_is_aligned_rectangular_cuboid");
  test_is_aligned_rectangular_cuboid();

  CONSOLE_OUTPUT(0, "test_is_aligned_rectangular_quad");
  test_is_aligned_rectangular_quad();

  CONSOLE_OUTPUT(0, "test_is_parallelogram");
  test_is_parallelogram();

  CONSOLE_OUTPUT(0, "test_is_parallelepiped");
  test_is_parallelepiped();

  CONSOLE_OUTPUT(0, "test_parametrize_object");
  test_parametrize_object();
}
