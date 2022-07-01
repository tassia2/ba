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

/// \author Michael Schick

#include "polynomial_chaos/pc_basis.h"
#include "common/log.h"
#include "common/macros.h"
#include "common/permutation.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>

namespace hiflow {
namespace polynomialchaos {

PCBasis::PCBasis() {
  N_ = -1;
  No_ = -1;
  Dim_ = -1;
  q_ = -1.0;
  t_ = -1;
  num_quad_points_ = -1;

  // Precomputed quadrature is exact up to polynomial order 41
  x_qpts_precomp_.resize(ndist_);
  w_qpts_precomp_.resize(ndist_);

  double x_qpts_tmp[] = {-9.937521706203894523e-01, -9.672268385663064238e-01,
                         -9.200993341504007939e-01, -8.533633645833176296e-01,
                         -7.684399634756780006e-01, -6.671388041974124494e-01,
                         -5.516188358872193831e-01, -4.243421202074389997e-01,
                         -2.880213168024010617e-01, -1.455618541608955929e-01,
                         1.544987264365841335e-17,  1.455618541608950933e-01,
                         2.880213168024010617e-01,  4.243421202074387222e-01,
                         5.516188358872200492e-01,  6.671388041974123384e-01,
                         7.684399634756776676e-01,  8.533633645833172965e-01,
                         9.200993341504010159e-01,  9.672268385663062018e-01,
                         9.937521706203894523e-01};

  double x_qpts_tmp_g[] = {-7.849382895113821590e+00, -6.751444718717457327e+00,
                           -5.829382007304467095e+00, -4.994963944782024434e+00,
                           -4.214343981688423391e+00, -3.469846690475375084e+00,
                           -2.750592981052374153e+00, -2.049102468257162801e+00,
                           -1.359765823211230407e+00, -6.780456924406442765e-01,
                           -1.474106067249749284e-16, 6.780456924406442765e-01,
                           1.359765823211230629e+00,  2.049102468257163689e+00,
                           2.750592981052373709e+00,  3.469846690475375972e+00,
                           4.214343981688421614e+00,  4.994963944782024434e+00,
                           5.829382007304468871e+00,  6.751444718717459992e+00,
                           7.849382895113824254e+00};

  double w_qpts_tmp[] = {8.008614128887271352e-03, 1.847689488542610808e-02,
                         2.856721271342867532e-02, 3.805005681418983238e-02,
                         4.672221172801710454e-02, 5.439864958357445990e-02,
                         6.091570802686405162e-02, 6.613446931666877582e-02,
                         6.994369739553660259e-02, 7.226220199498513408e-02,
                         7.304056682484481866e-02, 7.226220199498505081e-02,
                         6.994369739553646381e-02, 6.613446931666837336e-02,
                         6.091570802686474551e-02, 5.439864958357445990e-02,
                         4.672221172801706290e-02, 3.805005681418925645e-02,
                         2.856721271342847063e-02, 1.847689488542662156e-02,
                         8.008614128887031960e-03};

  double w_qpts_tmp_g[] = {2.098991219565668087e-14, 4.975368604121643035e-11,
                           1.450661284493110865e-08, 1.225354836148250053e-06,
                           4.219234742551674646e-05, 7.080477954815380141e-04,
                           6.439697051408762966e-03, 3.395272978654294976e-02,
                           1.083922856264193380e-01, 2.153337156950596021e-01,
                           2.702601835728762336e-01, 2.153337156950596021e-01,
                           1.083922856264195184e-01, 3.395272978654288731e-02,
                           6.439697051408775109e-03, 7.080477954815368215e-04,
                           4.219234742551678712e-05, 1.225354836148263605e-06,
                           1.450661284493089523e-08, 4.975368604121606200e-11,
                           2.098991219565668087e-14};

  x_qpts_precomp_[UNIFORM].reserve(21);
  x_qpts_precomp_[UNIFORM].insert(x_qpts_precomp_[UNIFORM].begin(), x_qpts_tmp,
                                  x_qpts_tmp + 21);
  w_qpts_precomp_[UNIFORM].reserve(21);
  w_qpts_precomp_[UNIFORM].insert(w_qpts_precomp_[UNIFORM].begin(), w_qpts_tmp,
                                  w_qpts_tmp + 21);

  x_qpts_precomp_[GAUSSIAN].reserve(21);
  x_qpts_precomp_[GAUSSIAN].insert(x_qpts_precomp_[GAUSSIAN].begin(),
                                   x_qpts_tmp_g, x_qpts_tmp_g + 21);
  w_qpts_precomp_[GAUSSIAN].reserve(21);
  w_qpts_precomp_[GAUSSIAN].insert(w_qpts_precomp_[GAUSSIAN].begin(),
                                   w_qpts_tmp_g, w_qpts_tmp_g + 21);

  x_qpts_precomp_[CUSTOM].reserve(21);
  x_qpts_precomp_[CUSTOM].insert(x_qpts_precomp_[CUSTOM].begin(), x_qpts_tmp,
                                 x_qpts_tmp + 21);
  w_qpts_precomp_[CUSTOM].reserve(21);
  w_qpts_precomp_[CUSTOM].insert(w_qpts_precomp_[CUSTOM].begin(), w_qpts_tmp,
                                 w_qpts_tmp + 21);
}

int PCBasis::CalculateDimension(int No, int N) const {
  // Calculate total dimension
  long int fak_No = 1;

  for (int i = 1; i <= No; ++i) {
    fak_No *= i;
  }

  long int tmp = 1;
  for (int i = N + 1; i <= No + N; ++i) {
    tmp *= i;
  }

  return tmp / fak_No;
}

void PCBasis::Init(int N, int No,
                   std::vector< Distribution > const &distributions,
                   const std::vector< PDF< double > * > &custom_distributions) {
  assert(static_cast< int >(distributions.size()) == N);

  N_ = N;
  No_ = No;
  num_quad_points_ = std::max((int)ceil(1.5 * No_), 30);

  q_ = 1.0;
  t_ = N_;

  SubDim_.resize(No_ + 1);
  SubDim_[0] = 1;

  for (int p = 1; p < static_cast< int >(SubDim_.size()); ++p) {
    SubDim_[p] = CalculateDimension(p, N_);
  }

  // Calculate total dimension
  int fak_N = 1;
  int fak_No = 1;
  int fak_NplusNo = 1;

  for (int i = 1; i <= N_; ++i) {
    fak_N *= i;
  }
  for (int i = 1; i <= No_; ++i) {
    fak_No *= i;
  }
  for (int i = 1; i <= N_ + No_; ++i) {
    fak_NplusNo *= i;
  }

  int tmp = 1;
  for (int i = No_ + 1; i <= N_ + No_; ++i) {
    tmp *= i;
  }

  Dim_ = tmp / fak_N;

  distributions_ = distributions;
  custom_distr_ = custom_distributions;

  if (custom_distr_.empty()) {
    for (int i = 0; i < distributions_.size(); ++i) {
      if (distributions_[i] == CUSTOM) {
        LOG_ERROR("Need to specify custom_distributions, when using CUSTOM");
        exit(-1);
      }
    }
  } else {
    if (custom_distr_.size() != distributions_.size()) {
      LOG_ERROR(
          "custom_distribution must be of size 0 (default) when not using "
          "CUSTOM PDFs or same size as distributions when using CUSTOM PDFs");
      exit(-1);
    }
  }

  ComputeRecursionCoeff();

  quadrature_computed_.resize(ndist_, false);

  x_qpts_.resize(N_);
  w_qpts_.resize(N_);
  ComputeQuadrature();

  ComputeMultiIndices();
  Precomputed3rdOrderIntegrals();
}

void PCBasis::Init(int N, int No,
                   std::vector< Distribution > const &distributions, double q,
                   int t,
                   const std::vector< PDF< double > * > &custom_distributions) {
  assert(static_cast< int >(distributions.size()) == N);

  N_ = N;
  No_ = No;
  num_quad_points_ = std::max((int)ceil(1.5 * No_), 30);
  q_ = q;
  t_ = t;

  SubDim_.resize(No_ + 1);
  SubDim_[0] = 1;

  for (int p = 1; p < static_cast< int >(SubDim_.size()); ++p) {
    SubDim_[p] = CalculateDimension(p, N_);
  }

  distributions_ = distributions;
  custom_distr_ = custom_distributions;

  if (custom_distr_.empty()) {
    for (size_t i = 0; i < distributions_.size(); ++i) {
      if (distributions_[i] == CUSTOM) {
        LOG_ERROR("Need to specify custom_distributions, when using CUSTOM");
        exit(-1);
      }
    }
  } else {
    if (custom_distr_.size() != distributions_.size()) {
      LOG_ERROR(
          "custom_distribution must be of size 0 (default) when not using "
          "CUSTOM PDFs or same size as distributions when using CUSTOM PDFs");
      exit(-1);
    }
  }

  ComputeRecursionCoeff();
  quadrature_computed_.resize(ndist_, false);

  x_qpts_.resize(N_);
  w_qpts_.resize(N_);
  // ComputeQpointsAndWeights ( );

  ComputeQuadrature();
  ComputeMultiIndices();

  Dim_ = alpha_.size();

  Precomputed3rdOrderIntegrals();
}

void PCBasis::ComputeQuadrature() {
#ifdef WITH_GAUSSQ
  for (int i = 0; i < N_; ++i) {
    if (distributions_[i] == UNIFORM)
      ComputeGaussLegendreQuadrature();
    if (distributions_[i] == GAUSSIAN)
      ComputeGaussHermiteQuadrature();
    if (distributions_[i] == NOT_SET)
      assert(0);
  }
#else
  for (size_t i = 0; i < N_; ++i) {
    w_qpts_[i] = w_qpts_precomp_[distributions_[i]];
    x_qpts_[i] = x_qpts_precomp_[distributions_[i]];
    if (distributions_[i] == CUSTOM) {
      PDF< double > *pdf = custom_distr_[i];
      double c, d;
      pdf->GetIntegrationInterval(&c, &d);
      for (int j = 0; j < x_qpts_[i].size(); ++j) {
        x_qpts_[i][j] = (d - c) * (x_qpts_[i][j] + 1.) / 2. + c;
        w_qpts_[i][j] =
            (d - c) * (w_qpts_[i][j]) * (*custom_distr_[i])(x_qpts_[i][j]);
        ;
      }
    }
    quadrature_computed_[distributions_[i]] = true;
  }
#endif
  ComputeQpointsAndWeights();
}

void PCBasis::GetQuadrature(std::vector< std::vector< double > > &qpts,
                            std::vector< double > &weights) const {
  int num_quad_points_total = 1.;
  for (int n = 0; n < N_; ++n) {
    num_quad_points_total *= w_qpts_[n].size();
  }

  qpts.resize(num_quad_points_total, std::vector< double >(N_));
  weights.resize(num_quad_points_total);
  std::vector< int > curr_indices(N_);
  int id;
  for (int i = 0; i < num_quad_points_total; ++i) {
    id = i;
    for (int n = 0; n < N_; ++n) {
      curr_indices[n] = id % w_qpts_[n].size();
      id -= curr_indices[n];
      id /= w_qpts_[n].size();
    }

    weights[i] = 1.;
    for (int n = 0; n < N_; ++n) {
      weights[i] *= w_qpts_[n][curr_indices[n]];
      qpts[i][n] = x_qpts_[n][curr_indices[n]];
    }
  }
}

void PCBasis::ComputeGaussLegendreQuadrature() {
  // if ( !quadrature_computed_[UNIFORM] )
  //{
  int n;
  // Exact up to 4th order tensor
  if ((4 * No_ + 1) % 2 == 0) {
    n = (4 * No_ + 1) / 2;
  } else {
    n = (4 * No_ + 1) / 2 + 1;
  }

  assert(n > 0);

  double *t = new double[n];
  double *w = new double[n];

#ifdef WITH_GAUSSQ
  double alpha = 0.0;
  double beta = 0.0;
  int kpts = 0;
  double *endpts = new double[2];
  double *b = new double[n];

  int kind = 1;

  gaussq_(&kind, &n, &alpha, &beta, &kpts, endpts, b, t, w);
#endif
  w_qpts_[UNIFORM].resize(n);
  x_qpts_[UNIFORM].resize(n);

  for (int i = 0; i < n; ++i) {
    w_qpts_[UNIFORM][i] = 0.5 * w[i];
    x_qpts_[UNIFORM][i] = t[i];
  }

#ifdef WITH_GAUSSQ
  delete[] endpts;
  delete[] b;
#endif
  delete[] t;
  delete[] w;

  quadrature_computed_[UNIFORM] = true;
  //}
}

void PCBasis::ComputeGaussHermiteQuadrature() {
  // if ( !quadrature_computed_[GAUSSIAN] )
  //{
  int n;
  // Exact up to 4th order tensor
  if ((4 * No_ + 1) % 2 == 0) {
    n = (4 * No_ + 1) / 2;
  } else {
    n = (4 * No_ + 1) / 2 + 1;
  }

  assert(n > 0);

  double *t = new double[n];
  double *w = new double[n];

#ifdef WITH_GAUSSQ
  double alpha = 0.0;
  double beta = 0.0;
  int kpts = 0;
  double *endpts = new double[2];
  double *b = new double[n];

  int kind = 4;

  gaussq_(&kind, &n, &alpha, &beta, &kpts, endpts, b, t, w);
#endif

  w_qpts_[GAUSSIAN].resize(n);
  x_qpts_[GAUSSIAN].resize(n);
  for (int i = 0; i < n; ++i) {
    w_qpts_[GAUSSIAN][i] = (1.0 / std::sqrt(M_PI)) * w[i];
    x_qpts_[GAUSSIAN][i] = std::sqrt(2.0) * t[i];
  }

#ifdef WITH_GAUSSQ
  delete[] endpts;
  delete[] b;
#endif
  delete[] t;
  delete[] w;

  quadrature_computed_[GAUSSIAN] = true;
  //}
}

double PCBasis::LegendreP(int degree, double x) const {
  if (degree == 0) {
    return 1.0;
  }
  if (degree == 1) {
    return x;
  }
  return ((2.0 * degree - 1.0) * x * LegendreP(degree - 1, x) / degree -
          (degree - 1.0) * LegendreP(degree - 2, x) / degree);
}

double PCBasis::HermiteP(int degree, double x) const {
  if (degree == 0) {
    return 1.0;
  }
  if (degree == 1) {
    return x;
  }
  return (x * HermiteP(degree - 1, x) -
          (degree - 1.0) * HermiteP(degree - 2, x));
}

double PCBasis::CustomP(int degree, double x, int index) const {
  if (degree == 0) {
    return 1.; /// normalizer_[index];
  }
  if (degree == 1) {
    return (x - custom_alpha_[index][degree - 1]) /
           custom_beta_[index][degree - 1] * CustomP(degree - 1, x, index);
  }
  return ((x - custom_alpha_[index][degree - 1]) *
              CustomP(degree - 1, x, index) -
          custom_beta_[index][degree - 2] * CustomP(degree - 2, x, index)) /
         custom_beta_[index][degree - 1];
}

/*double PCBasis::poly ( Distribution dist, int degree, double x ) const
{
    switch ( dist )
    {
        case UNIFORM:
            return LegendreP ( degree, x );
        case GAUSSIAN:
            return HermiteP ( degree, x );
        default:
        {
            assert ( 0 );
            return 0.0;
        }
    };
}*/

double PCBasis::poly(int dist_index, int degree, double x) const {
  switch (distributions_[dist_index]) {
  case UNIFORM:
    return LegendreP(degree, x);
  case GAUSSIAN:
    return HermiteP(degree, x);
  case CUSTOM:
    return CustomP(degree, x, dist_index);
  default: {
    assert(0);
    return 0.0;
  }
  };
}

void PCBasis::Precomputed3rdOrderIntegrals() {
  third_order_integrals_.clear();
  third_order_integrals_.resize(N_);
  for (int n = 0; n < N_; ++n) {
    third_order_integrals_[n].resize(No_ + 1);
    for (int i = 0; i <= No_; ++i) {
      third_order_integrals_[n][i].resize(No_ + 1);
      for (int j = 0; j <= No_; ++j) {
        third_order_integrals_[n][i][j].resize(No_ + 1, 0.0);
      }
    }
  }
  for (int n = 0; n < N_; ++n) {
    for (int i = 0; i <= No_; ++i) {
      for (int j = 0; j <= No_; ++j) {
        for (int k = 0; k <= No_; ++k) {
          if (k <= i + j && i <= j + k && j <= i + k) {
            for (int qpt = 0; qpt < static_cast< int >(x_qpts_[n].size());
                 ++qpt) {
              third_order_integrals_[n][i][j][k] +=
                  w_qpts_[n][qpt] * poly(n, i, x_qpts_[n][qpt]) *
                  poly(n, j, x_qpts_[n][qpt]) * poly(n, k, x_qpts_[n][qpt]);
            }
          }
        }
      }
    }
  }
}

double PCBasis::ComputeIntegral3rdOrder(int i, int j, int k) const {
  double integral = 1.0;
  for (int n = 0; n < N_; ++n) {
    integral *=
        third_order_integrals_[n][alpha_[i][n]][alpha_[j][n]][alpha_[k][n]];
  }
  // std::cout << i << '\t' << j << '\t' << k << '\t' << integral << std::endl;
  return integral;
}

double
PCBasis::ComputeIntegral(std::vector< int > const &global_indices) const {
  int size = global_indices.size();

  // 1st filter polynomial degrees according to distributions
  std::vector< std::vector< int > > degrees(N_);
  for (int k = 0; k < size; ++k) {
    for (int i = 0; i < N_; ++i) {
      degrees[i].push_back(alpha_[global_indices[k]][i]);
    }
  }

  // 2nd compute integral with appropriate quadrature and distribution
  std::vector< double > loc_integral(N_, 0.0);
  for (int j = 0; j < N_; ++j) {
    for (int qpt = 0; qpt < static_cast< int >(x_qpts_[j].size()); ++qpt) {
      double f = 1.0;
      for (int i = 0; i < static_cast< int >(degrees[j].size()); ++i)
        f *= poly(j, degrees[j][i], x_qpts_[j][qpt]);
      loc_integral[j] += w_qpts_[j][qpt] * f;
    }
  }

  // 3rd compute total integral as product of local ones
  double integral = 1.0;
  for (int i = 0; i < N_; ++i) {
    integral *= loc_integral[i];
  }

  return integral;
}

void PCBasis::ComputeMultiIndices() {
  if (q_ == 0 && t_ == 0) {
    alpha_.resize(1);
    alpha_[0].resize(N_, 0);
    return;
  }

  int minDim_ = CalculateDimension(No_, N_);
  bool notfinished = true;
  std::vector< std::vector< int > > beta_;

  beta_.resize(minDim_);
  if (q_ != 0) {
    alpha_.resize(minDim_);
  }
  if (q_ == 0) {
    alpha_.resize(N_ + 1);
  }

  for (int i = 0; i < minDim_; ++i) {
    beta_[i].resize(N_, 0);
    if (q_ != 0) {
      alpha_[i].resize(N_, 0);
    }
    if (q_ == 0 && i < N_ + 1) {
      alpha_[i].resize(N_, 0);
    }
  }

  if (No_ > 0) {
    for (int i = 1; i <= N_; ++i) {
      beta_[i][i - 1] = 1;
      alpha_[i][i - 1] = 1;
    }
  }

  if (No_ > 1) {
    int P = N_;
    std::vector< std::vector< int > > p(N_);

    for (int i = 0; i < N_; ++i)
      p[i].resize(No_, 0);

    for (int i = 0; i < N_; ++i)
      p[i][0] = 1;

    for (int k = 1; notfinished; ++k) {

      if (q_ == 0 && k == No_ - 1) {
        notfinished = false;
      }

      if (k >= No_) {
        notfinished = false;
        int lastDim_ = beta_.size();
        int newDim_ = CalculateDimension(k + 1, N_);
        beta_.resize(newDim_);
        for (int i = lastDim_; i < newDim_; ++i) {
          beta_[i].resize(N_, 0);
        }
        for (int i = 0; i < N_; ++i) {
          p[i].resize(k + 1, 0);
        }
      }
      int L = P;

      for (int i = 0; i < N_; ++i) {
        int sum = 0;
        for (int m = i; m < N_; ++m)
          sum += p[m][k - 1];
        p[i][k] = sum;
      }

      for (int j = 0; j < N_; ++j) {
        for (int m = L - p[j][k] + 1; m <= L; ++m) {
          P++;
          for (int i = 0; i < N_; ++i) {
            beta_[P][i] = beta_[m][i];
          }
          beta_[P][j] = beta_[P][j] + 1;

          if (q_ == 0) {
            int max = *std::max_element(beta_[P].begin(), beta_[P].end());
            if (max <= 1) {
              int vectorsum = 0;
              for (int i = 0; i < N_; ++i) {
                vectorsum += beta_[P][i];
              }
              if (vectorsum <= t_) {
                int size = alpha_.size();
                alpha_.resize(size + 1);
                alpha_[size].resize(N_, 0);
                for (int i = 0; i < N_; ++i) {
                  alpha_[size][i] = beta_[P][i];
                }
              }
            }
          }

          if (k < No_ && q_ != 0) {
            for (int i = 0; i < N_; ++i) {
              alpha_[P][i] = beta_[P][i];
            }
          }

          if (k >= No_) {
            double vectorsum = 0;
            for (int i = 0; i < N_; ++i) {
              vectorsum += pow(beta_[P][i], q_);
            }
            if (vectorsum <= No_) {
              notfinished = true;
              int size = alpha_.size();
              alpha_.resize(size + 1);
              alpha_[size].resize(N_, 0);
              for (int i = 0; i < N_; ++i) {
                alpha_[size][i] = beta_[P][i];
              }
            }
          }
        }
      }
    }
  }
}

double PCBasis::poly(int index, std::vector< double > const &x) const {
  assert(static_cast< int >(x.size()) == N_);

  double res = 1.0;
  const std::vector< int > &loc_index = alpha_[index];

  for (int j = 0; j < N_; ++j) {
    res *= poly(j, loc_index[j], x[j]);
  }

  return res;
}

void PCBasis::ComputeRecursionCoeff() {
  int num_dist = distributions_.size();
  custom_alpha_.resize(num_dist);
  custom_beta_.resize(num_dist);
  normalizer_.resize(num_dist);
  ;
  for (int dist = 0; dist < num_dist; ++dist) {
    if (distributions_[dist] != CUSTOM) {
      if (distributions_[dist] == UNIFORM) {
      } else if (distributions_[dist] == GAUSSIAN) {
      } else {
        assert(0);
      }
    } else {
      // the recursive cooefficients are calculated
      // using the 3-term recursive formula.
      // Here there is used a modified version to
      // make the calculations more stable!
      PDF< double > *curr_pdf = custom_distr_[dist];

      double left, right;
      curr_pdf->GetIntegrationInterval(&left, &right);

      // use tschebycheff integration
      const int NN = 1000;
      std::vector< double > x_chebby(NN);
      std::vector< double > w_chebby(NN, 1.);

      for (int k = 0; k < NN; ++k) {
        double thetak = (2. * ((double)k + 1.) - 1.) * M_PI / (2. * (double)NN);
        x_chebby[k] = (cos(thetak) + 1.) / 2.;
        for (int m = 1; m < NN / 2 + 1; ++m) {
          w_chebby[k] -= 2. * cos(2. * (double)m * thetak) /
                         (4. * (double)m * (double)m - 1.);
        }
        w_chebby[k] *= 2. / (double)NN * (right - left) / 2.;
      }

      // arrays to store the values.
      double value_storage_polynomial_0[NN];
      double value_storage_polynomial_1[NN];
      double value_storage_pdf[NN];
      // pointers that are used to access the values
      // in the storage. This construction is used to easily
      // "swap" the values (they stay at the same location
      // in memory but the pointers swap instead)!
      double *p_values_prev = &value_storage_polynomial_0[0];
      double *p_values_curr = &value_storage_polynomial_1[0];
      double *pdf_values = &value_storage_pdf[0];
      double norm_factor = 0.;
      // calculating the values of the pdf at the quadrature
      // points (once).
      for (int k = 0; k < NN; ++k) {
        pdf_values[k] = (*curr_pdf)(x_chebby[k] * (right - left) + left);
        norm_factor += pdf_values[k] * w_chebby[k];
      }

      // calculating the values of the first to polynomials
      // (zero polynomial and constant polynomial)
      for (int i = 0; i < NN; ++i) {
        p_values_prev[i] = 0.;
        p_values_curr[i] = 1. / sqrt(norm_factor);
      }
      custom_alpha_[dist].resize(num_quad_points_);
      custom_beta_[dist].resize(num_quad_points_);
      custom_beta_[dist][0] = 0.;
      for (int n = 0; n < num_quad_points_; ++n) {
        double int_xpp = 0.;  // (xp_n, p_n)
        double int_xpxp = 0.; // (xp_n, xp_n)

        for (int k = 0; k < NN; ++k) {
          int_xpp += pdf_values[k] * x_chebby[k] * p_values_curr[k] *
                     p_values_curr[k] * w_chebby[k];
          int_xpxp += pdf_values[k] * x_chebby[k] * x_chebby[k] *
                      p_values_curr[k] * p_values_curr[k] * w_chebby[k];
        }
        custom_alpha_[dist][n] = int_xpp;
        // handling special case at n == 0
        double beta = (n > 0) ? custom_beta_[dist][n - 1] / (right - left) : 0.;
        custom_beta_[dist][n] =
            sqrt(int_xpxp - int_xpp * int_xpp - beta * beta);

        // swap current and previous
        double *temp = p_values_curr;
        p_values_curr = p_values_prev;
        p_values_prev = temp;
        // calculate new values (will override old ones)
        for (int k = 0; k < NN; ++k) {
          p_values_curr[k] = (x_chebby[k] - custom_alpha_[dist][n]) /
                                 custom_beta_[dist][n] * p_values_prev[k] -
                             beta / custom_beta_[dist][n] * p_values_curr[k];
        }
        // calculating the standard alpha and beta from the
        // one obtained by the transformed formulation!
        custom_alpha_[dist][n] = custom_alpha_[dist][n] * (right - left) + left;
        custom_beta_[dist][n] = custom_beta_[dist][n] * (right - left);

        // the new polynomial should be normed!
        double norm2 = 0.;
        for (int k = 0; k < NN; ++k) {
          norm2 +=
              pdf_values[k] * p_values_curr[k] * p_values_curr[k] * w_chebby[k];
        }
        LOG_DEBUG(2, "Squared norm of polynomial with degree " << n << " : "
                                                               << norm2);
        // we will normalize in case it isn't. (Check debug log to see if thats
        // the case).
        if (std::abs(norm2 - 1.) > 1e-6) {
          for (int k = 0; k < NN; ++k) {
            p_values_curr[k] /= sqrt(norm2);
          }
        }
      }

      // old, unstable calculation of recursive coefficients
      /*assert(custom_distr_[dist] != NULL);
      PDF<double>* curr_pdf = custom_distr_[dist];

      std::vector<double> moments( 2*num_quad_points_ + 1,0. );
      for(int i = 0; i < 2*num_quad_points_ + 1; ++i)
      {
          moments[i] = curr_pdf->GetMoment(i);
      }
      normalizer_[dist] = sqrt(moments[0]);

      // Moments define Matrix M through
      // M_ij = moments[i+j], i,j = 0...No_

      //cholesky decomposition of M = R^T R

      // R is upper - right triangular matrix with
      // R_ij = r_ij[i][j-i]  for j >= i
      // R_ij = 0             else
      std::vector< std::vector<double> > r_ij(num_quad_points_ + 1);
      for(int i = 0; i < num_quad_points_ + 1; ++i) // loop over rows
      {

          r_ij[i].resize(num_quad_points_ + 1 - i);
          std::vector<double>::iterator curr_ele = r_ij[i].begin();
          //curr_ele now pointing on r_ij[i][0]

          // calculate diagonal element R_ii

          //r_ij[i][0] = moments[i+i];
          *curr_ele = moments[i+i];

          for( int k = 0; k < i; ++k)
          {
              //r_ij[i][0] -= r_ij[k][i-k] * r_ij[k][i-k];
              *curr_ele -= r_ij[k][i-k] * r_ij[k][i-k];
          }
          //r_ij[i][0] = std::sqrt(r_ij[i][0]);
          *curr_ele = sqrt(*curr_ele);
          //make high level debug command
          //std::cout << "R_" << i << i << ": " << *curr_ele << std::endl;

          // calculate rest of row
          for( int j = i + 1; j < num_quad_points_ + 1; ++j)
          {
              ++curr_ele; //curr_ele now poiting on r_ij[i][j-i]
              //r_ij[i][j-i] = moments[i+j];
              *curr_ele = moments[i+j];

              for( int k = 0; k < i; ++k)
              {
                  //r_ij[i][i-j] -= r_ij[k][i-k] * r_ij[k][j-k];
                  *curr_ele -= r_ij[k][i-k] * r_ij[k][j-k];
              }
              *curr_ele = *curr_ele /r_ij[i][0];
              //make high level debug command?
              //std::cout << "R_" << i << j << ": " << *curr_ele << std::endl;
          }
          ++curr_ele;
          assert(curr_ele == r_ij[i].end());
      }

      // now one can easily calculate the coefficients:
      custom_alpha_[dist].resize(num_quad_points_);
      custom_beta_[dist].resize(num_quad_points_);
      //special case for first alpha
      custom_alpha_[dist][0] = r_ij[0][1]/r_ij[0][0];
      custom_beta_[dist][0] = r_ij[1][0]/r_ij[0][0];

      for (int j = 1; j < num_quad_points_; ++j)
      {
          custom_alpha_[dist][j] = r_ij[j][1]/
              r_ij[j][0] - r_ij[j-1][1]/r_ij[j-1][0];
          custom_beta_[dist][j] = r_ij[j+1][0]/r_ij[j][0];
      }*/

      LOG_DEBUG(1, "Recursive Coeffients for Distribution "
                       << dist << std::endl
                       << "Alpha : " << std::endl
                       << string_from_range(custom_alpha_[dist].begin(),
                                            custom_alpha_[dist].end())
                       << std::endl
                       << "Beta  : " << std::endl
                       << string_from_range(custom_beta_[dist].begin(),
                                            custom_beta_[dist].end()));
    }
  }
}

void calc_rotation(double &d, double &ac, double &an, double &bp, double &bc,
                   double &bn, double &zc, double &zn) {
  double aJ = ac;
  double aJp1 = an;
  double l = sqrt(d * d + bp * bp);
  double sinT = d / l;
  double cosT = bp / l;
  ac = aJ * cosT * cosT + 2 * bc * cosT * sinT + aJp1 * sinT * sinT;
  an = aJ * sinT * sinT - 2 * bc * cosT * sinT + aJp1 * cosT * cosT;
  bp = l;
  bc = (aJ - aJp1) * sinT * cosT + bc * (sinT * sinT - cosT * cosT);
  d = bn * sinT;
  bn = -bn * cosT;
  double zJ = zc;
  zc = zJ * cosT + zn * sinT;
  zn = zJ * sinT - zn * cosT;
}

struct DoubleCompareOP {
  DoubleCompareOP(std::vector< double > &v) : lookup(v) {}

  bool operator()(unsigned int a, unsigned int b) {
    return lookup[a] < lookup[b];
  }
  std::vector< double > &lookup;
};

void PCBasis::ComputeQpointsAndWeights() {

  int num_dist = distributions_.size();
  // custom_alpha_.resize(num_dist);
  // custom_beta_.resize(num_dist);
  // int max_iter = 1000;
  int local_max_iter = 100;
  double TOL = 1e-14;
  for (int dist = 0; dist < num_dist; ++dist) {
    if (distributions_[dist] != CUSTOM) {
      continue;
    }

    int N = num_quad_points_;
    std::vector< double > aN(custom_alpha_[dist]);
    assert(aN.size() == N);
    std::vector< double > bN(1);
    bN.reserve(N);
    bN.insert(bN.end(), custom_beta_[dist].begin(),
              custom_beta_[dist].end() - 1);
    assert(bN.size() == N);
    std::vector< double > zN(N, 0);
    zN[0] = 1.;
    int local_iter = 0;
    while (N > 1) {
      if (std::abs(bN[N - 1]) < TOL || local_iter > local_max_iter) {
        local_iter = 0;
        N = N - 1;
        continue;
      }
      double T = aN[N - 2] + aN[N - 1];
      double D = aN[N - 2] * aN[N - 1] - bN[N - 1] * bN[N - 1];

      if (T * T / 4. - D < 0)
        LOG_ERROR("Not possible to calculate quadrature points and weights for "
                  "this pdf. Possible problems are: Moments of pdf do not "
                  "exist, accuracy is not sufficient. Possible solutions: "
                  "Check your pdfs moments, try lower polynomial degree, use "
                  "more quadrature points to calculate moments,...");

      double lambda = T / 2. - sqrt(T * T / 4. - D);
      bN[0] = aN[0] - lambda;
      double dJ = bN[1];
      for (int j = 0; j < N - 1; ++j) {
        double dummy = 0;
        double &bn = (j == N - 2) ? dummy : bN.at(j + 2);
        calc_rotation(dJ, aN[j], aN[j + 1], bN[j], bN[j + 1], bn, zN[j],
                      zN[j + 1]);
      }

      ++local_iter;
    }
    x_qpts_[dist].assign(aN.begin(), aN.end());
    w_qpts_[dist].resize(aN.size());
    double moment = custom_distr_[dist]->GetMoment(0);
    int j = 0;
    for (std::vector< double >::iterator it = w_qpts_[dist].begin();
         it != w_qpts_[dist].end(); ++it, ++j) {
      *it = zN[j] * zN[j] * moment;
    }

    // sort Quadrature Points and apply permutation to Weights (Actually not
    // necessary)
    {
      std::vector< int > permutation(x_qpts_[dist].size());
      int i = 0;
      for (std::vector< int >::iterator it = permutation.begin();
           it != permutation.end(); ++it, ++i) {
        *it = i;
      }
      DoubleCompareOP cmp(x_qpts_[dist]);
      std::sort(permutation.begin(), permutation.end(), cmp);
      std::vector< double > temp_container(x_qpts_[dist].size());
      permute_vector(permutation, x_qpts_[dist], temp_container);
      x_qpts_[dist] = temp_container;
      permute_vector(permutation, w_qpts_[dist], temp_container);
      w_qpts_[dist] = temp_container;
    }
    // make low level debug command
    LOG_DEBUG(
        1, "Quadrature Rule for Distribution "
               << dist << std::endl
               << "Quadpoints : " << std::endl
               << string_from_range(x_qpts_[dist].begin(), x_qpts_[dist].end())
               << std::endl
               << "Quadweights: " << std::endl
               << string_from_range(w_qpts_[dist].begin(), w_qpts_[dist].end()))
  }
}

} // namespace polynomialchaos
} // namespace hiflow
