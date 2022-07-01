
###-------------------------------------------------------------------------###
### Import python modules
###-------------------------------------------------------------------------###

import chaospy as cp

def identical_dist ( dist, N ):
  """
  Generate identical indepentend distributions
  input:
    dist: 1D distribution to multiply
    N: Dimension of stochastical space
  return:
    Joint distribution
  """
  if N == 1:
    return dist
  return cp.Iid( dist, N )


def combine_dist ( dists ):
  """
  Combine mixed independent distribution
  input:
    dists: List of distributions to combine
  return:
    Joint distribution
  """
  if len( dists ) == 1:
    return dists[0]
  return cp.J( *dists )

### Available distributions in chaospy
### Alpha(shape=1, scale=1, shift=0)
### Anglit(loc=0, scale=1)
### Arcsinus(shape=0.5, lo=0, up=1)
### Beta(a, b, lo=0, up=1)
### Bradford(shape=1, lo=0, up=1)
### Burr(c=1, d=1, loc=0, scale=1)
### Cauchy(loc=0, scale=1)
### Chi(df=1, scale=1, shift=0)
### Chisquard(df=1, scale=1, shift=0, nc=0)
### Dbl_gamma(shape=1, scale=1, shift=0)
### Dbl_weibull(shape=1, scale=1, shift=0)
### Exponential(scale=1, shift=0)
### Exponpow(shape=0, scale=1, shift=0)
### Exponweibull(a=1, c=1, scale=1, shift=0)
### F(n=1, m=1, scale=1, shift=0, nc=0)
### Fatiguelife(shape=1, scale=1, shift=0)
### Fisk(shape=1, scale=1, shift=0)
### Foldcauchy(shape=0, scale=1, shift=0)
### Foldnormal(mu=0, sigma=1, loc=0)
### Frechet(shape=1, scale=1, shift=0)
### Gamma(shape=1, scale=1, shift=0)
### Genexpon(a=1, b=1, c=1, scale=1, shift=0)
### Genextreme(shape=0, scale=1, loc=0)
### Gengamma(shape1, shape2, scale, shift)
### Genhalflogistic(shape, scale, shift)
### Gilbrat(scale=1, shift=0)
### Gompertz(shape, scale, shift)
### Logweibul(scale=1, loc=0)
### Hypgeosec(loc=0, scale=1)
### Kumaraswamy(a, b, lo=0, up=1)
### Laplace(mu=0, scale=1)
### Levy(loc=0, scale=1)
### Loggamma(shape=1, scale=1, shift=0)
### Logistic(loc=0, scale=1, skew=1)
### Loglaplace(shape=1, scale=1, shift=0)
### Lognormal(mu=0, sigma=1, shift=0, scale=1)
### Loguniform(lo=0, up=1, scale=1, shift=0)
### Maxwell(scale=1, shift=0)
### Mielke(kappa=1, expo=1, scale=1, shift=0)
### MvLognormal(loc=[0,0], scale=[[1,.5],[.5,1]])
### MvNormal(loc=[0,0], scale=[[1,.5],[.5,1]])
### MvStudent_t(df=1, loc=[0,0], scale=[[1,.5],[.5,1]])
### Nakagami(shape=1, scale=1, shift=0)
### Normal(mu=0, sigma=1)
### OTDistribution(distribution)
### Pareto1(shape=1, scale=1, loc=0)
### Pareto2(shape=1, scale=1, loc=0)
### Powerlaw(shape=1, lo=0, up=1)
### Powerlognormal(shape=1, mu=0, sigma=1, shift=0, scale=1)
### Powernorm(shape=1, mu=0, scale=1)
### Raised_cosine(loc=0, scale=1)
### Rayleigh(scale=1, shift=0)
### Reciprocal(lo=1, up=2)
### Student_t(df, loc=0, scale=1, nc=0)
### Triangle(lo, mid, up)
### Truncexpon(up=1, scale=1, shift=0)
### Truncnorm(lo=-1, up=1, mu=0, sigma=1)
### Tukeylambda(shape=0, scale=1, shift=0)
### Uniform(lo=0, up=1)
### Wald(mu=0, scale=1, shift=0)
### Weibull(shape=1, scale=1, shift=0)
### Wigner(radius=1, shift=0)
### Wrapcauchy(shape=0.5, scale=1, shift=0)
### SampleDist(samples, lo=None, up=None)


