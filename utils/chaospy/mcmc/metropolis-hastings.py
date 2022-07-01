### Metropolis-Hastings algorithm

import numpy as np
import chaospy as cp
from scipy import integrate

'''
### uq_postprocess.py für alle x in dist_list oder hier metropolis-hastings übergeben

# def metropolis_hastings ( proposal_dist, target_dist, start = np.zeros( proposal_dist.np.shape ), chain_size = 100 ):
# def metropolis_hastings ( proposal_dist, target_dist, start = False, chain_size = 100 ):   

#    if start == False:
#       x0 = np.zeros(proposal_dist.shape)
#
#    samples = np.zeros( proposal_dist.shape, order = chain_size )
#    
#    for i in range(chain_size):
#        proposal = [dist.sample(1) for dist in proposal_dist]
#
#        pi = [target_dist(p)*proposal_dist[p] for p in proposal]
#
#        p = [p*q]


# TODO: check for all distributions in chaospy if symmetric 
# TODO: different acceptance probability r
'''

def metropolis_hastings ( proposal_dist, target_dist, start = 0, chain_size = 100, symmetric = True ):  

    samples = np.zeros( chain_size )
    pi = target_dist(start)
    
    if symmetric:
        for i in range(chain_size):
            proposal = proposal_dist.sample(1)

            pi_new = target_dist(proposal)

            if pi_new > pi:
                alpha = 1
            elif pi_new < pi:
                alpha = pi_new/pi

            r = np.random.rand()
            if alpha > r:
                x = proposal
                pi = pi_new

            samples[i] = x
    
    return samples

###-------------------------------------------------------------------------###
### surrogate Likelihood for target distribution
###-------------------------------------------------------------------------###
def surrLikelihood ( x, data, surrogate, num_nodes, prior_dist ):
    
    D = data[0] - surrogate(x)[0]
    p = prior_dist(D)

    for i in range(num_nodes):
        D = data[i] - surrogate(x)[i]
        p = p * prior_dist(D)

    return p

###-------------------------------------------------------------------------###
### surrogate posterior distribution (for posteriori distribution)
###-------------------------------------------------------------------------###
def surrDist ( x, data, surrogate, num_nodes, prior_dist ):
# def surrDist ( x, data, surrogate, num_nodes, prior_dist ): # TODO: mit anderer dist

    p = ( surrLikelihood( x, data, surrogate, num_nodes, prior_dist ) * prior_dist ) 
    i = scipy.integrate.quad(p, -np.inf, np.inf) ### eine Quadratur. welche?

    ###Bayes
    posterior = p/i

    return posterior

