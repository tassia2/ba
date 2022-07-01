import sys
import numpy as np
import mc3
from configparser import ConfigParser
import random
import numpoly
import os
import shutil
from pathlib import Path
import re


class ChaosPolynomials:
    def __init__(self,polyfolder,points) -> None:
        x = self.polyarray
        for point in points:
            x.append(numpoly.load(os.path.join(polyfolder,str(point[0]*65+point[1])+".npy")))
    
    @classmethod
    def complete_grid(self,polyfolder) -> None:
        x = self.polymatrix
        for i in range(65):
            x.append([])
            for j in range(65):
                x[i].append(numpoly.load(os.path.join(polyfolder,str(i*65+j)+".npy")))

    def eval(self,params):
        result = []
        for poly in self.polyarray:
            result.append(poly(params[0], params[1]))
        return np.array(result)


    polymatrix = []
    polyarray = []
    

def GetConfig(str):
    Config = ConfigParser()
    Config.read(str)
    dict1 = {}
    try:
        options = Config.options("MCMC")
    except Exception as e: 
        print(e)
        print("\nscript needs to run in the console inside the same directory as mcmc.config, or else change the hard coded path for mcmc.config.")
        exit()
    for option in options:
        try:
            dict1[option] = Config.get("MCMC", option)
            if dict1[option] == -1:
                print("skip: %s" % option)
        except:
            print("exception on %s!" % option)
            dict1[option] = None
    return dict1
    
# def posterior(x):
#     return likelihood(x)*prior(x)

# def likelihood(x):
#     return np.exp((x - func))


# Get Configurations
config = GetConfig("./mcmc.config")
# copy config file
Path(config['savepath']).mkdir(parents=True, exist_ok=True)
shutil.copy("./mcmc.config",os.path.join(config['savepath'],'mcmc.config'))

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# Array of initial-guess values of fitting parameters:
params = np.array([ float(config['prandtl']), float(config['rayleigh'])])
nsamples=int(float(config['nsamples']))
ncpu=int(config['ncpu'])

# Preamble (create a synthetic dataset, in a real scenario you would
# get your dataset from your own data analysis pipeline):
if config['points'] == 'None' or config['points'] == '[]' or config['points'] == '[[]]' :
    x = np.array(random.sample(range(3,62), int(config['npoints'])))
    y = np.array(random.sample(range(3,62), int(config['npoints'])))
    points = np.hstack((x[None].T,y[None].T))
    print('taking meassurements at \n', points)
else:
    points = np.array(eval(config['points']))

uncert = np.array([float(config['noise'])]* points.size)

# Load polynomial grid, a matrix made out of the chaos polynomials at every point, created by vtk_postpressing.py
cp_polynomials = ChaosPolynomials(config['poly_folder'],points)
messurements = np.random.normal(cp_polynomials.eval(params),float(config['noise']))

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# # Define the modeling function as a callable:
# func = quad

# List of additional arguments of func (if necessary):
indparams = []

# Lower and upper boundaries for the MCMC exploration:
pmin = [float(x) for x in re.sub(' +',',',config['pmin']).split(',')]
pmax = [float(x) for x in re.sub(' +',',',config['pmax']).split(',')]
# Parameters' stepping behavior:
pstep = [float(x) for x in re.sub(' +',',',config['pstep']).split(',')]

# Parameter prior probability distributions - see Documentation of mc3
# a priorlow  of 0 sets a uniform prior between the pmin and pmax values
prior    = np.array([ 0.0, 0.0])
priorlow = np.array([ 0.0, 0.0])
priorup  = np.array([ 0.0, 0.0])

# Parameter names:
pnames   = ['prandtl-number', 'rayeigh-number']
texnames = ['prandtl-number', 'rayeigh-number']

# Sampler algorithm, choose from: 'snooker', 'demc' or 'mrw'.
sampler = 'mrw'

# MCMC initial draw, choose from: 'normal' or 'uniform', how the initial selection is done, normal distributed or uniform
kickoff = 'normal'
burnin=int(config['burnin'])
thinning=int(config['thinning'])
nsamples=int(float(config['nsamples']))

# Optimization before MCMC, choose from: 'lm' or 'trf':
leastsq    = 'lm'
chisqscale = False

# MCMC Convergence test:
grtest  = (config['grtest'] == 'True' or config['grtest'] == 't' or config['grtest'] == 'true')
# For a value smaller than 1.01 convergence is reached (most of the time). Will stop the algorithm.
grbreak = 1.01
# percentage of nsamples need to run bevor brake possible
grnmin  = 0.5

# Logging:
log = os.path.join(config['savepath'],'MCMC.log')

plots=(config['plots'] == 'True' or config['plots'] == 't' or config['plots'] == 'true')

# File outputs:
savefile = os.path.join(config['savepath'],'MCMC.npz')
rms      = False

# Carter & Winn (2009) Wavelet-likelihood method:
wlike = False

# Run the MCMC:
mc3_output = mc3.sample(data=messurements, uncert=uncert, func=cp_polynomials.eval, params=params,
     pmin=pmin, pmax=pmax, pstep=pstep,
     pnames=pnames, texnames=texnames,
     prior=prior, priorlow=priorlow, priorup=priorup,
     sampler=sampler, nsamples=nsamples, nchains=12,
     ncpu=ncpu, burnin=burnin, thinning=thinning,
     leastsq=leastsq, chisqscale=chisqscale,
     grtest=grtest, grbreak=grbreak, grnmin=grnmin,
     kickoff=kickoff, indparams=indparams,
     wlike=wlike, log=log,
     plots=plots, savefile=savefile, rms=rms)


