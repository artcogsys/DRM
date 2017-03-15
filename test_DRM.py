from iterators import DRMIterator
from population import DRMPopulation
from readout import DRMReadout
#from base import *
import numpy as np

#######
# Parameters

n_epochs = 150

#######
# Dataset

# define toy dataset with random data; this dataset has no coupling between (random) stimuli and responses!
# once the framework works; we can generate data from a constructed model and test parameter recovery

n_in = 5 # number of input stimuli
n_pop = 5 # number of assumed neural populations
n_out = 5 # number of output responses

# now chosen such as to have no missing data!

n_stim = 500 # number of stimuli presented
n_resp = 500 # number of responses recorded

stim_res = 1 # stimulus resolution
pop_res  = 2 # population (update) resolution
resp_res = 2 # response resolution

stim_offset = 0 # stimulus offset relative to start of population sampling
resp_offset = 0 # response offset relative to start of population sampling

# a stimulus is being presented every other sample
stimulus = np.random.randn(n_stim,n_in)
stim_time = np.arange(stim_offset,stim_offset + n_stim * stim_res, stim_res).tolist()

# a response is being recorded every other sample
response = np.random.randn(n_resp,n_out)
resp_time = np.arange(resp_offset,resp_offset + n_resp * resp_res, resp_res).tolist()

#######
# Iterator which generates stimuli and responses

data_iter = DRMIterator(stimulus, response, resolution=1, stim_time=stim_time, resp_time=resp_time, batch_size=32)

#######
# define model

# standard populations
populations = [DRMPopulation(out_shape=1) for i in range(n_pop)]

# standard readout mechanism
readout = DRMReadout(out_shape=n_out)

# setup model
#drm = DRM(data_iter, populations=populations, readout=readout, Ws=None, Wp=None, Wr=None)

# run DRM
#drm.run()

