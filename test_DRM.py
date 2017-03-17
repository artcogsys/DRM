from iterators import DRMIterator
from population import DRMPopulation
from readout import DRMReadout
from connection import DRMConnection
from base import DRM
import numpy as np

#######
# Parameters

n_epochs = 150

#######
# Dataset

# define toy dataset with random data; this dataset has no coupling between (random) stimuli and responses!
# once the framework works; we can generate data from a constructed model and test parameter recovery

n_stim = 5 # number/shape of input stimuli
n_pop = 5 # number of assumed neural populations
n_resp = 5 # number/shape of output responses

# now chosen such as to have no missing data!

stim_len = 500 # number of stimuli presented
resp_len = 500 # number of responses recorded

stim_res = 1 # stimulus resolution
pop_res  = 2 # population (update) resolution
resp_res = 2 # response resolution

stim_offset = 0 # stimulus offset relative to start of population sampling
resp_offset = 0 # response offset relative to start of population sampling

# a stimulus is being presented every other sample
stimulus = np.random.randn(stim_len, n_stim)
stim_time = np.arange(stim_offset, stim_offset + stim_len * stim_res, stim_res).tolist()

# a response is being recorded every other sample
response = np.random.randn(resp_len, n_resp)
resp_time = np.arange(resp_offset, resp_offset + resp_len * resp_res, resp_res).tolist()

#######
# Iterator which generates stimuli and responses

data_iter = DRMIterator(stimulus, response, resolution=1, stim_time=stim_time, resp_time=resp_time, batch_size=32)

#######
# define model

# standard populations
n_pop_out = 1
n_pop_in = n_stim + (n_pop-1) * n_pop_out # stimulus input plus scalar output from all other populations
populations = [DRMPopulation(n_in=n_pop_in, n_out=n_pop_out) for i in range(n_pop)]

# standard readout mechanism - receives output from all populations
readout = DRMReadout(n_in = n_pop * n_pop_out, n_out=n_resp)

# link stimulus to all populations; each population receives the (delayed) stimulus input
ws = [DRMConnection(n_in=n_stim, n_out=n_stim) for i in range(n_pop)]

# create full population matrix
Wp = np.array([DRMConnection(n_in=n_pop_out, n_out=n_pop_out) for i in range(n_pop * n_pop)]).reshape([n_pop, n_pop])

# remove self connections (these can be handled internally by e.g. an RNN)
for i in range(n_pop):
    Wp[i,i] = None

# link populations to the response
wr = [DRMConnection(n_in=n_pop_out, n_out=n_pop_out) for i in range(n_pop)]

# setup model
drm = DRM(populations=populations, readout=readout, ws=ws, Wp=Wp, wr=wr)

# debugging
for data in data_iter:
    drm.model.forward(data)

# run DRM
drm.estimate(data_iter)

