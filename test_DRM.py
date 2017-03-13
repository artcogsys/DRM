from base import *
from population import DRMPopulation
from readout import DRMReadout

#######
# Parameters

n_epochs = 150

#######
# Dataset

# define toy dataset; this dataset has no coupling between (random) stimuli and responses!
# once the framework works; we can generate data from a constructed random model

n_in = 5 # number of input stimuli
n_pop = 5 # number of assumed neural populations
n_out = 5 # number of output responses

# now chosen such as to have no missing data!

n_stim = 1000 # number of stimuli presented
n_resp = 1000 # number of responses recorded

stim_res = 1 # stimulus resolution
pop_res  = 1 # population (update) resolution
resp_res = 1 # response resolution

stim_offset = 0 # stimulus offset
resp_offset = 0 # response offset

stimulus = np.random.randn(n_stim,n_in)
stim_time = np.arange(stim_offset,stim_offset + n_stim * stim_res, stim_res).tolist()

response = np.random.randn(n_resp,n_out)
resp_time = np.arange(resp_offset,resp_offset + n_resp * resp_res, resp_res).tolist()

#######
# Iterator

data_iter = DRMIterator(stimulus, response, resolution=1, stim_time=stim_time, resp_time=resp_time, batch_size=32)

#######
# define model

# standard populations
populations = [DRMPopulation(out_shape=1) for i in range(n_pop)]

# standard readout mechanism
readout = DRMReadout(out_shape=n_out)

# setup model
drm = DRM(data_iter, populations=populations, readout=readout, Ws=None, Wp=None, Wr=None)

# run DRM
drm.run()

