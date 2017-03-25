from iterators import DRMIterator
from population import DRMPopulation
from readout import DRMReadout
from connection import DRMConnection
from base import DRM
import numpy as np

#######
# Parameters

n_epochs = 200

#######
# Dataset

# define toy dataset with random data; this dataset has no coupling between (random) stimuli and responses!
# once the framework works; we can generate data from a constructed model and test parameter recovery

n_stim = 5 # number/shape of input stimuli
n_pop = 3 # number of assumed neural populations
n_resp = 5 # number/shape of output responses

# now chosen such as to have no missing data!

stim_len = 500 # number of stimuli presented
resp_len = 500 # number of responses recorded

stim_res = 1 # stimulus resolution
pop_res  = 2 # population (update) resolution
resp_res = 2 # response resolution

stim_offset = 0 # stimulus offset relative to start of population sampling
resp_offset = 0 # response offset relative to start of population sampling

# actual times at which stimuli/responses are sampled
stim_time = np.arange(stim_offset, stim_offset + stim_len * stim_res, stim_res).tolist()
resp_time = np.arange(resp_offset, resp_offset + resp_len * resp_res, resp_res).tolist()

# population parameters
n_pop_out = 1
n_pop_in = n_stim + (n_pop-1) * n_pop_out # stimulus input plus scalar output from all other populations

#######
# define model used for forward simulation

# standard populations
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

# setup model
drm = DRM(populations=populations, ws=ws, Wp=Wp, readout=readout)

#######
# Generate responses based on sensory input when running the model in forward mode

stimulus1 = np.random.randn(stim_len, n_stim)
data_iter = DRMIterator(resolution=1, stimulus=stimulus1, stim_time=stim_time, batch_size=1)
response1, _ = drm.forward(data_iter)

stimulus2 = np.random.randn(stim_len, n_stim)
data_iter = DRMIterator(resolution=1, stimulus=stimulus2, stim_time=stim_time, batch_size=1)
response2, activity2 = drm.forward(data_iter)

#######
# Iterator which generates stimuli and responses

data_iter = DRMIterator(resolution=1, stimulus=stimulus1, stim_time=stim_time, response=response1, resp_time=resp_time, batch_size=32)
val_iter = DRMIterator(resolution=1, stimulus=stimulus2, stim_time=stim_time, response=response2, resp_time=resp_time, batch_size=32)

#######
# define model used for parameter inference

# standard populations
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

# setup model
drm2 = DRM(populations=populations, ws=ws, Wp=Wp, readout=readout)

#######
# create new data and compute population activity for real model and initial model

test_stim = np.random.randn(stim_len, n_stim)
test_iter = DRMIterator(resolution=1, stimulus=test_stim, stim_time=stim_time, batch_size=1)
response1, activity1 = drm.forward(test_iter)
response2, activity2 = drm2.forward(test_iter)

# compute correlations between population activity of real and initial model
c1 = np.corrcoef(np.hstack([activity1, activity2]).transpose())
c1 = [c1[i,i+n_pop] for i in range(n_pop)]

#######
# estimate model

# run DRM
train_loss, validation_loss = drm2.estimate(data_iter, val_iter, n_epochs)

#######
# check decrease in loss

print 'validation loss:'
print validation_loss

#######
# compute population activity for estimated model

response2, activity2 = drm2.forward(test_iter)

# compute correlations between population activity of real and initial model
c2 = np.corrcoef(np.hstack([activity1, activity2]).transpose())
c2 = [c2[i,i+n_pop] for i in range(n_pop)]

# print correlations between model and actual population response for initial and estimated model
print 'correlations for initial model:'
print c1
print 'correlations for estimated model:'
print c2
