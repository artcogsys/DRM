from iterators import DRMIterator
from population import DRMPopulation
from readout import DRMReadout
from connection import DRMConnection
from base import DRM, DRMNet
import numpy as np
import matplotlib.pyplot as plt

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

# link stimulus to the first population only to ensure identifiability
# ws = [DRMConnection(n_in=n_stim, n_out=n_stim)] + [None for i in range(n_pop-1)]

# create full population matrix
Wp = np.array([DRMConnection(n_in=n_pop_out, n_out=n_pop_out) for i in range(n_pop * n_pop)]).reshape([n_pop, n_pop])

# remove self connections (these can be handled internally by e.g. an RNN)
for i in range(n_pop):
    Wp[i,i] = None

# set up ground truth model
drm_net = DRMNet(populations, ws, Wp, readout)

#######
# Generate responses based on sensory input when running the model in forward mode

# drm is responsible for running the network
drm = DRM(drm_net)

# data used for model estimation
stimulus1 = np.random.randn(stim_len, n_stim)
data_iter = DRMIterator(resolution=1, stimulus=stimulus1, stim_time=stim_time, batch_size=1)
response_gt, _ = drm.forward(data_iter)

# data used for model validation
stimulus2 = np.random.randn(stim_len, n_stim)
data_iter = DRMIterator(resolution=1, stimulus=stimulus2, stim_time=stim_time, batch_size=1)
response2, activity2 = drm.forward(data_iter)

#######
# Iterator which generates stimuli and responses

data_iter = DRMIterator(resolution=1, stimulus=stimulus1, stim_time=stim_time, response=response_gt, resp_time=resp_time, batch_size=32)
val_iter = DRMIterator(resolution=1, stimulus=stimulus2, stim_time=stim_time, response=response2, resp_time=resp_time, batch_size=32)

#######
# define model used for parameter inference - same structure as ground truth model but different initial parameters

# standard populations
populations = [DRMPopulation(n_in=n_pop_in, n_out=n_pop_out) for i in range(n_pop)]

# standard readout mechanism - receives output from all populations
readout = DRMReadout(n_in = n_pop * n_pop_out, n_out=n_resp)

# link stimulus to all populations; each population receives the (delayed) stimulus input
ws = [DRMConnection(n_in=n_stim, n_out=n_stim) for i in range(n_pop)]

# link stimulus to the first population only to ensure identifiability
# ws = [DRMConnection(n_in=n_stim, n_out=n_stim)] + [None for i in range(n_pop-1)]

# create full population matrix
Wp = np.array([DRMConnection(n_in=n_pop_out, n_out=n_pop_out) for i in range(n_pop * n_pop)]).reshape([n_pop, n_pop])

# remove self connections (these can be handled internally by e.g. an RNN)
for i in range(n_pop):
    Wp[i,i] = None

# set up estimation model
drm_net2 = DRMNet(populations, ws, Wp, readout)

drm2 = DRM(drm_net2)

#######
# create new data and compute population activity for real model and initial model

test_stim = np.random.randn(stim_len, n_stim)
test_iter = DRMIterator(resolution=1, stimulus=test_stim, stim_time=stim_time, batch_size=1)
response_gt, activity_gt = drm.forward(test_iter)
response_init, activity_init = drm2.forward(test_iter)

# compute correlations between population activity of real and initial model
c1 = np.corrcoef(np.hstack([activity_gt, activity_init]).transpose())
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

response_estim, activity_estim = drm2.forward(test_iter)

# compute correlations between population activity of real and initial model
c2 = np.corrcoef(np.hstack([activity_gt, activity_estim]).transpose())
c2 = [c2[i,i+n_pop] for i in range(n_pop)]

#######
# compute correlation between population responses for initial and estimated model wrt ground truth model

print 'correlations for initial model:'
print c1
print 'correlations for estimated model:'
print c2

#######
# plot population activity - first 100 datapoints

for i in range(n_pop):

    x = np.vstack([activity_gt[:, i], activity_init[:, i], activity_estim[:, i]]).T

    plt.subplot(n_pop,1,i)
    plt.plot(x[:100])
    plt.title('population {0}; R={1}'.format(i, np.corrcoef(x.T)[0,1]))
    plt.legend(['ground truth', 'initial', 'estimate'])

plt.show()


