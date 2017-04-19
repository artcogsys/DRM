#######
# Test recovery of parameters for one population with fixed readout and no missing data (except first sample)
# test_drm1.py contains a very basic example to check if learning works optimally. It shows that it can recover the
# population activity for one population which gets a sensory input of length 3 and where the measurements are a direct copy of
# the population activity. Population activity itself is delayed relative to the stimulus by one time step (conductance delay).
# This example shows that we can recover the population activity after training.

from iterators import DRMIterator
from population import DRMPopulation
from connection import DRMConnection
from base import DRM, DRMNet
from readout import DRMReadout
import numpy as np
import matplotlib.pyplot as plt

#######
# Parameters

n_epochs = 100

#######
# Dataset

# define toy dataset with random data; this dataset has no coupling between (random) stimuli and responses!
# once the framework works; we can generate data from a constructed model and test parameter recovery

n_stim = 3  # number/shape of input stimuli
n_pop = 1   # number of assumed neural populations
n_resp = 1  # number/shape of output responses

stim_len = 500 # number of stimuli presented
resp_len = 500 # number of responses recorded

stim_res = 1 # stimulus resolution
pop_res  = 1 # population (update) resolution
resp_res = 1 # response resolution

stim_offset = 0 # stimulus offset relative to start of population sampling
resp_offset = 0 # response offset relative to start of population sampling

# actual times at which stimuli/responses are sampled
stim_time = np.arange(stim_offset, stim_offset + stim_len * stim_res, stim_res).tolist()
resp_time = np.arange(resp_offset, resp_offset + resp_len * resp_res, resp_res).tolist()

def create_model():
    """ 
    Helper function to create a DRMNet with a particular structure. Used to create multiple instances with different 
    initial parameters
    
    :return: DRMNet object
    """

    # standard populations
    populations = [DRMPopulation() for i in range(n_pop)]

    readout = DRMReadout()

    # link stimulus to all populations; each population receives the (delayed) stimulus input
    ws = [DRMConnection() for i in range(n_pop)]

    # create full population matrix
    Wp = np.array([None]).reshape([n_pop, n_pop])
    for i in range(n_pop):
        for j in range(n_pop):
            if i != j:
                Wp[i,j] = DRMConnection()

    # set up ground truth model
    drm_net = DRMNet(populations, ws, Wp, readout)

    return drm_net

#######
# define ground truth model used for forward simulation

drm = DRM(create_model())

#######
# Generate responses based on sensory input when running the ground truth model in forward mode

# data used for model estimation
stimulus1 = np.random.randn(stim_len, n_stim)
data_iter = DRMIterator(resolution=1, stimulus=stimulus1, stim_time=stim_time, batch_size=1)
response_gt, _ = drm(data_iter)

# data used for model validation
stimulus2 = np.random.randn(stim_len, n_stim)
data_iter = DRMIterator(resolution=1, stimulus=stimulus2, stim_time=stim_time, batch_size=1)
response2, activity2 = drm(data_iter)

# iterators which generate stimuli and responses
data_iter = DRMIterator(resolution=1, stimulus=stimulus1, stim_time=stim_time, response=response_gt, resp_time=resp_time, batch_size=32)
val_iter = DRMIterator(resolution=1, stimulus=stimulus2, stim_time=stim_time, response=response2, resp_time=resp_time, batch_size=32)

#######
# define model used for parameter inference - same structure as ground truth model but different initial parameters

drm2 = DRM(create_model())

#######
# create new data and compute population activity for real model and initial model

test_stim = np.random.randn(stim_len, n_stim)
test_iter = DRMIterator(resolution=1, stimulus=test_stim, stim_time=stim_time, batch_size=1)

# ground truth responses and population activity for this test data
response_gt, activity_gt = drm(test_iter)

# responses and population activity for this test data for untrained initial model
response_init, activity_init = drm2(test_iter)

# compute MSE between population activity of real and initial model
c1 = []
for i in range(n_pop):
    c1.append(((np.squeeze(activity_gt[:,i]) - np.squeeze(activity_init[:,i])) ** 2).mean(axis=0))

#######
# estimate model

# run DRM
train_loss, validation_loss = drm2.estimate(data_iter, val_iter, n_epochs)

#######
# check decrease in loss

plt.figure()
plt.plot(train_loss)
plt.hold(True)
plt.plot(validation_loss)
plt.legend(['training loss', 'validation loss'])

#######
# compute responses and population activity for estimated model

response_estim, activity_estim = drm2(test_iter)

# compute MSE between population activity of real and estimated model
c2 = []
for i in range(n_pop):
    c2.append(((np.squeeze(activity_gt[:,i]) - np.squeeze(activity_estim[:,i])) ** 2).mean(axis=0))

#######
# compute MSE between population responses for initial and estimated model wrt ground truth model

print 'MSE for initial model:'
print c1
print 'MSE for estimated model:'
print c2

#######
# plot population activity - first 100 datapoints

plt.figure()
for i in range(n_pop):

    x = np.vstack([activity_gt[:, i], activity_init[:, i], activity_estim[:, i]]).T

    plt.subplot(n_pop,1,i+1)
    plt.plot(x[:100])
    plt.title('population {0}; initial MSE={1:4.3f}; estim MSE={2:4.3f}'.format(i, c1[i], c2[i]))
    plt.legend(['ground truth', 'initial', 'estimate'])

plt.show()


