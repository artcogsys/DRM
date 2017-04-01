import numpy as np
import torch
from torch.autograd import Variable

#####
## DRM iterator

class DRMIterator(object):

    def __init__(self, resolution, stimulus, stim_time, response=None, resp_time=None, batch_size=None, n_batches=None):
        """ Initializer

        Generates stimulus and response outputs. The data stream is sampled at a particular sampling rate. At each
        point in time, stimuli and/or responses can be either present or absent depending on the stim_time and resp_times.
        This leads to a partially observed stream both on the input and output side. The stream can generate data in batch
        mode. Only generates stimulus input in case response=None which can be used for forward simulation.

        :param response: output responses - nsamples x d1 x ... numpy array (float32)
        :param stimulus: input stimulus - nsamples x d1 x ... numpy array (float32)
        :param stim_time: list of times in ms at which stimuli were presented relative to start of simulation t=0
        :param resolution: temporal resolution for simulation in ms
        :param resp_time: list of times in ms at which responses were observed relative to start of simulation t=0
        :param batch_size: number of batches to process sequentially
        :param n_batches: number of time steps to take per batch
        """

        # assumed sampling rate of population responses in ms
        self.resolution = resolution

        # set stimulus
        self.stimulus = stimulus
        self.n_in = self.stimulus[0].size

        # times at which stimuli were presented
        self.stim_time = stim_time

        # check if lengths agree
        assert(len(self.stimulus) == len(self.stim_time))

        # steps in the simulation should be at least as small as the minimal step in stimulus or response time
        assert(np.min(np.diff(self.stim_time)) >= self.resolution)

        self.response = response

        if not self.response is None:

            self.n_out = self.response[0].size

            self.resp_time = resp_time

            assert(len(self.response) == len(self.resp_time))
            assert(np.min(np.diff(self.resp_time)) >= self.resolution)

            # determine total number of time steps to take according to temporal resolution
            self.n_steps = np.ceil((np.max(self.stim_time + self.resp_time) + 1) / self.resolution).astype('int32')

        else:

            self.n_out = None
            self.resp_time = None

            # determine total number of time steps to take according to temporal resolution
            self.n_steps = np.ceil((np.max(self.stim_time)+1) / self.resolution).astype('int32')

        # by default we run once through the whole dataset
        if batch_size is None:
            batch_size = 1

        # division into number of batches (time steps in terms of population updates)
        if n_batches is None:
            n_batches = self.n_steps // batch_size

        self.batch_size = batch_size
        self.n_batches = n_batches

    def __iter__(self):
        """ Initializes data generator. Should be invoked at the start of each epoch

        :return: self
        """

        self.idx = 0

        # select batch indices at which to start sampling
        offsets = [i * self.n_batches for i in range(self.batch_size)]

        # determine time points to sample
        self._order = []
        for iter in range(self.n_batches):
            x = [(offset + iter) % self.n_steps for offset in offsets]
            self._order += x

        return self

    def next(self):
        """Produces next data item

        :return: dictionary containing the stimulus and the response as torch variables
        """

        if self.idx == self.n_batches:
            raise StopIteration

        i = self.idx * self.batch_size

        sample_times = self._order[i:(i + self.batch_size)]

        # find closest time point in sensory stream
        idx = map(lambda t: np.where((self.stim_time >= t - self.resolution/2) & (self.stim_time <= t + self.resolution/2))[0], sample_times)

        # create partially observed data (zeros for no input)
        stim_data = np.array(map(lambda x: np.zeros(self.stimulus[0].shape) if len(x) == 0 else self.stimulus[x[0]], idx)).astype('float32')

        data = {}
        data['stimulus'] = Variable(torch.from_numpy(stim_data))

        if not self.response is None:

            # find closest time point in response stream
            idx = map(lambda t: np.where((self.resp_time >= t - self.resolution/2) & (self.resp_time <= t + self.resolution/2))[0], sample_times)

            # create partially observed data (nans for no output)
            resp_data = np.array(map(lambda x: np.full(self.response[0].shape, np.nan) if len(x) == 0 else self.response[x[0]], idx)).astype('float32')

            data['response'] = Variable(torch.from_numpy(resp_data))

        self.idx += 1

        return data

    def is_final(self):
        """Flags if final iteration is reached

        :return: boolean if final batch is reached
        """

        return (self.idx==self.n_batches)