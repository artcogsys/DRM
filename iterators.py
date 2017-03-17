import numpy as np

#####
## DRM iterator

class DRMIterator(object):

    def __init__(self, stimulus, response, resolution, stim_time, resp_time, batch_size=None, n_batches=None):
        """

        :param stimulus: input stimulus - nsamples x d1 x ... numpy array (float32)
        :param response: output responses - nsamples x d1 x ... numpy array (float32)
        :param resolution: temporal resolution for simulation in ms
        :param stim_time: list of times in ms at which stimuli were presented relative to start of simulation t=0
        :param resp_time: list of times in ms at which responses were observed relative to start of simulation t=0
        :param batch_size: number of batches to process sequentially
        :param n_batches: number of time steps to take per batch
        """

        self.stimulus = stimulus
        self.response = response

        self.n_in = self.stimulus[0].size
        self.n_out = self.response[0].size

        # assumed sampling rate of population responses in ms
        self.resolution = resolution

        # times at which stimuli were presented
        self.stim_time = stim_time

        # times at which responses were presented
        self.resp_time = resp_time

        # check if lengths agree
        assert(len(self.stimulus) == len(self.stim_time))
        assert(len(self.response) == len(self.resp_time))

        # steps in the simulation should be at least as small as the minimal step in stimulus or response time
        assert(np.min(np.diff(self.stim_time)) >= self.resolution)
        assert(np.min(np.diff(self.resp_time)) >= self.resolution)

        # determine total number of time steps to take according to temporal resolution
        self.n_steps = np.ceil(np.max(self.stim_time + self.resp_time) / self.resolution).astype('int32')

        # by default we run once through the whole dataset
        if batch_size is None:
            batch_size = 1

        # division into number of batches (time steps in terms of population updates)
        if n_batches is None:
            n_batches = self.n_steps // batch_size

        self.batch_size = batch_size
        self.n_batches = n_batches

    def __iter__(self):

        self.idx = -1

        # select batch indices at which to start sampling
        offsets = [i * self.n_batches for i in range(self.batch_size)]

        # determine time points to sample
        self._order = []
        for iter in range(self.n_batches):
            x = [(offset + iter) % self.n_steps for offset in offsets]
            self._order += x

        return self

    def next(self):
        """

        :return: a dictionary containing the stimulus and the response batches
        """

        if self.idx == self.n_batches-1:
            raise StopIteration

        self.idx += 1

        i = self.idx * self.batch_size

        sample_times = self._order[i:(i + self.batch_size)]

        # find closest time point in sensory stream
        idx = map(lambda t: np.where((self.stim_time >= t - self.resolution/2) & (self.stim_time <= t + self.resolution/2))[0], sample_times)

        # create partially observed data (zeros for no input)
        stim_data = np.array(map(lambda x: np.zeros(self.stimulus[0].shape) if len(x) == 0 else self.stimulus[x[0]], idx)).astype('float32')

        # find closest time point in response stream
        idx = map(lambda t: np.where((self.resp_time >= t - self.resolution/2) & (self.resp_time <= t + self.resolution/2))[0], sample_times)

        # create partially observed data (nans for no output)
        resp_data = np.array(map(lambda x: np.full(self.response[0].shape, np.nan) if len(x) == 0 else self.response[x[0]], idx)).astype('float32')

        data = {}
        data['stimulus'] = stim_data
        data['response'] = resp_data

        return data

    def process(self, agent):
        pass