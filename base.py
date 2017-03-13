# Implementation of dynamic representational modeling. We assume the existence of a recurrent network containing the
# 'representations'. Each neural population has its own (set of) RNN units. Each population projects to one output
# variable (e.g. its BOLD response).

from agent.supervised import StatefulAgent
from brain.models import *
from brain.networks import *
from world.base import World
from world.data import *
from connection import DRMConnection
from population import DRMPopulation
from readout import DRMReadout

#####
## DRM iterator

class DRMIterator(Iterator):

    def __init__(self, stimulus, response, resolution=None, stim_time=None, resp_time=None, batch_size=None, n_batches=None):
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

        # assume sampling at same rate of 1 ms if nothing is specified
        self.resolution = resolution or 1

        if stim_time is None:
            self.stim_time = np.arange(0, len(stimulus)).tolist()
        else:
            self.stim_time = stim_time

        if resp_time is None:
            self.resp_time = np.arange(0, len(response)).tolist()
        else:
            self.resp_time = resp_time

        # check if lengths agree
        assert(len(self.stimulus) == len(self.stim_time))
        assert(len(self.response) == len(self.resp_time))

        # steps in the simulation should be at least as small as the minimal step in stimulus or response time
        assert(np.min(np.diff(self.stim_time)) >= self.resolution)
        assert(np.min(np.diff(self.resp_time)) >= self.resolution)

        # determine total number of time steps to take according to temporal resolution
        self.n_steps = np.ceil(np.max(self.stim_time + self.resp_time) / self.resolution).astype('int32')

        if batch_size is None:
            batch_size = 1

        if n_batches is None:
            n_batches = self.n_steps // batch_size

        super(DRMIterator, self).__init__(batch_size=batch_size, n_batches=n_batches)

    def __iter__(self):

        self.idx = 0

        offsets = [i * self.n_batches for i in range(self.batch_size)]

        # determine time points to sample
        self._order = []
        for iter in range(self.n_batches):
            x = [(offset + iter) % self.n_steps for offset in offsets]
            self._order += x

        # multiply points by temporal resolution
        self._order *= self.resolution

        return self

    def next(self):

        if self.idx == self.n_batches:
            raise StopIteration

        i = self.idx * self.batch_size

        self.idx += 1

        sample_times = self._order[i:(i + self.batch_size)]

        # find closest time point in sensory stream
        idx = map(lambda t: np.where((self.stim_time >= t - self.resolution/2) & (self.stim_time <= t + self.resolution/2))[0], sample_times)

        # create partially observed data (zeros for no input)
        stim_data = np.array(map(lambda x: np.zeros(self.stimulus[0].shape) if len(x) == 0 else self.stimulus[x[0]], idx)).astype('float32')

        # find closest time point in response stream
        idx = map(lambda t: np.where((self.resp_time >= t - self.resolution/2) & (self.resp_time <= t + self.resolution/2))[0], sample_times)

        # create partially observed data (nans for no output)
        resp_data = np.array(map(lambda x: np.full(self.response[0].shape, np.nan) if len(x) == 0 else self.response[x[0]], idx)).astype('float32')

        return [stim_data, resp_data]

    def process(self, agent):
        pass

#####
## DRM

# focus: fast/scalable

class DRM(object):

    def __init__(self, data_iter, populations, readout, Ws=None, Wp=None, Wr=None):
        """

        :param data_iter: iterator which generates sensations/responses at some specified resolution
        :param populations: list of populations (neural networks)
        :param readout: list of readout mechanisms
        :param Ws: n_pop object array specifying a connection between the stimulus and each population.
        :param Wp: npop x npop object array specifying a connection between all populations
        :param Wr: n_resp x npop object array specifying a readout mechanism for each internal population.
        """

        self.data_iter = data_iter

        self.model = Regressor(DRMNet(populations, readout, Ws, Wp, Wr))

    def run(self):

        # define agent
        agent = StatefulAgent(self.model, chainer.optimizers.Adam())

        # add hook
        agent.optimizer.add_hook(chainer.optimizer.WeightDecay(1e-5))

        # define world
        world = World(agent)

        # run world in training mode with validation
        world.train(self.data_iter, n_epochs=100, plot=-1)

class DRMNet(ChainList, Network):

    def __init__(self, populations, readout, Ws, Wp, Wr):
        """ Each population receives either sensory input or input from other populations.
        This reception is mediated by connections which are neural networks themselves.
        There are three kinds of connections: sensory-population, population-population, population-response
        These all derive from the same object but we can have specific default implementations. Absent connections are
        represented as None.

        :param populations: npop list specifying each population
        :param readout: list of readout mechanisms
        :param Ws: list specifying a connection between the stimulus and each population.
        :param Wp: npop x npop object array specifying a connection between all populations
        :param Wr: list specifying a readout mechanism for each internal population.
        """

        self.n_pop = len(populations)

        # add populations
        self.populations = populations

        # add readouts
        self.readout = readout

        ### add connections and inialize to defaults if needed

        self.Ws = Ws or [DRMConnection() for i in range(self.n_pop)]

        if Wp is None:

            # add default connections
            self.Wp = np.array([DRMConnection() for i in range(self.n_pop * self.n_pop)]).reshape([self.n_pop, self.n_pop])

            # remove self connections (these are handled internall by e.g. an RNN)
            for i in range(self.n_pop):
                self.Wp[i,i] = None

        else:

            self.Wp = Wp

        self.Wr = Wr or [DRMConnection() for i in range(self.n_pop)]

        ### add links

        links = ChainList()

        for i in range(self.n_pop):
            links.add_link(populations[i])

        links.add_link(readout)

        for i in range(self.n_pop):
            links.add_link(self.Ws[i])

        wp = np.ravel(self.Wp)
        for i in range(self.Wp.size):
            if not wp[i] is None:
                links.add_link(wp[i])

        for i in range(self.n_pop):
            links.add_link(self.Wr[i])

        super(DRMNet, self).__init__(links)

    def __call__(self, x, train=False):
        """

        :param x: sensory input at this point in time (zeros for no input)
        :param train: whether we are in train or test mode
        :return:
        """

        # initialize population outputs
        pop_output = [Variable(np.zeros([x.shape[0], p.out_shape]).astype('float32')) for p in self.populations]

        # randomly update each population
        for i in np.random.permutation(self.n_pop):

            # pass sensory input to the sensory-population connection associated with this population
            # the result is the sensory input for that population (e.g. delayed input)
            ws = self.Ws[i]
            if ws is None:
                pop_input = []
            else:
                pop_input = [self.Ws[i](x)]

            # get population connections entering this population and pass other populations output through the connections
            # note that population outputs can be zero or not depending on whether they have been updated already
            wp = self.Wp[i]
            for j in range(self.n_pop):
                if not wp[j] is None:
                    pop_input.append(wp[j](pop_output[j]))

            # compute population output for this population
            # population receives the concatenated inputs of sensation and incoming populations
            pop_output[i] = self.populations[i](F.concat(pop_input))

        # now we have all the outputs, we can pass it to the readout mechanism
        return self.readout(F.concat(pop_output))
