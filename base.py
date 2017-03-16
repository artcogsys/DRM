import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
import numpy as np

#####
## DRMNode; populations, readouts and connections are DRMNodes

class DRMNode(nn.Module):

    def __init__(self, n_in=1, n_out=1):
        """

        :param n_in: number/shape of inputs; scalar or numpy tensor
        :param n_out: number/shape of outputs; scalar or numpy tensor
        """

        super(DRMNode, self).__init__()

        self._n_in = n_in
        self._n_out = n_out

    def forward(self, x):
        """ Forward pass for this node

        :param x: input data
        :return: output data
        """

        raise NotImplementedError

    def reset_state(self):
        """ The function that is called when resetting internal state
        """

        pass

#####
## DRM; wrapper object that trains and analyses the model at hand

class DRM(object):

    def __init__(self, data_iter, populations, readout, ws, Wp, wr):
        """

        :param data_iter: iterator which generates sensations/responses at some specified resolution
        :param populations: list of populations (neural networks)
        :param readout: list of readout mechanisms
        :param ws: n_pop object array specifying a connection between the stimulus and each population.
        :param Wp: npop x npop object array specifying a connection between all populations
        :param wr: n_resp x npop object array specifying a readout mechanism for each internal population.
        """

        self.data_iter = data_iter

        self.model = DRMNet(populations, readout, ws, Wp, wr)

    def estimate(self):
        # here the estimation via truncated backprop takes place

        pass

#####
## DRMNet; the neural network which includes populations, connections and readouts

class DRMNet(nn.Sequential):

    def __init__(self, populations, readout, ws, Wp, wr):
        """ Each population receives either sensory input or input from other populations.
        This reception is mediated by connections which are neural networks themselves.
        There are three kinds of connections: sensory-population, population-population, population-response
        These all derive from the same object but we can have specific default implementations. Absent connections are
        represented as None.

        :param populations: npop list specifying each population
        :param readout: list of readout mechanisms
        :param ws: list specifying a connection between the stimulus and each population.
        :param Wp: npop x npop object array specifying a connection between all populations
        :param wr: list specifying a readout mechanism for each internal population.
        """

        self.n_pop = len(populations)

        # add populations
        self.populations = populations

        # add readouts
        self.readout = readout

        ### add connections

        self.ws = ws

        self.Wp = Wp

        self.wr = wr

        ### add components to nn.Sequence object

        _dict = OrderedDict()
        for i in range(self.n_pop):
            _dict['Population-' + str(i)] = populations[i]

        _dict['Readout'] = readout

        for i in range(self.n_pop):
            _dict['S-P-' + str(i)] = self.ws[i]

        wp = np.ravel(self.Wp)
        for i in range(self.Wp.size):
            if not wp[i] is None:
                _dict['P-P-' + str(i)] = wp[i]

        for i in range(self.n_pop):
            _dict['P-R-' + str(i)] = self.wr[i]

        super(DRMNet, self).__init__(_dict)

    def forward(self, x):
        """

        :param x: sensory input at this point in time (zeros for no input)
        :param train: whether we are in train or test mode (NOT USED FOR NOW)
        :return:
        """

        # initialize population outputs
        pop_output = [Variable(torch.zeros([x.shape[0], p.out_shape])) for p in self.populations]

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
