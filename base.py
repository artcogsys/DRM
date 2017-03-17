import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
import numpy as np
import tqdm

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

    def reset(self):
        """ The function that is called when resetting internal state
        """

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

        ### add components to nn.Sequence object

        _dict = OrderedDict()
        for i in range(self.n_pop):
            _dict['Population-' + str(i)] = populations[i]

        _dict['Readout'] = readout

        for i in range(self.n_pop):
            _dict['S-P-' + str(i)] = ws[i]

        wp = np.ravel(Wp)
        for i in range(Wp.size):
            if not wp[i] is None:
                _dict['P-P-' + str(i)] = wp[i]

        for i in range(self.n_pop):
            _dict['P-R-' + str(i)] = wr[i]

        super(DRMNet, self).__init__(_dict)

        # add populations
        self.populations = populations

        # add readouts
        self.readout = readout

        ### add connections

        self.ws = ws

        self.Wp = Wp

        self.wr = wr

    def forward(self, x):
        """

        :param x: sensory input at this point in time (zeros for no input)
        :param train: whether we are in train or test mode (NOT USED FOR NOW)
        :return:
        """

        batch_size = x['stimulus'].shape[0]

        # initialize population outputs
        pop_output = [Variable(torch.zeros([batch_size, p._n_out])) for p in self.populations]

        # randomly update each population
        for i in np.random.permutation(self.n_pop):

            # pass sensory input to the sensory-population connection associated with this population
            # the result is the sensory input for that population (e.g. delayed input)
            ws = self.ws[i]
            if ws is None:
                pop_input = []
            else:
                pop_input = [self.ws[i](Variable(torch.from_numpy(x['stimulus'])))]

            # get population connections entering this population and pass other populations output through the connections
            # note that population outputs can be zero or not depending on whether they have been updated already
            wp = self.Wp[i]
            for j in range(self.n_pop):
                if not wp[j] is None:
                    pop_input.append(wp[j](pop_output[j]))

            # compute population output for this population
            # population receives the concatenated inputs of sensation and incoming populations
            pop_output[i] = self.populations[i](torch.cat(pop_input, 1))

        # now we have all the outputs, we can pass it to the readout mechanism
        return self.readout(torch.cat(pop_output, 1))

    def reset(self):
        """ Reset states of model components

        :return:
        """

        raise NotImplementedError

#####
## DRM; wrapper object that trains and analyses the model at hand

class DRM(object):

    def __init__(self, populations, readout, ws, Wp, wr):
        """

        :param populations: list of populations (neural networks)
        :param readout: list of readout mechanisms
        :param ws: n_pop object array specifying a connection between the stimulus and each population.
        :param Wp: npop x npop object array specifying a connection between all populations
        :param wr: n_resp x npop object array specifying a readout mechanism for each internal population.
        """

        self.model = DRMNet(populations, readout, ws, Wp, wr)

    def estimate(self, data_iter, n_epochs=1):
        """ Here the estimation via truncated backprop takes place

        :param data_iter: iterator which generates sensations/responses at some specified resolution
        :param n_epochs: number of training epochs
        :return:
        """

        max_iter = int(n_epochs * data_iter.n_batches)

        # iterate over indices
        for _iter in tqdm.tqdm(xrange(0, max_iter)):

            # reset agents at start of each epoch
            map(lambda x: x.reset(), self.agents)

            if val_iter:
                map(lambda x: x.reset(), val_agents)

            try:
                data = d_it.next()
            except StopIteration:
                d_it = iter(data_iter)

            losses += map(lambda x: x.run(data, train=train, idx=_iter, final=data_iter.is_final()),
                          self.agents)





        loss = self.model(map(lambda x: Variable(self.xp.asarray(x)), data), train=True)

        if self.last:  # used in case we propagate back at end of trials only
            self.loss = loss
        else:
            self.loss += loss

        # normalize by number of datapoints in minibatch
        _loss = float(loss.data / data[0].shape[0])

        # backpropagate if we reach the cutoff for truncated backprop or if we processed the last batch
        if train and ((self.cutoff and (idx % self.cutoff) == 0) or final):
            self.optimizer.zero_grads()
            self.loss.backward()
            self.loss.unchain_backward()
            self.optimizer.update()

            self.loss = Variable(self.xp.zeros((), 'float32'))

        return _loss

        pass

