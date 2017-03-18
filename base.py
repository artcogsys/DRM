import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from collections import OrderedDict
import numpy as np
import tqdm
import copy

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

    def __init__(self, populations, ws, Wp, readout):
        """ Each population receives either sensory input or input from other populations.
        This reception is mediated by connections which are neural networks themselves.
        There are three kinds of connections: sensory-population, population-population, population-response
        These all derive from the same object but we can have specific default implementations. Absent connections are
        represented as None.

        :param populations: npop list specifying each population
        :param ws: list specifying a connection between the stimulus and each population.
        :param Wp: npop x npop object array specifying a connection between all populations
        :param readout: list of readout mechanisms
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

        super(DRMNet, self).__init__(_dict)

        # add populations
        self.populations = populations

        # add readout mechanism
        self.readout = readout

        ### add connections

        self.ws = ws

        self.Wp = Wp

        ## define loss criterion
        self.loss = nn.MSELoss()

    def forward(self, x):
        """

        :param x: sensory input at this point in time (zeros for no input); numpy array
        :param train: whether we are in train or test mode (NOT USED FOR NOW)
        :return:
        """

        batch_size = x.size()[0]

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
                # push stimulus into connection that links stimulus to i-th population
                pop_input = [self.ws[i](x)]

            # get population connections entering this population and pass other populations output through the connections
            # note that population outputs can be zero or not depending on whether they have been updated already
            wp = self.Wp[i]
            for j in range(self.n_pop):
                if not wp[j] is None:
                    # push output of j-th population into connection that links it to i-th population
                    pop_input.append(wp[j](pop_output[j]))

            # pop_input now contains the output of all connections that provide the input to the i-th population

            # compute population output for the i-th population
            pop_output[i] = self.populations[i](pop_input)

        # now we have all the outputs, we can pass it to the readout mechanism
        return self.readout(pop_output)

    def reset(self):
        """ Reset states of model components

        :return:
        """

        for i in range(self.n_pop):

            # reset stimulus connections
            self.ws[i].reset()

            # reset population connections
            wp = self.Wp[i]
            for j in range(self.n_pop):
                if not wp[j] is None:
                    wp[j].reset()

            self.populations[i].reset()

        self.readout.reset()

#####
## DRM; wrapper object that trains and analyses the model at hand

class DRM(object):

    def __init__(self, populations, ws, Wp, readout):
        """

        :param populations: list of populations (neural networks)
        :param ws: n_pop object array specifying a connection between the stimulus and each population.
        :param Wp: npop x npop object array specifying a connection between all populations
        :param readout: list of readout mechanisms
        """

        self.model = DRMNet(populations, ws, Wp, readout)

        self.model.train(True)

        # hard coded for now
        self.optimizer = optim.Adam(self.model.parameters())

    def estimate(self, data_iter, val_iter=None, n_epochs=1, cutoff=None):
        """ Here the estimation via truncated backprop takes place

        :param data_iter: iterator which generates sensations/responses at some specified resolution
        :param val_iter: optional iterator which generates sensations/responses at some specified resolution used for validation
        :param n_epochs: number of training epochs
        :param cutoff: cutoff for truncated backpropagation
        :return:
        """

        # initialization for validation
        min_loss = optimal_model = None

        # track training and validation loss
        train_loss = np.zeros(n_epochs)
        validation_loss = np.zeros(n_epochs)

        idx = 0
        for epoch in tqdm.tqdm(xrange(0, n_epochs)):

            # keep track of loss
            _loss = Variable(torch.zeros(1))

            # reset agents at start of each epoch
            self.model.reset()

            for data in data_iter:

                # compute training loss
                _l = self.model.loss(self.model.forward(data['stimulus']), data['response'])
                _loss += _l

                train_loss[epoch] += _l.data[0] / data['stimulus'].size()[0]

                # backpropagate if we reach the cutoff for truncated backprop or if we processed the last batch
                if (cutoff and idx == cutoff-1) or data_iter.is_final():

                    # update model - NOT SURE HOW TO HANDLE PERSISTENT STATES.........
                    self.optimizer.zero_grad()
                    _loss.backward()
                    self.optimizer.step()

                    _loss = Variable(torch.zeros(1))

                    idx = 0

                idx += 1

            # run validation
            if not val_iter is None:

                # copy parameters of trained model
                val_model = copy.copy(self.model) # DEEP COPY NOT ALLOWED.......... HOPING THIS IS SUFFICIENT
                val_model.train(False)

                # reset agents at start of each epoch
                val_model.reset()

                for data in val_iter:

                    # compute validation loss
                    _l = self.model.loss(self.model.forward(data['stimulus']), data['response'])

                    validation_loss[epoch] += _l.data[0] / data['stimulus'].size()[0]

                # store best model in case loss was minimized
                if not val_iter is None:

                    if min_loss is None:
                        optimal_model = val_model
                        min_loss = validation_loss[epoch]
                    else:
                        if validation_loss[epoch] < min_loss:
                            optimal_model = copy.deepcopy(val_model)
                            min_loss = validation_loss[epoch]

        if not val_iter is None:
            self.model = optimal_model

        return train_loss, validation_loss
