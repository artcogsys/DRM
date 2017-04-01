import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from collections import OrderedDict
import numpy as np
import tqdm
import copy

class DRMLoss(nn.Module):
    """MSE loss which ignores missing data
    """

    def __init__(self):

        super(DRMLoss, self).__init__()

        self.loss = nn.MSELoss()

    def forward(self, prediction, target):
        """Computes loss on a prediction and a target

        Computes MSE loss but ignores those terms where the target is equal to nan, indicating missing data.

        :param prediction: Prediction of output
        :param target: Target output
        :type prediction: Variable
        :type target: Variable
        :return: MSE loss
        :rtype: Variable
        """

        idx = np.where(np.any(np.isnan(target.data.numpy()), axis=1) == False)[0]
        indices = Variable(torch.LongTensor(idx))
        prediction = torch.index_select(prediction, 0, indices)
        target = torch.index_select(target, 0, indices)

        return self.loss(prediction, target)


class DRMNode(nn.Module):
    """Base class for populations, readouts and connections
    """

    def __init__(self, n_in=1, n_out=1):
        """

        :param n_in: number/shape of inputs
        :param n_out: number/shape of outputs
        :type n_in: scalar or numpy tensor
        :type n_out: scalar or numpy tensor
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

    def detach_(self):

        pass

#####
## DRMNet; the neural network which includes populations, connections and readouts

class DRMNet(nn.Sequential):

    def __init__(self, populations, ws, Wp, readout):
        """ DRMNet initializer

        Each population receives either sensory input or input from other populations.
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

        ### add populations
        self.populations = populations

        # add readout mechanism
        self.readout = readout

        ### add connections

        self.ws = ws

        self.Wp = Wp

        ### define loss criterion

        self.loss = DRMLoss()

        ### store population activity
        self.pop_output = []

    def detach_(self):
        """Detach gradients for truncation
        """

        for p in self.populations:
            p.detach_()

        for w in self.ws:
            if w:
                w.detach_()

        for w in self.Wp.ravel():
            if w:
                w.detach_()

    def forward(self, x):
        """Forward propagation

        :param x: sensory input at this point in time (zeros for no input); numpy array
        :return: predicted output measurements
        """

        batch_size = x.size()[0]

        # initialize population outputs
        self.pop_output = [Variable(torch.zeros([batch_size, p._n_out])) for p in self.populations]

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
                    pop_input.append(wp[j](self.pop_output[j]))

            # pop_input now contains the output of all connections that provide the input to the i-th population

            # compute population output for the i-th population
            self.pop_output[i] = self.populations[i](pop_input)

        # now we have all the outputs, we can pass it to the readout mechanism
        return self.readout(self.pop_output)

    def reset(self):
        """ Reset states of model components
        """

        for i in range(self.n_pop):

            # reset stimulus connections
            if not self.ws[i] is None:
                self.ws[i].reset()

            # reset population connections
            wp = self.Wp[i]
            for j in range(self.n_pop):
                if not wp[j] is None:
                    wp[j].reset()

            self.populations[i].reset()

        self.readout.reset()

class DRM(object):
    """wrapper object that trains and analyses the model at hand
    """

    def __init__(self, drm_net):
        """

        :param drm_net: a DRM network
        """

        self.model = drm_net

        # stores optimal model according to validation loss
        self._optimal_model = copy.deepcopy(self.model)

        # optimizer
        self.optimizer = optim.Adam(self.model.parameters())

    def forward(self, data_iter):
        """Forward propagation

        :param data_iter:
        :return: generated response and population activity
        """

        response = []
        activity = []

        self.model.reset()

        # evaluation mode
        self.model.eval()

        for data in data_iter:

            r = self.model.forward(data['stimulus']).data.numpy()

            # keep track of population activity
            activity.append(np.array([x.data.numpy()[0,0] for x in self.model.pop_output]))

            response.append(r)

        return np.vstack(response), np.vstack(activity)

    def estimate(self, data_iter, val_iter=None, n_epochs=1, cutoff=None):
        """Estimation via truncated backprop

        :param data_iter: iterator which generates sensations/responses at some specified resolution
        :param val_iter: optional iterator which generates sensations/responses at some specified resolution used for validation
        :param n_epochs: number of training epochs
        :param cutoff: cutoff for truncated backpropagation
        :return: train loss and validation loss
        """

        # initialization for validation
        min_loss = None

        # track training and validation loss
        train_loss = np.zeros(n_epochs)
        validation_loss = np.zeros(n_epochs)

        idx = 0
        for epoch in tqdm.tqdm(xrange(0, n_epochs)):

            # keep track of loss
            _loss = Variable(torch.zeros(1))

            # reset at start of each epoch
            self.model.reset()

            # training mode
            self.model.train()

            for data in data_iter:

                # compute training loss
                _l = self.model.loss(self.model.forward(data['stimulus']), data['response'])
                _loss += _l

                train_loss[epoch] += _l.data[0] / data['stimulus'].size()[0]

                # backpropagate if we reach the cutoff for truncated backprop or if we processed the last batch
                if (cutoff and idx == cutoff-1) or data_iter.is_final():

                    self.optimizer.zero_grad()
                    _loss.backward()
                    self.optimizer.step()
                    self.model.detach_()

                    _loss = Variable(torch.zeros(1))

                    idx = 0

                idx += 1

            # run validation
            if not val_iter is None:

                # reset at start of each epoch
                self.model.reset()

                # evaluation mode
                self.model.eval()

                for data in val_iter:

                    # compute validation loss
                    _l = self.model.loss(self.model.forward(data['stimulus']), data['response'])

                    validation_loss[epoch] += _l.data[0] / data['stimulus'].size()[0]

                # store best model in case loss was minimized
                if not val_iter is None:
                    if min_loss is None or validation_loss[epoch] < min_loss:
                        self._optimal_model.load_state_dict(self.model.state_dict())
                        min_loss = validation_loss[epoch]

        # set model to optimal model
        if not val_iter is None:
            self.model.load_state_dict(self._optimal_model.state_dict())

        return train_loss, validation_loss
