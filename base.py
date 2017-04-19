import chainer
from chainer import ChainList, Chain
from chainer import Variable
import chainer.functions as F
import numpy as np
import tqdm
import copy

class DRMLoss(Chain):
    """MSE loss which ignores missing data
    
    Make this a feature request in chainer

    """

    def __init__(self):

        super(DRMLoss, self).__init__()

        self.loss = F.mean_squared_error

    def __call__(self, prediction, target):
        """Computes loss on a prediction and a target

        Computes MSE loss but ignores those terms where the target is equal to nan, indicating missing data.

        :param prediction: Prediction of output
        :param target: Target output
        :type prediction: Variable
        :type target: Variable
        :return: MSE loss
        :rtype: Variable
        """

        idx = np.where(np.any(np.isnan(target.data), axis=1) == False)[0].tolist()

        return self.loss(prediction[idx,:], target[idx,:])


class DRMNode(Chain):
    """Base class for populations, readouts and connections
    """

    def reset(self):
        """ The function that is called when resetting internal state
        """

        pass


#####
## DRMNet; the neural network which includes populations, connections and readouts

class DRMNet(ChainList):

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

        super(DRMNet, self).__init__()

        self.n_pop = len(populations)

        ### add component links

        for i in range(self.n_pop):
            self.add_link(populations[i])

        self.add_link(readout)

        for i in range(self.n_pop):
            if not ws[i] is None:
                self.add_link(ws[i])

        wp = np.ravel(Wp)
        for i in range(Wp.size):
            if not wp[i] is None:
                self.add_link(wp[i])

        ### add populations
        self.populations = populations

        ### add connections

        self.ws = ws

        self.Wp = Wp

        # add readout mechanism
        self.readout = readout

        ### store population activity
        self.pop_output = []


    def __call__(self, x):
        """Forward propagation

        :param x: sensory input at this point in time (zeros for no input); numpy array
        :return: predicted output measurements
        """

        batch_size = x.shape[0]

        # initialize population outputs
        self.pop_output = [Variable(np.zeros([batch_size, p.n_out], dtype='float32')) for p in self.populations]

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

    def __init__(self, drm_net, loss=None):
        """

        :param drm_net: a DRM network
        :param loss: loss function
        """

        self.model = drm_net

        # stores optimal model according to validation loss
        self._optimal_model = copy.deepcopy(self.model)

        # optimizer
        self.optimizer = chainer.optimizers.Adam()
        self.optimizer.setup(self.model)

        # loss function
        if loss is None:
            self.loss = DRMLoss()
        else:
            self.loss = loss

    def __call__(self, data_iter):
        """Forward propagation

        :param data_iter:
        :return: generated response and population activity
        """

        response = []
        activity = []

        self.model.reset()

        with chainer.using_config('train', False):

            for data in data_iter:

                r = self.model(data['stimulus']).data

                # keep track of population activity
                activity.append(np.array([x.data[0,0] for x in self.model.pop_output]))

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
            loss = Variable(np.zeros((), 'float32'))

            # reset at start of each epoch
            self.model.reset()

            with chainer.using_config('train', True):

                for data in data_iter:

                    # compute training loss
                    _loss = self.loss(self.model(data['stimulus']), data['response'])
                    loss += _loss

                    train_loss[epoch] += _loss.data / data['stimulus'].shape[0]

                    # backpropagate if we reach the cutoff for truncated backprop or if we processed the last batch
                    if (cutoff and idx == cutoff-1) or data_iter.is_final():

                        self.optimizer.zero_grads()
                        _loss.backward()
                        _loss.unchain_backward()
                        self.optimizer.update()

                        loss = Variable(np.zeros((), 'float32'))

                        idx = 0

                    idx += 1

            # run validation
            if not val_iter is None:

                # reset at start of each epoch
                self.model.reset()

                with chainer.using_config('train', False):

                    for data in val_iter:

                        # compute validation loss
                        _loss = self.loss(self.model(data['stimulus']), data['response'])

                        validation_loss[epoch] += _loss.data / data['stimulus'].shape[0]

                # store best model in case loss was minimized
                if not val_iter is None:
                    if min_loss is None or validation_loss[epoch] < min_loss:
                        self._optimal_model = copy.deepcopy(self.model)
                        min_loss = validation_loss[epoch]

        # set model to optimal model
        if not val_iter is None:
            self.model = copy.deepcopy(self._optimal_model)

        return train_loss, validation_loss
