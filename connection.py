from torch.autograd import Variable
import torch
import torch.nn as nn
from base import DRMNode

#####
## DRMConnection base class

class DRMConnection(DRMNode):

    def __init__(self, n_in=1, n_out=1, delay=1):
        """ Implements a basic delay mechanism

        :param delay: Conduction delay in terms of number of sampling steps
        """

        super(DRMConnection, self).__init__(n_in, n_out)

        self._delay = delay
        self._history = None

    def forward(self, x):

        if self.history is None:
            self.history = [Variable(torch.zeros(x.shape)) for i in range(self.delay)]

        self.history.append(x)
        y = self.history[0]
        self.history = self.history[1:]

        return y

    def reset_state(self):
        self.history = None