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
        """Forward propagation

        :param x: input to connection
        :return: connection output
        """

        if self._history is None:
            self._history = [Variable(torch.zeros(x.size())) for i in range(self._delay)]

        self._history.append(x)
        y = self._history.pop(0)

        return y

    def reset(self):
        """Reset state
        """

        self._history = None

    def detach_(self):
        """Detach gradients for truncation
        """

        for x in self._history:
            x.detach_()

