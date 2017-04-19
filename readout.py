import chainer.links as L
import chainer.functions as F
from base import DRMNode

#####
## DRMReadout base class

class DRMReadout(DRMNode):
    # Custom readout mechanism just copies the population response
    # It assumes that population activity can be read out directly

    def __call__(self, x):
        return x[0]

#####
## Linear readout

class DRMReadout2(DRMNode):

    def __init__(self, n_out=1):

        super(DRMReadout, self).__init__()

        self.n_out = n_out

        self.l1 = L.Linear(None, n_out)

    def __call__(self, x):
        """Forward propagation

        :param x: readout input
        :type x: list of afferent population outputs
        :return: predicted measurements
        """

        # this readout mechanism concatenates all population outputs for further processing
        x = F.concat(x, axis=1)

        return self.l1(x)
