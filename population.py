import chainer.links as L
import chainer.functions as F
from base import DRMNode

#####
## DRMPopulation base class

class DRMPopulation(DRMNode):
    # Default population is a linear layer with ReLU output

    def __init__(self, n_out=1):

        super(DRMPopulation, self).__init__()

        self.n_out = n_out

        self.add_link('l1', L.Linear(None, n_out))

    def __call__(self, x):
        """Forward propagation

        :param x: population input
        :type x: list of afferent connection outputs
        :return: population output
        """

        # the list of inputs (e.g. stimulus and other populations) are concatenated for further processing
        x = F.concat(x, axis=1)

        # relu unstable when we have few units and the input is always negative (zero gradient)
        return F.leaky_relu(self.l1(x))