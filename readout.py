import torch.nn as nn
import torch.nn.functional as F

#####
## Default readout mechanism
# A chain object which takes all population outputs and translates this into measurements that can act as input to a regressor

class DRMReadout(nn.Module):

    def __init__(self, in_shape, out_shape):
        """

        :param n_output: number of outputs that are sent by this model (these are the measurements to be predicted)
        """

        super(DRMReadout, self).__init__(
            l1=nn.Linear(in_shape, out_shape)
        )

    def forward(self, x, train=False):
        return self.l1(x)
