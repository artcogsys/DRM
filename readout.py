import torch.nn as nn
from base import DRMNode
import torch

#####
## DRMReadout base class

class DRMReadout(DRMNode):

    def __init__(self, n_in=1, n_out=1):

        super(DRMReadout, self).__init__(n_in, n_out)

        self.l1 = nn.Linear(n_in, n_out)

    def forward(self, x):

        # this readout mechanism concatenates all population outputs for further processing
        x = torch.cat(x, 1)

        return self.l1(x)
