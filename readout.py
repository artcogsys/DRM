import torch.nn as nn
from base import DRMNode

#####
## DRMReadout base class

class DRMReadout(DRMNode):

    def __init__(self, n_in=1, n_out=1, delay=1):

        super(DRMReadout, self).__init__(n_in, n_out)

        self.l1 = nn.Linear(n_in, n_out)

    def forward(self, x):
        return self.l1(x)
