import torch.nn as nn
import torch.nn.functional as F
from base import DRMNode
import torch


#####
## DRMPopulation base class

class DRMPopulation(DRMNode):
    # Default population is a linear layer with ReLU output

    def __init__(self, n_in=1, n_out=1, delay=1):

        super(DRMPopulation, self).__init__(n_in, n_out)

        self.l1 = nn.Linear(n_in, n_out)

        # zero value defines average population activity
        self.l1.bias.data.uniform_(0, 0)

    def forward(self, x):

        # the list of inputs (e.g. stimulus and other populations) are concatenated for further processing
        x = torch.cat(x, 1)

        # relu unstable when we have few units and the input is always negative (zero gradient)
        return F.leaky_relu(self.l1(x))