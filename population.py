import torch.nn as nn
import torch.nn.functional as F
from base import DRMNode

#####
## DRMPopulation base class

class DRMPopulation(DRMNode):
    # Default population is a linear layer with ReLU output

    def __init__(self, n_in=1, n_out=1, delay=1):

        super(DRMPopulation, self).__init__(n_in, n_out)

        self.l1 = nn.Linear(n_in, n_out)

    def forward(self, x):
        return F.relu(self.l1(x))
