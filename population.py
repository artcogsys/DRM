import torch.nn as nn
import torch.nn.functional as F

#####
## Default population object - a chain object whose inputs and outputs don't need to match

class DRMPopulation(nn.Module):
    """
    An identity mapping
    """

    def __init__(self, out_shape=1):
        """

        :param out_shape: shape of the output; required in base.py
        """

        self.out_shape = out_shape

        super(DRMPopulation, self).__init__()

    def forward(self, x, train=False):
        return x