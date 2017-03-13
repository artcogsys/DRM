from brain.networks import *

#####
## Default population object - a chain object whose inputs and outputs don't need to match

class DRMPopulation(Chain, Network):
    """
    An identity mapping
    """

    def __init__(self, out_shape=1):
        """

        :param out_shape: shape of the output; required in base.py
        """

        self.out_shape = out_shape

        super(DRMPopulation, self).__init__()

    def __call__(self, x, train=False):
        return x


# class DRMPopulation(Chain, Network):
#
#     def __init__(self, n_hidden=1, n_output=1):
#         """
#
#         :param n_hidden: number of hidden units
#         :param n_output: number of outputs that are sent by this model
#         """
#
#         super(DRMPopulation, self).__init__(
#             l1=Elman(None, n_hidden),
#             l2=L.Linear(n_hidden, n_output)
#         )
#
#     def __call__(self, x, train=False):
#         raise NotImplementedError
#
#     def reset_state(self):
#         raise NotImplementedError
