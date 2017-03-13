from brain.networks import *

#####
## Default readout mechanism
# A chain object which takes all population outputs and translates this into measurements that can act as input to a regressor

class DRMReadout(Chain, Network):

    def __init__(self, out_shape):
        """

        :param n_output: number of outputs that are sent by this model (these are the measurements to be predicted)
        """

        super(DRMReadout, self).__init__(
            l1=L.Linear(None, out_shape)
        )

    def __call__(self, x, train=False):
        return self.l1(x)



# class DRMReadout(Chain, Network):
#
#     def __init__(self, n_hidden=1, n_output=1):
#         """
#
#         :param n_hidden: number of hidden units
#         :param n_output: number of outputs that are sent by this model
#         """
#
#         super(DRMReadout, self).__init__(
#             l1=Elman(None, n_hidden),
#             l2=L.Linear(n_hidden, n_output)
#         )
#
#     def __call__(self, x, train=False):
#         raise NotImplementedError
#
#     def reset_state(self):
#         raise NotImplementedError
