#global
import numpy as np

#local
from .Activation import Activation

class ReLU(Activation):

    @staticmethod
    def fn(x : np.array):
        """
        ReLU(x) = max(0, x).
        """
        return np.vectorize(lambda i : max(0, i))(x)
