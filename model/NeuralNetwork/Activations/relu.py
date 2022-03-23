#global
import numpy as np

#local
from .activation import Activation

class ReLU(Activation):

    @staticmethod
    def fn(x : np.array):
        """
        ReLU(x) = max(0, x).
        """
        return np.vectorize(lambda i : max(0, i))(x)

    @staticmethod
    def derivative(x : np.array):
        """
        derivative(ReLU(x)) = 1 if x >= 0 else 0.
        """
        return np.vectorize(lambda i : 1 if i >= 0 else 0)(x)
