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
        return np.maximum(x, 0)

    @staticmethod
    def derivative(x : np.array):
        """
        derivative(ReLU(x)) = 1 if x >= 0 else 0.
        """
        return x > 0
