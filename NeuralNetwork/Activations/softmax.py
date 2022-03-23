#global
import numpy as np

#local
from .activation import Activation

class Softmax(Activation):

    @staticmethod
    def fn(x : np.array):
        """
        softmax(x) = e^x / sum (j over x (e^j))
        """
        return np.exp(x) / np.sum(np.exp(x))