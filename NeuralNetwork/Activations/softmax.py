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
        res = np.exp(x - np.max(x, axis=-1, keepdims=True))
        res /= np.sum(res, axis=-1, keepdims=True)
        return res