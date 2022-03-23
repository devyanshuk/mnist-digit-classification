#global
import numpy as np

#local
from .activation import Activation

class Sigmoid(Activation):

    @staticmethod
    def fn(x : np.array):
        """
        sigmoid(x) = 1 / (1 + e^(-x))
        """
        return np.vectorize(lambda i : (1.0 / (1.0 + np.exp(-i))))(x)

    @staticmethod
    def derivative(x):
        """
        derivative(sigmoid(x)) = sigmoid(x) * (1 - sigmoid(x))
        """
        return Sigmoid.fn(x) * (1 - Sigmoid.fn(x))