
import numpy as np

class CrossEntropy:
    
    @staticmethod
    def apply(P : np.array, Q : np.array):
        """
        The cross-entropy of the target distribution P and the estimated
        distribution Q is the function H(P, Q) = -sum for x in X (P(x) * log(Q(x)))
        """
        return -np.sum(P * Q)