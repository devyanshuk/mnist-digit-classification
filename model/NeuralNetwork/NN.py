#global
import numpy as np

#local
from Activations.relu import ReLU
from Activations.sigmoid import Sigmoid
from Activations.softmax import Softmax

class NeuralNetwork:

    def __init__(self, hidden_layer_sizes=[100], hidden_layer_activation=ReLU, seed=42, learning_rate=0.01):
        """
        Initialize a neural network.

        We assume a neural network with len(hidden_layer_sizes) hidden layers of sizes hidden_layer_sizes
        and an activation (default = ReLU), and an output layer with softmax activation.
        
        The value of the hidden layer k is computed as activation(layer(k-1) @ weights[k-1] + biases[k-1]).
        The value of the output layer is computed as softmax(hidden_layer[n] @ weights[n] + biases[n]), where
        n = len(hidden_layer_sizes) - 1 
        
        :params hidden_layer_sizes: 
            List containing sizes of
            neurons in each layer, where
            number of layers is the size
            of the list.

        :params activation:
            Activation function for the hidden layers (default = 'relu')
            1) relu : ReLU(x) = max(0, x)
            2) softmax : softmax(x) = (e^x) / sum (j in x (e^j))
            3) sigmoid : sigmoid(x) = 1 / (1 + e^(-x))
        """
        
        self.hidden_layer_sizes = hidden_layer_sizes
        self.hidden_layers = len(hidden_layer_sizes)
        self.hidden_layer_activation = hidden_layer_activation
        self.seed = seed
        self.learning_rate = learning_rate
        
        self.generator = np.random.RandomState(seed)
        self.weights = []
        self.biases = []

    def fit(self, X, y):
        """
        Train the neural network using minibatch SGD algorithm. 
        """
        if (np.any(param <= 0 for param in self.hidden_layer_sizes)):
            raise ValueError("Hidden layer must be an integer greater than 0")
        

    
    def forward(self, input):
        """
        returns the output of the current layer, given the input of the current one,
        after applying the activation function.
        """