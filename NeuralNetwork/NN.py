#global
import numpy as np

#local
from Activations.relu import ReLU
from Activations.sigmoid import Sigmoid
from Activations.softmax import Softmax

class NeuralNetwork:

    def __init__(self,
        classes=10,
        hidden_layer_sizes=[100],
        hidden_layer_activation=ReLU,
        seed=42,
        learning_rate=0.01,
        epochs=10,
        batch_size=10):
        """
        Initialize a neural network with len(hidden_layer_sizes) hidden layers of sizes hidden_layer_sizes
        and an activation (default = ReLU), and an output layer with softmax activation.
        
        The value of the hidden layer k is computed as activation(layer(k-1) @ weights[k-1] + biases[k-1]).
        The value of the output layer is computed as softmax(hidden_layer[n] @ weights[n] + biases[n]), where
        n = len(hidden_layer_sizes) - 1 
        
        :params hidden_layer_sizes: 
            List containing sizes of neurons in each layer, where number of layers is the size of the list.

        :params activation:
            Activation function for the hidden layers (default = 'relu')
            1) relu : ReLU(x) = max(0, x)
            2) softmax : softmax(x) = (e^x) / sum (j in x (e^j))
            3) sigmoid : sigmoid(x) = 1 / (1 + e^(-x))
        """
        self.classes = classes
        self.hidden_layer_sizes = hidden_layer_sizes
        self.hidden_layers_len = len(hidden_layer_sizes)
        self.hidden_layer_activation = hidden_layer_activation
        self.seed = seed
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size

        self.generator = np.random.RandomState(seed)
        self.weights = []
        self.biases = []

    def fit(self, X, y):
        """
        Train the neural network using minibatch SGD algorithm. 
        """
        if (any(param <= 0 for param in self.hidden_layer_sizes)):
            raise ValueError("Hidden layer must be an integer greater than 0")
        all_layer_sizes = [X.shape[1]] + self.hidden_layer_sizes + [self.classes]
        self.weights = np.array([
            self.generator.uniform(size=[all_layer_sizes[i], all_layer_sizes[i+1]], low=-0.1, high=0.1)
            for i in range(len(all_layer_sizes) - 2)
        ])
        self.biases = np.array([
            np.zeros(all_layer_sizes[i])
            for i in range(1, len(all_layer_sizes) - 1)
        ])
        self.minibatchSGD(X, y)

    
    def forward(self, input, applyActivations=False):
        """
        Forward Propagation; given the input to the first layer,
        return all layer results (activations applied based on the argument given
        to the applyActivations parameter).
        """
        layer_results = []
        prev_input = input
        for i in np.stack((self.weights, self.biases), axis=1):
            weight, bias = i[0], i[1]
            prev_input = prev_input.T @ weight + bias
            layer_results.append(prev_input)
        return layer_results if not applyActivations else \
               [self.hidden_layer_activation(i) for i in layer_results[:-1]] + [Softmax.fn(layer_results[-1])]


    def minibatchSGD(self, train_data, train_target):
        """
        """
        for epoch in range(self.epochs):
            permutation = self.generator.permutation(train_data.shape[0])
            for i in range(0, train_data.shape[0], self.batch_size):
                weight_loss_derivative = np.array(np.zeros(weight.shape) for weight in self.weights)
                biases_loss_derivative = np.array([np.zeros(bias.shape) for bias in self.biases])
                for j in range(i, i + self.batch_size):
                    train_j = train_data[permutation[j]]
                    train_target_j = train_target[permutation[j]]
                    wld, bld = self.backdrop(
                        train_j,
                        train_target_j,
                        weight_loss_derivative,
                        biases_loss_derivative
                    )
                    weight_loss_derivative += wld
                    biases_loss_derivative += bld

                self.updateWeightsAndBiases(weight_loss_derivative, biases_loss_derivative)


    def backdrop(self, train, target, weight_loss_derivative, biases_loss_derivative):
        """
        """
        layer_res = self.forward(train)
        layer_res_activation = np.array([self.hidden_layer_activation.fn(i) for i in layer_res[:-1]] + Softmax.fn(layer_res[-1]))

        delta = layer_res_activation[-1] - target
        biases_loss_derivative[-1] = delta
        weight_loss_derivative[-1] = np.dot(delta, layer_res_activation[-2].T)
        rest_layer_res_activation = np.array(layer_res[:-1])
        rest_layer_res_activation_derivative = self.hidden_layer_activation.derivative(rest_layer_res_activation)

        for i in range(2, self.hidden_layer_sizes + 2):
            delta = np.dot(self.weights[-i+1].T, delta) * rest_layer_res_activation_derivative[-i+1]
            biases_loss_derivative[-i] = delta
            weight_loss_derivative[-i] = np.dot(delta, rest_layer_res_activation[-i+1].T)

        return weight_loss_derivative, biases_loss_derivative
            


    def updateWeightsAndBiases(self, weight_loss_derivative, biases_loss_derivative):
        """
        update weights and biases after every processing self.batch_size minibatch examples.
        """
        self.weights -= self.learning_rate * (weight_loss_derivative / self.batch_size)
        self.biases -= self.learning_rate * (biases_loss_derivative / self.batch_size)

    def getVectorizedResult(self, j):
        """
        Given a target class, return a self.classes dimentional unit vector
        with 1 in the jth index (One-Hot Encoding of j)
        """
        res = np.zeros((self.classes, 1))
        res[j] = 1
        return res