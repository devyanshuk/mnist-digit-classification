#global
import numpy as np

#local
from .Activations.relu import ReLU
from .Activations.sigmoid import Sigmoid
from .Activations.softmax import Softmax
from Logger.logger import log

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

        :params seed:
            Random seed to be used to extract train data for training the neural network.

        :params learning rate:
            Learning rate(alpha) hyperparameter to be used to update the weights and biases
            in the network.

        :params epochs:
            Number of stochastic gradient descent epochs.

        :params batch_size:
            Size of minibatch to be used to compute derivatives to update the weights and biases.
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
        Train the neural network using train data, X \in |R^(N * K), and targets y \in |R^(N),
        where K is the number of features in the input data.
        """
        if (any(param <= 0 for param in self.hidden_layer_sizes)):
            raise ValueError("Hidden layer must be an integer greater than 0")
        self.initWeightsAndBiases(X)
        self.minibatchSGD(X, y)

    def validate_trained(self):
        if self.weights == [] or self.biases == []:
            raise ValueError("Train the model using fit() method before having the network predict it.")

    def predict(self, X):
        """
        Use the learned weights and biases to predict outputs for the test dataset X
        """
        self.validate_trained()
        return np.argmax(np.array([np.array(self.forwardpropagation(x)[-1]) for x in X]), axis=1)

    def predict_single(self, x):
        """
        use the learned weights and biases to predict an output for a single input x
        """
        self.validate_trained()
        return np.argmax(self.forwardpropagation(x)[-1])

    def get_output_layer_single(self, x):
        """
        return the results of the output layer (probability distribution of the classes),
        given an input to neurons in the first layer.
        """
        self.validate_trained()
        return self.forwardpropagation(x)[-1]

    def initWeightsAndBiases(self, X):
        """
        Initialize the weights and biases based on the number of neurons in the network.
        Number of neurons in the network = 2 + number of hidden layers (input neurons n_in \in |R^(K) + output neurons n_out \in |R^(d)),
        where K is the number of features in the input data, and d is the number of target classes.
        """
        all_layer_sizes = [X.shape[1]] + self.hidden_layer_sizes + [self.classes]
        log.info(f"all layer sizes = {all_layer_sizes}")
        log.info(f"Training the model. Number of features = {X.shape[1]}")
        self.weights = np.array([
            self.generator.uniform(size=(all_layer_sizes[i], all_layer_sizes[i+1]), low=-0.1, high=0.1)
            for i in range(len(all_layer_sizes) - 1)
        ], dtype=object)
        self.biases = np.array([
            np.zeros((all_layer_sizes[i]))
            for i in range(1, len(all_layer_sizes))
        ], dtype=object)
        log.info(f"Weights shape = {self.weights.shape} and biases shape = {self.biases.shape}")
        for w in self.weights:
            log.info(f"weight shape = {w.shape}")
        for b in self.biases:
            log.info(f"bias shape = {b.shape}")
    
    def forwardpropagation(self, inputs):
        """
        Forward Propagation; given the input to the first layer,
        return the hidden layers with activations applied, and also the
        output layer with Softmax activation applied.
        """
        hidden_layers = []
        output_layer = None
        count = 0
        prev_input = inputs

        for i in np.stack((self.weights, self.biases), axis=1):
            count += 1
            hidden_layer = prev_input @ i[0] + i[1]
            prev_input = hidden_layer
            if count == len(self.weights):
                output_layer = hidden_layers[-1] @ self.weights[-1] + self.biases[-1]
                output_layer = Softmax.fn(output_layer)
                break
            hidden_layer = self.hidden_layer_activation.fn(hidden_layer)
            hidden_layers.append(hidden_layer) 

        return hidden_layers, output_layer


    def minibatchSGD(self, train_data, train_target):
        """
        Train the network using minibatch stochastic gradient descent.
        """
        for epoch in range(self.epochs):
            permutation = self.generator.permutation(train_data.shape[0])
            for i in range(0, len(permutation), self.batch_size):
                batch = permutation[i : i + self.batch_size]
                train_batch = train_data[batch]

                hidden_layers, output_layer = self.forwardpropagation(train_batch)

                cost, hidden_layer_gradients = self.backpropagation(
                    output_layer=output_layer,
                    target_batch=train_target[batch],
                    hidden_layers=hidden_layers
                )

                self.updateWeightsAndBiases(
                    hidden_layers=hidden_layers,
                    hidden_layer_gradients=hidden_layer_gradients,
                    cost=cost,
                    train_batch=train_data[batch]
                )

            train_accuracy = self.getAccuracy(train_data, train_target)
            log.info(f"epoch number {epoch} finished. Training accuracy = {(100 * train_accuracy):.2f}%")


    def backpropagation(self, output_layer, target_batch, hidden_layers):
        """
        After processing a mini-batch, return the total cost and the gradients of all the
        layers, which will then be used to update the weights and biases after processing a
        minibatch training examples.
        """
        hidden_layer_gradients = []
        cost = output_layer - self.getVectorizedResultForABatch(target_batch)
        grad = cost
        for j in range(len(hidden_layers) - 1, -1, -1):
            grad = grad @ self.weights[j+1].T * self.hidden_layer_activation.derivative(hidden_layers[j])
            hidden_layer_gradients.append(grad)
        return cost, hidden_layer_gradients
            

    def updateWeightsAndBiases(
        self,
        hidden_layers,
        hidden_layer_gradients,
        cost,
        train_batch):
        """
        update weights and biases after processing a mini-batch (both the forward and the backward propagations).
        """
        b_grad = hidden_layer_gradients + [cost]
        w_grad = [train_batch] + hidden_layers
        for k in range(len(self.weights) - 1, -1, -1):
            self.biases[k] -= self.learning_rate * np.mean(b_grad[k], axis=0)
            self.weights[k] -= self.learning_rate * (w_grad[k].T @ b_grad[k] / self.batch_size)

    def getVectorizedResultForABatch(self, minibatch):
        """
        Given a target class, return a (batch_size * self.classes) dimentional unit vectors
        with 1 in the jth index of each unit vector of the mini-batch (One-Hot Encoding of j)
        """
        return np.eye(self.classes)[minibatch]

    def getAccuracy(self, X, y):
        """
        Return the accuracy of the learned weights and biases, given an input X \in |R^(N * K) and target
        y \in |R^(N), where K is the number of features of the input data. 
        """
        prediction = self.predict(X)
        accuracy = sum(1 if a == b else 0 for a,b in np.stack((prediction, y), axis=1))
        return accuracy / len(prediction)
    
        

