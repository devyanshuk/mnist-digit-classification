#global
import jsonpickle

#local
from NeuralNetwork.NN import NeuralNetwork

def save(nn : NeuralNetwork, file):
    """
    Save all learned weights and biases to be used later on without having to train the
    network again.
    """
    nn.validate_trained()
    out_file = open(file, "w")
    out_file.write(jsonpickle.encode(nn))

def load():
    """
    Load all learned weights and biases that was trained before.
    """
    in_file = open("train.json", "r")
    return jsonpickle.decode(in_file.read())