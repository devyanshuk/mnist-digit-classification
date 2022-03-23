from abc import abstractmethod


class Activation:

    @abstractmethod
    def fn(x):
        raise NotImplementedError(f"{type(Activation).__name__} class cannot be used as an activation function")

    @abstractmethod
    def derivative(x):
        raise NotImplementedError(f"{type(Activation).__name__} class does not have a derivative function implemented because it's not a hidden layer activation function.")