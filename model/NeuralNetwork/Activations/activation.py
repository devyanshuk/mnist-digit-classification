from abc import abstractmethod


class Activation:

    @abstractmethod
    def fn(x):
        raise NotImplementedError(f"{type(Activation).__name__} class cannot be used as an activation function")