#global
import numpy as np

#local
from Logger.logger import log
from Dataset.labels import *
from Dataset.dataset import Dataset
from NeuralNetwork.NN import NeuralNetwork
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    seed = 42
    train = Dataset(name=TRAIN_DATA, isTarget=False)
    t_target = Dataset(name=TRAIN_TARGET, isTarget=True)

    train_data, validation_data, train_target, validation_target = \
        train_test_split(train.data, t_target.data, test_size=10000, random_state=seed)
    
    log.info(f"train data shape = {train_data.shape}")
    log.info(f"validation data shape = {validation_data.shape}")
    log.info(f"train target shape = {train_target.shape}")
    log.info(f"validation target shape = {validation_target.shape}")
    nn = NeuralNetwork(classes=10, seed = 42)
    nn.fit(train_data, train_target)

