#global
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#local
from helpers import save, load
from Logger.logger import log
from Dataset.labels import *
from Dataset.dataset import Dataset
from NeuralNetwork.NN import NeuralNetwork


if __name__ == "__main__":
    seed = 42
    train = Dataset(name=TRAIN_DATA, isTarget=False)
    t_target = Dataset(name=TRAIN_TARGET, isTarget=True)
    test = Dataset(name=TEST_DATA, isTarget=False)
    test_target = Dataset(name=TEST_TARGET, isTarget=True)

    train_data, validation_data, train_target, validation_target = \
        train_test_split(train.data, t_target.data, test_size=10000, random_state=seed)
    
    log.info(f"train data shape = {train_data.shape}")
    log.info(f"validation data shape = {validation_data.shape}")
    log.info(f"train target shape = {train_target.shape}")
    log.info(f"validation target shape = {validation_target.shape}")
    #nn = NeuralNetwork(classes=10, seed = 42, epochs=20)
    #nn.fit(train_data, train_target)

    #save trained model
    #save(nn)

    #load trained model
    nn = load()

    log.info(f"validation dataset score = {(100 * accuracy_score(validation_target, nn.predict(validation_data))):.2f}%")
    log.info(f"test dataset score = {(100 * accuracy_score(test_target.data, nn.predict(test.data))):.2f}%")

