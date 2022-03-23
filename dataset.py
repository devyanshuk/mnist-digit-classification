#global
import gzip
import numpy as np

class Dataset:
    
    def __init__(self):
        self.train_data, self.train_target, \
        self.validation_data, self.validation_target = self.load_train_validation()


    def load_train_validation(self):
        train_file = gzip.open("dataset/train-images-idx3-ubyte.gz")
        train_target_file = gzip.open("dataset/train-labels-idx1-ubyte.gz")

        train_data = np.load(train_file, allow_pickle=True)
        train_target = np.load(train_target_file, allow_pickle=True)

        train_file.close()
        train_target_file.close()

        print("train_data", train_data)
        print("train_target", train_target)

        return train_data, train_target
