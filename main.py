#local
from Dataset.labels import *
from Dataset.dataset import Dataset

if __name__ == "__main__":
    train = Dataset(name=TRAIN_DATA, isTarget=False)
    print(train.data.shape)
