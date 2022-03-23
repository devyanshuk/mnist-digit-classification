import pickle
import sklearn_json
import lzma
import json
import jsonpickle

from dataset import Dataset

if __name__ == "__main__":
    # with lzma.open("train.model", "rb") as model_file:
    #     model = pickle.load(model_file)
    #     out_file = open("train.json", "w")
    #     out_file.write(jsonpickle.encode(model))
    #     in_file = open("train.json", "r")
    #     new_model = jsonpickle.decode(in_file.read())
    #     print(new_model)
    dataset = Dataset()
