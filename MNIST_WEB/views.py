from django.shortcuts import render
from django.http import HttpResponse
from NeuralNetwork.NN import NeuralNetwork
from helpers import load, save

def home(request):
    nn = load()
    return render(request, 'mnist_classifier.html')
