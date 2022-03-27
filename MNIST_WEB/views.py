from django.shortcuts import render
from django.http import HttpResponse
from NeuralNetwork.NN import NeuralNetwork
from helpers import load, save
from django.views.decorators.csrf import csrf_exempt
import re
import numpy as np
from io import BytesIO
from PIL import Image
import base64

nn = load()

def home(request):
    nn = load()
    return render(request, 'mnist_classifier.html')


@csrf_exempt
def handle_image(request):
    datauri = request.POST['image']
    image_data = re.sub("^data:image/png;base64,", "", datauri)
    image_data = base64.b64decode(image_data)
    image_data = BytesIO(image_data)
    img = Image.open(image_data).resize(size=(28, 28)).convert('LA')
    alpha = np.array([a[-1] for a in list(img.getdata())]) / np.float32(256)
    prediction = nn.predict_single(alpha)
    print("PREDICTION ", prediction)
    return HttpResponse("OK");
