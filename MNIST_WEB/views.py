#global
from curses.ascii import SI
from django.shortcuts import render
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
import numpy as np
from io import BytesIO
from PIL import Image, ImageOps
import base64
import json
from NeuralNetwork.Activations.activation import Activation
from NeuralNetwork.Activations.relu import ReLU
from NeuralNetwork.Activations.sigmoid import Sigmoid
from NeuralNetwork.Activations.softmax import Softmax
from NeuralNetwork.NN import NeuralNetwork

#local
from helpers import load, save

nn = load()
t_data = []
t_target = []

def home(request):
    return render(request, 'mnist_classifier.html')

@csrf_exempt
def _save(request):
    np.savez('self_train.npz', train_data=np.array(t_data), train_target=np.array(t_target))
    return HttpResponse("OK");

@csrf_exempt
def handle_image(request):
    datauri = request.POST.get('data_url')
    print(request.POST)
    target = np.array([int(request.POST.get('field'))]).astype(np.uint8)

    offset = datauri.index(',')+1
    img_bytes = base64.b64decode(datauri[offset:])
    img = Image.open(BytesIO(img_bytes))
    img = img.resize((28,28), Image.ANTIALIAS)
    r,g,b,a = img.split()
    rgb_image = Image.merge('RGB', (r,g,b))
    inverted_image = ImageOps.invert(rgb_image)
    r2,g2,b2 = inverted_image.split()
    img = Image.merge('RGBA', (r2,g2,b2,a))
    data = np.array([x[-1] for x in img.getdata()]).astype(np.uint8)
    t_data.append(data)
    t_target.append(target[0])

    outim = Image.fromarray(data.reshape(28, 28)).convert('L')
    # im.save("image.png")

    data = data / np.float64(255)
    data[data < 0.5] = 0.

    datatemp = data.reshape([-1, 28*28])
    print("Data shape", data.shape)
    print("target shape", target.shape)

    nn.fit_single(datatemp, target)
    # nn2.fit_single(datatemp, target)
    # nn3.fit_single(datatemp, target)
    save(nn, "train.json")
    # save(nn2, "train_self_relu.json")
    # save(nn3, "train_self_sigmoid.json")

    softmax_prediction = nn.get_output_layer_single(data)
    print(softmax_prediction)
    prediction = np.argmax(softmax_prediction).item()

    file_obj = BytesIO()
    outim.save(file_obj, format='png')
    encoded = base64.b64encode(file_obj.getvalue()).decode('utf-8')

    return HttpResponse(json.dumps(
        {
            "softmax" : [format(x, '.5f') for x in list(softmax_prediction)],
            "prediction" : prediction,
            "processed_image" : encoded
        }
    ));
