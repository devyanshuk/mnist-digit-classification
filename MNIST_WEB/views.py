#global
from django.shortcuts import render
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
import numpy as np
from io import BytesIO
from PIL import Image, ImageOps
import base64
import json

#local
from helpers import load

nn = load()

def home(request):
    return render(request, 'mnist_classifier.html')

@csrf_exempt
def handle_image(request):
    datauri = request.POST.get('data_url')
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

    outim = Image.fromarray(data.reshape(28, 28)).convert('L')
    # im.save("image.png")

    data = data / np.float64(255)
    data[data < 0.5] = 0.
    softmax_prediction = nn.get_output_layer_single(data)
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
