#global
import os
import ssl
import gzip
import numpy as np
from urllib.request import urlretrieve

#local
from .labels import *
from Logger.logger import log

class Dataset:
    """MNIST Dataset.
    The train set contains 60000 images of handwritten digits. The data
    contain 28*28=784 values in range 0-255, the targets are numbers 0-9.
    """
    def __init__(self,
                 name,
                 isTarget,
                 shape=[-1, 28*28],
                 baseDir="data",
                 url="http://yann.lecun.com/exdb/mnist"):

        if not os.path.isdir(baseDir):
            log.info(f"{baseDir} directory does not exist. Making a new one...")
            os.mkdir(baseDir)

        fileName = f"{baseDir}/{name}"

        if not os.path.exists(fileName):
            log.info(f"Downloading dataset {name}...")
            if (not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None)):
                ssl._create_default_https_context = ssl._create_unverified_context
            urlretrieve(f"{url}/{name}", filename=fileName)
        else:
            log.info(f"{fileName} exists")

        with gzip.open(fileName, 'rb') as f:
            self.data = np.frombuffer(f.read(), np.uint8, offset=16 if not isTarget else 8)
        if not isTarget:
            self.data = self.data.reshape(shape)
            self.data = self.data / np.float32(256)