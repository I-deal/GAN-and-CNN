from __future__ import print_function
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from keras import backend as K
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
import scipy.misc as misc
import os

""" Starting with a random seed ensures the reproducibility of the tests. """
np.random.seed(1337)

""" Initialize some variables. """
nb_classes = 10
nb_epoch = 20
batch_size = 128
nb_filter = 32          # Number of convolutional filters to use
pool_size = (2, 2)      # Size of poolig area
kernel_size = (3, 3)    # Convolution kernel size

"""Load data"""
data_dir = "../generated_data"
files = os.listdir(data_dir)
files = sorted(files)

X = []
for file in files:
    im = misc.imread(data_dir+"/"+file, flatten=True).astype('float32')
    im = im.reshape([28, 28, 1])
    X.append(im)

X = np.array(X)

""" Width and height of the training images. """
input_shape = (X.shape[1], X.shape[2], 1)

X /= 255

print('Input shape: ', X.shape)
print(X.shape[0], 'Number of generated data.')

""" Create a sequential model. """
model = Sequential()

model.add(Convolution2D(nb_filter, kernel_size[0], kernel_size[1],
                        border_mode='valid', input_shape=input_shape))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filter, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.25)) # 0.5?
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

""" Let's look at the summary of the model. """
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

weights_file = Path('trained_weights.h5')

""" Load pre-computed weights """
print('Loading weights...')
model.load_weights(weights_file.name)

""" Compute probability by the trained model """
pred = model.predict(X, batch_size=1, verbose=0)

def compute_entropy(p):
    return np.sum(-p*np.log2(p))

h = [[idx, compute_entropy(p)] for idx, p in enumerate(pred)]
h = sorted(h, key=lambda x: -x[1])

with open("../entropy_list.csv", 'w') as output:
    for sample in h:
        print("%s,%s,%s" % (str(sample[0]).zfill(5), str(sample[0]//169%10), sample[1]), file=output, end="")
        for prob in pred[sample[0]]:
            print(",%s" % prob, file=output, end="")
        print("", file=output)