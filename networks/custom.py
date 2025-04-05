from __future__ import absolute_import
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Conv2D, ZeroPadding2D

def layers(input_shape):
    return [
        ZeroPadding2D((3, 3), input_shape=input_shape),
        Conv2D(64, (7, 7), padding='valid'),
        Activation('relu'),
        Flatten(),
        Dense(128),
        Activation('relu'),
    ]
