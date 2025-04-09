from __future__ import absolute_import
from keras.layers.core import Dense, Activation, Flatten


def layers(input_shape):
    return [
        Dense(512, input_shape=input_shape),
        Activation('relu'),
        Dense(512, input_shape=input_shape),
        Activation('relu'),
        Flatten(),
        Dense(512),
        Activation('relu'),
    ]
