from __future__ import absolute_import
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Conv2D, ZeroPadding2D

def layers(input_shape):
    return [
        Conv2D(32, (7, 7), data_format='channels_first', padding='same', activation='relu'),
        Conv2D(16, (5, 5), data_format='channels_first', padding='same', activation='relu'),
        Flatten(),
        Dense(32, activation='relu'),
    ]