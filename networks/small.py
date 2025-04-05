from __future__ import absolute_import

# tag::small_network[]
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Conv2D, ZeroPadding2D


def layers(input_shape):
    return [
        ZeroPadding2D(padding=3, input_shape=input_shape),  # <1>
        Conv2D(48, (7, 7)),
        Activation('relu'),

        ZeroPadding2D(padding=2),  # <2>
        Conv2D(32, (5, 5)),
        Activation('relu'),

        ZeroPadding2D(padding=2),
        Conv2D(32, (5, 5)),
        Activation('relu'),

        ZeroPadding2D(padding=2),
        Conv2D(32, (5, 5)),
        Activation('relu'),

        Flatten(),
        Dense(512),
        Activation('relu'),
    ]

# <1> We use zero padding layers to enlarge input images.
# <2> By using `channels_first` we specify that the input plane dimension for our features comes first.
# end::small_network[]
