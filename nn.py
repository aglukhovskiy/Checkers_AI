import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Conv2D, Flatten, MaxPooling2D, ZeroPadding2D

np.random.seed(123)


X = np.load('features.npy')
Y = np.load('label.npy')

# samples = X.shape[0]
# board_size = 64
# X = X.reshape(samples, board_size)

# Y = Y.reshape(samples, board_size)

samples = X.shape[0]
size = 8
input_shape = (size, size, 1)

train_samples = int(0.9 * samples)
X_train, X_test = X[:train_samples], X[train_samples:]
Y_train, Y_test = Y[:train_samples], Y[train_samples:]

model = Sequential()
model.add(ZeroPadding2D(padding=1, input_shape=input_shape)),
model.add(Conv2D(filters=48,
                 kernel_size=(3, 3),
                 activation='relu',
                 padding='same'))
# model.add(Dropout(rate=0.5))
model.add(Conv2D(48, (3, 3),
                 padding='same',
                 activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(rate=0.5))

# model.add(ZeroPadding2D(padding=3, input_shape=input_shape,data_format='channels_first')),
# model.add(Conv2D(48, (7, 7), data_format='channels_first'),Activation('relu')),
# model.add(ZeroPadding2D(padding=2, data_format='channels_first')),
# model.add(Conv2D(32, (5, 5), data_format='channels_first'), Activation('relu')),
# model.add(ZeroPadding2D(padding=2, data_format='channels_first')),
# model.add(Conv2D(32, (5, 5), data_format='channels_first')),
# model.add(Activation('relu')),
# model.add(ZeroPadding2D(padding=2, data_format='channels_first')),
# model.add(Conv2D(32, (5, 5), data_format='channels_first')),

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(rate=0.2))
model.add(Dense(1, activation='relu'))
model.summary()
model.compile(loss='mean_squared_error',
             optimizer='sgd',
             metrics=['mean_absolute_error'])

model.fit(X_train, Y_train,
         batch_size=32,
         epochs=500,
         verbose=1,
         validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model.save('model_1.h5')  # creates a HDF5 file 'my_model.h5'

model = load_model('model_1.h5')
model.save('model_1.keras')  # creates a HDF5 file 'my_model.h5'

# preds = model.predict(X_test)

# print(preds)