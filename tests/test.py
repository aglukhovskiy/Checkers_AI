from rl.experience import load_experience
import h5py
import encoders
import numpy as np
from keras import layers, models
import matplotlib.pyplot as plt
from rl.pg_agent import PolicyAgent, load_policy_agent
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, ZeroPadding2D, Lambda

exp = load_experience(h5py.File('rl/models_n_exp/experience_checkers_reinforce_all_iters_thirteenplane.hdf5'))
encoder = encoders.get_encoder_by_name('thirteenplane')

action_result = exp.action_results[10]

input_shape = (13, 8, 8)

layers = [
        # ZeroPadding2D(padding=3, input_shape=input_shape),  # <1>
        Lambda(lambda x: x, input_shape=input_shape, name="Input"),
        Conv2D(6, (3, 3), padding='same', activation ='relu' , input_shape=input_shape, data_format='channels_first'),
        # Activation('relu'),

        # ZeroPadding2D(padding=2),  # <2>
        # Conv2D(32, (5, 5)),
        # Activation('relu'),
        #
        # ZeroPadding2D(padding=2),
        # Conv2D(32, (5, 5)),
        # Activation('relu'),
        #
        # ZeroPadding2D(padding=2),
        # Conv2D(32, (5, 5)),
        # Activation('relu'),

        Flatten(),
        Dense(32),
        Activation('relu'),
    ]

def create_model(input_shape=(13, 8, 8)):  # Изменяем порядок размерностей
    """Создаёт модель нейронной сети для агента"""
    model = Sequential()
    for layer in layers:
        model.add(layer)
    model.add(Dense(1, activation='linear'))  # Выход - скор
    return model

model = create_model()


layer_activations = [layer.output for layer in model.layers]
model_activations = models.Model(inputs=model.input, outputs=layer_activations)
activations = model_activations.predict(np.array([action_result]))

first_layer_activation = activations[1]

print(first_layer_activation.shape)
print(first_layer_activation[0, 2, :, :])

fig = plt.figure(figsize=(10, 10))
plt.imshow(first_layer_activation[0, 2, :, :], cmap ='viridis')

# ax = fig.add_axes([0, 0, 200, 200])
# ax.set_yticks([])
# ax.set_xticks([])
# ax.imshow(activations[layer_num][0, i, :, :], cmap=cmap
#
fig.savefig('test.png', bbox_inches="tight")

# print(len(activations))
# print(activations[1].shape)

