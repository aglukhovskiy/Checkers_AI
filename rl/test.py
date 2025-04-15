
import matplotlib.pyplot as plt
from rl.pg_agent import PolicyAgent, load_policy_agent
from keras import layers, models
from experience import load_experience
import h5py
import encoders
import numpy as np

exp_15 = load_experience(h5py.File('models_n_exp/experience_checkers_reinforce_all_iters_small_fiveteenplane_test_cf.hdf5'))
exp_13 = load_experience(h5py.File('models_n_exp/experience_checkers_reinforce_all_iters_thirteenplane.hdf5'))
encoder_15 = encoders.get_encoder_by_name('fiveteenplane')
encoder_13 = encoders.get_encoder_by_name('thirteenplane')

with h5py.File('models_n_exp/test_model_custom_thirteenplane_trained.hdf5', 'r') as agent_file:
    agent = load_policy_agent(agent_file)


layer_names = ["Input",
               "Convolutional #1", "ReLU", "Max Pooling",
               "Convolutional #2", "ReLU", "Max Pooling",
               "Convolutional #3", "ReLU", "----- Flattening -----",
               "Dense #1", "ReLU",
               "Dense #2", "ReLU",
               "Dense #3", "Softmax"]

def predict_save(inp, model, filename=None):
    layer_activations = [layer.output for layer in model.layers]
    model_activations = models.Model(inputs=model.input, outputs=layer_activations)
    activations = model_activations.predict(inp)

    fig = plt.figure(figsize=(10, 10))
    cmap = "coolwarm"
    last_height = 0
    for layer_num, layer in enumerate(layer_activations):
        if layer.shape.rank == 2 and layer.shape[1] < 512: # Dense layers
            width, height = 7.2, 0.034
            x0, y0 = (10 - width ) /2, -height - last_height

            ax = fig.add_axes([x0, y0, width, height])
            ax.set_yticks([])
            ax.set_xticks([])
            ax.imshow([activations[layer_num][0 ,:]], cmap=cmap)
        elif layer.shape.rank == 4: # Convolutional layers
            for i in range(layer.shape[-3]):
                if layer.shape[3] >= 12:
                    width, height = 0.095, 0.095
                else:
                    width, height = 0.2, 0.2
                x0 = i * (width + 0.01) + (10 - layer[3].shape[2] * (width + 0.01)) / 2
                y0 = - height - last_height

                ax = fig.add_axes([x0, y0, width, height])
                ax.set_yticks([])
                ax.set_xticks([])
                ax.imshow(activations[layer_num][0, i ,: ,: ], cmap=cmap)
        else:
            height = -0.01

        fig.text(5, - last_height + 0.03, layer_names[layer_num], fontdict={"size" :16}, horizontalalignment='center', verticalalignment='center')
        last_height += 0.06 + height

    if not filename:
        plt.show()
    else:
        fig.savefig(filename, bbox_inches="tight")
    plt.clf()
    plt.close()

inp = exp_13.action_results[10] #.reshape(1,15,8,8)
print(inp.shape)
encoder_15.show_board_from_matrix(inp)

predict_save(np.array([inp]), agent._model, 'test.png')

# print(agent._model.input)

layer_activations = [layer.output for layer in agent._model.layers]
model_activations = models.Model(inputs=agent._model.input, outputs=layer_activations)
activations = model_activations.predict(np.array([inp]))

print(len(activations))
print(activations[0].shape)
print(activations[1].shape)
print(activations[2].shape)
print(activations[3].shape)
print(activations[4].shape)

# fig = plt.figure(figsize=(10, 10))
# predict_save(np.array([inp]), agent._model, 'test.png')