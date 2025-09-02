# tag::avg_imports[]
import numpy as np
from load_mnist import load_data
from matplotlib import pyplot as plt

# end::avg_imports[]

# tag::average_digit[]
def average_digit(data, digit):  # <1>
    filtered_data = [x[0] for x in data if np.argmax(x[1]) == digit]
    filtered_array = np.asarray(filtered_data)
    return np.average(filtered_array, axis=0)

train, test = load_data()
avg_eight = average_digit(train, 8)  # <2>

img = (np.reshape(avg_eight, (28, 28)))
# plt.imshow(img)
# plt.show()

# tag::eval_eight[]
x_3 = train[2][0]    # <1>
x_18 = train[17][0]  # <2>

img2 = (np.reshape(x_18, (28, 28)))
plt.imshow(img2)
plt.show()

W = np.transpose(avg_eight)
print(np.dot(W, x_3))   # <3>
print(np.dot(W, x_18))  # <4>

def sigmoid_double(x):
    return 1.0 / (1.0 + np.exp(-x))
def sigmoid(z):
    return np.vectorize(sigmoid_double)(z)

def predict(x, W, b):  # <1>
    return sigmoid_double(np.dot(W, x) + b)


b = -45  # <2>

print(predict(x_3, W, b))   # <3>
print(predict(x_18, W, b))  # <4> 0.96