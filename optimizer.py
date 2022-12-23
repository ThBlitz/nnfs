import numpy as np
import classes
from classes import Layer_Dense, Activation_ReLU, Activation_Softmax, Loss_CategoricalCrossentropy, Activation_Softmax_Loss_CategoricalCrossentropy
import nnfs
from nnfs.datasets import vertical_data, spiral_data
import matplotlib.pyplot as plt

nnfs.init()

# x, y = vertical_data(samples = 100, classes = 3)
# plt.scatter(x[:, 0], x[:, 1], c = y, s = 40, cmap='brg')
# plt.show()


# x, y = vertical_data(samples = 100, classes = 3)

# dense1 = Layer_Dense(2, 3)
# activation1 = Activation_ReLU()
# dense2 = Layer_Dense(3, 3)
# activation2 = Activation_Softmax()

# loss_function = Loss_CategoricalCrossentropy()

# lowest_loss = 9_999_999
# best_dense1_weigths = dense1.weights.copy()
# best_dense1_biases = dense1.biases.copy()
# best_dense2_weights = dense2.weights.copy()
# best_dense2_biases = dense2.biases.copy()

# for iteration in range(100_000):
#     dense1.weights += 0.05 * np.random.randn(2, 3)
#     dense1.biases += 0.05 * np.random.randn(1, 3)
#     dense2.weights += 0.05 * np.random.randn(3, 3)
#     dense2.biases += 0.05 * np.random.randn(1, 3)

#     dense1.forward(x)
#     activation1.forward(dense1.output)
#     dense2.forward(activation1.output)
#     activation2.forward(dense2.output)

#     loss = loss_function.calculate(activation2.output, y)
#     predictions = np.argmax(activation2.output, axis = 1)
#     accuracy = np.mean(predictions == y)

#     if loss < lowest_loss:
#         print("new set of weights at iter:", iteration, "loss:", loss, "acc:", accuracy)
#         best_dense1_weights = dense1.weights.copy()
#         best_dense1_biases = dense1.biases.copy()
#         best_dense2_weigths = dense2.weights.copy()
#         best_dense2_biases = dense2.biases.copy()
#         lowest_loss = loss
#     else:
#         dense1.weights = best_dense1_weights.copy()
#         dense1.biases = best_dense1_biases.copy()
#         dense2.weights = best_dense2_weights.copy()
#         dense2.biases = best_dense2_biases.copy()
    
x, y = spiral_data(samples = 100, classes = 3)

dense1 = Layer_Dense(2, 3)

activation1 = Activation_ReLU()

dense2 = Layer_Dense(3, 3)

loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

dense1.forward(x)

activation1.forward(dense1.output)

dense2.forward(activation1.output)

loss = loss_activation.forward(dense2.output, y)

print(loss_activation.output[:5])

print('loss: ', loss)

predictions = np.argmax(loss_activation.output, axis = 1)
if len(y.shape) == 2:
    y = np.argmax(y, axis = 1)
accuracy = np.mean(predictions == y)

print('acc: ', accuracy)

loss_activation.backward(loss_activation.output, y)
dense2.backward(loss_activation.dinputs)
activation1.backward(dense2.dinputs)
dense1.backward(activation1.dinputs)

print(dense1.dweights)
print(dense1.dbiases)
print(dense2.dweights)
print(dense2.dbiases)

