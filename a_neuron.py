import numpy as np

inputs =  [
    [1.0, 2.0, 3.0, 2.5],
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]
]

weights =  [
    [0.2, 0.8, -0.5, 1.0],
    [0.5, -0.91, 0.26, -0.5],
    [-0.26, -0.27, 0.17, 0.87]
]

biases = [2.0, 3.0, 0.5]

layer_outputs = np.dot(inputs, np.array(weights).T) + biases

print(layer_outputs)

import nnfs
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt

nnfs.init()

x, y = spiral_data(samples = 100, classes = 3)
plt.scatter(x[:, 0],x[:, 1])
# plt.show()

from layer_dense import Layer_Dense

dense1 = Layer_Dense(2, 3)
dense1.forward(x)

print(dense1.output[:5])

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
    
activation1 = Activation_ReLU()

activation1.forward(dense1.output)

print(activation1.output[:5])

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis = 1, keepdims = True))
        probabilites = exp_values / np.sum(exp_values, axis = 1, keepdims = True)
        self.output = probabilites

import math

softmax_output = [1, 0, 0]
softmax_output = np.clip(softmax_output, 1e-7, 1 - 1e-7)

target_output = [1, 0, 0]

loss = -(
    math.log(softmax_output[0]) * target_output[0] +
    math.log(softmax_output[1]) * target_output[1] +
    math.log(softmax_output[2]) * target_output[2] 
)

print(loss)

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss
    
class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[
                range(samples),
                y_true
            ]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(
                y_pred_clipped * y_true,
                axis = 1
            )
        negative_log_likelihoods = - np.log(correct_confidences)
        return negative_log_likelihoods

softmax_outputs = np.array([
    [0.7, 0.1, 0.2],
    [0.1, 0.5, 0.4],
    [0.02, 0.9, 0.08]
])
class_targets = np.array([
    [0, 1, 1],
    [0, 1, 0],
    [0, 1, 0]
])

loss_function = Loss_CategoricalCrossentropy()
loss = loss_function.calculate(softmax_outputs, class_targets)
print(loss)

