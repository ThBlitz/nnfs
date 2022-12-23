import numpy as np
import nnfs
import classes 
from classes import Layer_Dense, Activation_ReLU, Activation_Softmax, Loss_CategoricalCrossentropy


softmax_outputs = np.array([
    [0.7, 0.1, 0.2],
    [0.1, 0.5, 0.4],
    [0.02, 0.9, 0.08]
])

class_targets = np.array([0, 1, 1])

softmax_loss = Activa