import math

import matplotlib.pyplot as plt
import numpy as np

INPUT_SIZE = 1000
VALIDATION_SIZE = 300
TESTING_SIZE = 100

ANN_MODEL = [1, 10, 10, 10, 1]
BATCH_SIZE = 100  # Full batch is 1000
ACTIVATION_FUNCTION = "logistic"

# Initialization of the input, output and validation array
# b1
X_input = np.zeros(INPUT_SIZE)
Y_output = np.zeros(INPUT_SIZE)

SPLIT_INPUT = (4 * math.pi) / 999
for i in range(INPUT_SIZE):
    X_input[i] = (-2 * math.pi) + (i * SPLIT_INPUT)
    Y_output[i] = math.sin(X_input[i])

# b2
VALIDATION_set = np.random.uniform(
    low=(-2 * math.pi), high=(2 * math.pi), size=VALIDATION_SIZE
)

# plotting
# plt.figure(figsize=(10, 6))
# plt.plot(X_input, Y_output)
# plt.title("Sin() Wave")
# plt.xlabel("x")
# plt.ylabel("Sin")
# plt.grid(True)
# plt.axhline(y=0, color = 'k', linestyle = '--')
# plt.axvline(x=0, color = 'k', linestyle = '--')
# plt.show()


def Cal_Activation_func(X, func):

    if func == "logistic":
        return 1 / (1 + np.exp(-X))

    elif func == "tanh":
        return (np.exp(X) - np.exp(-X)) / (np.exp(X) + np.exp(-X))

    elif func == "relu":
        return X * (X > 0)


class NeuralNetwork:
    def __init__(self, architecture):
        self.architecture = architecture
        self.num_layers = len(architecture)
        self.weights = []
        self.biases = []

        for i in range(0, self.num_layers - 1):
            self.weights.append(
                np.random.rand(self.architecture[i + 1], self.architecture[i])
            )
            self.biases.append(np.random.rand(self.architecture[i + 1], 1))

    def forward_prop(self, X, Activation_func):
        X = X.reshape(-1, 1)
        self.activation = [X]

        for i in range(self.num_layers - 1):
            v = np.dot(self.weights[i], self.activation[-1]) + self.biases[i]
            activation = Cal_Activation_func(v, Activation_func)
            self.activation.append(activation)
        return self.activation[-1]
