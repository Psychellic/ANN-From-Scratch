import math

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")
import numpy as np

PRINT = 250
EPOCHS = 30000
INPUT_SIZE = 1000
VALIDATION_SIZE = 300
TESTING_SIZE = 100
NORMALIZE_TO = 1
ANN_MODEL = [1, 15, 15, 15, 1]
BATCH_SIZE = 64
ACTIVATION_FUNCTION = "tanh"
COST_FUNCTION = "MSE"
LEARNING_RATE = 0.0005

OPTIMIZER = "rmsprop"
BETA = 0.9
BETA2 = 0.999
EPSILON = 1e-8
LAMBDA = 0.001


def Cal_Activation_func(X, func):
    if func == "logistic":
        return 1 / (1 + np.exp(-X))
    elif func == "tanh":
        return np.tanh(X)
    elif func == "relu":
        return np.maximum(0, X)
    elif func == "linear":
        return X


def Dif_Activation(X, func):
    if func == "logistic":
        return X * (1 - X)
    elif func == "tanh":
        return 1 - np.square(X)
    elif func == "relu":
        return (X > 0).astype(float)
    elif func == "linear":
        return np.ones_like(X)


def Cost_func(X, Y, func):

    if func == "MSE":
        return np.mean((X - Y) ** 2)


def lr_schedule(epoch, initial_lr):
    return initial_lr * (0.04 ** (epoch // (EPOCHS // 3)))


class NeuralNetwork:
    def __init__(self, architecture):
        self.architecture = architecture
        self.num_layers = len(architecture)
        self.weights = []
        self.biases = []
        self.optimizer = OPTIMIZER
        self.learning_rate = LEARNING_RATE
        self.beta = BETA
        self.beta2 = BETA2
        self.epsilon = EPSILON
        self.lambda_reg = LAMBDA

        for i in range(self.num_layers - 1):
            # Xavier/Glorot initialization for weights
            limit = np.sqrt(6 / (self.architecture[i] + self.architecture[i + 1]))
            self.weights.append(
                np.random.uniform(
                    -limit, limit, (self.architecture[i + 1], self.architecture[i])
                )
            )
            # Initialize biases with small random values
            self.biases.append(np.random.randn(self.architecture[i + 1], 1) * 0.01)

        # Initialize optimizer-related variables (v and s) here as before
        self.v_weights = [np.zeros_like(w) for w in self.weights]
        self.v_biases = [np.zeros_like(b) for b in self.biases]
        self.s_weights = [np.zeros_like(w) for w in self.weights]
        self.s_biases = [np.zeros_like(b) for b in self.biases]

    def forward_prop(self, X, activation_func):
        self.activation = [X]
        for i in range(self.num_layers - 1):
            Z = np.dot(self.weights[i], self.activation[-1]) + self.biases[i]
            if i == self.num_layers - 2:  # Last layer
                A = Z  # Linear activation for output layer
            else:
                A = Cal_Activation_func(Z, activation_func)
            self.activation.append(A)
        return self.activation[-1]

    def backward_prop(self, Y, activation_func, cost_function):
        m = Y.shape[1]
        Y = Y.reshape(self.activation[-1].shape)

        dZ = self.activation[-1] - Y
        dW = [np.zeros_like(w) for w in self.weights]
        db = [np.zeros_like(b) for b in self.biases]

        for i in reversed(range(self.num_layers - 1)):
            dW[i] = (1 / m) * np.dot(dZ, self.activation[i].T) + (
                self.lambda_reg / m
            ) * self.weights[i]
            db[i] = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
            if i > 0:
                dZ = np.dot(self.weights[i].T, dZ) * Dif_Activation(
                    self.activation[i], activation_func
                )

        self.optimize(dW, db)

        # Regularization
        cost = Cost_func(self.activation[-1], Y, cost_function) + (
            self.lambda_reg / (2 * m)
        ) * sum(np.sum(w**2) for w in self.weights)
        return cost

    def optimize(self, dW, db):
        if self.optimizer == "sgd_momentum":
            self.sgd_momentum(dW, db)
        elif self.optimizer == "rmsprop":
            self.rmsprop(dW, db)
        elif self.optimizer == "adam":
            self.adam(dW, db)

    # Momentum
    def sgd_momentum(self, dW, db):
        for i in range(len(self.weights)):
            self.v_weights[i] = self.beta * self.v_weights[i] + (1 - self.beta) * dW[i]
            self.v_biases[i] = self.beta * self.v_biases[i] + (1 - self.beta) * db[i]

            self.weights[i] -= self.learning_rate * self.v_weights[i]
            self.biases[i] -= self.learning_rate * self.v_biases[i]

    def rmsprop(self, dW, db):
        for i in range(len(self.weights)):
            self.s_weights[i] = self.beta * self.s_weights[i] + (1 - self.beta) * (
                dW[i] ** 2
            )
            self.s_biases[i] = self.beta * self.s_biases[i] + (1 - self.beta) * (
                db[i] ** 2
            )

            self.weights[i] -= (
                self.learning_rate / (np.sqrt(self.s_weights[i] + self.epsilon))
            ) * dW[i]
            self.biases[i] -= (
                self.learning_rate / (np.sqrt(self.s_biases[i] + self.epsilon))
            ) * db[i]

    def adam(self, dW, db):
        for i in range(len(self.weights)):
            self.v_weights[i] = self.beta * self.v_weights[i] + (1 - self.beta) * dW[i]
            self.v_biases[i] = self.beta * self.v_biases[i] + (1 - self.beta) * db[i]

            self.s_weights[i] = self.beta2 * self.s_weights[i] + (1 - self.beta2) * (
                dW[i] ** 2
            )
            self.s_biases[i] = self.beta2 * self.s_biases[i] + (1 - self.beta2) * (
                db[i] ** 2
            )

            v_weights_corrected = self.v_weights[i] / (1 - self.beta)
            v_biases_corrected = self.v_biases[i] / (1 - self.beta)

            s_weights_corrected = self.s_weights[i] / (1 - self.beta2)
            s_biases_corrected = self.s_biases[i] / (1 - self.beta2)

            self.weights[i] -= (
                self.learning_rate / (np.sqrt(s_weights_corrected + self.epsilon))
            ) * v_weights_corrected
            self.biases[i] -= (
                self.learning_rate / (np.sqrt(s_biases_corrected + self.epsilon))
            ) * v_biases_corrected


def main():
    # Initialization of the input, output and validation array
    X_input = np.linspace(-2 * math.pi, 2 * math.pi, INPUT_SIZE)
    Y_output = np.sin(X_input)

    # Validation set
    VALIDATION_set = np.random.uniform(
        low=(-2 * math.pi), high=(2 * math.pi), size=VALIDATION_SIZE
    )

    # Normalization
    X_input_min, X_input_max = np.min(X_input), np.max(X_input)
    X_input_normalized = (NORMALIZE_TO - (-NORMALIZE_TO)) * (
        (X_input - X_input_min) / (X_input_max - X_input_min)
    ) + (-NORMALIZE_TO)

    Y_output_min, Y_output_max = np.min(Y_output), np.max(Y_output)
    Y_output_normalized = (NORMALIZE_TO - (-NORMALIZE_TO)) * (
        (Y_output - Y_output_min) / (Y_output_max - Y_output_min)
    ) + (-NORMALIZE_TO)

    X_input_normalized = X_input_normalized.reshape(1, -1)  # Shape: (1, num_samples)
    Y_output_normalized = Y_output_normalized.reshape(1, -1)  # Shape: (1, num_samples)

    # Normalize validation set
    VALIDATION_set_normalized = (NORMALIZE_TO - (-NORMALIZE_TO)) * (
        (VALIDATION_set - X_input_min) / (X_input_max - X_input_min)
    ) + (-NORMALIZE_TO)
    VALIDATION_set_normalized = VALIDATION_set_normalized.reshape(
        1, -1
    )  # Shape: (1, VALIDATION_SIZE)

    ann = NeuralNetwork(ANN_MODEL)

    costs = []  # List to store cost values

    # Batch training
    for i in range(EPOCHS):
        epoch_cost = 0
        for j in range(0, INPUT_SIZE, BATCH_SIZE):
            end = min(j + BATCH_SIZE, INPUT_SIZE)
            X_batch = X_input_normalized[:, j:end]
            Y_batch = Y_output_normalized[:, j:end]

            # Forward Propagation
            ann.forward_prop(X_batch, ACTIVATION_FUNCTION)

            # Backward Propagation
            batch_cost = ann.backward_prop(Y_batch, ACTIVATION_FUNCTION, COST_FUNCTION)
            epoch_cost += batch_cost

            ann.learning_rate = lr_schedule(i, LEARNING_RATE)

        # Calculate average cost for the entire epoch
        epoch_cost /= INPUT_SIZE // BATCH_SIZE
        costs.append(epoch_cost)

        if i % PRINT == 0:
            print(f"EPOCH {i}, Cost function = {epoch_cost}")

    # Plotting the cost function
    plt.figure(figsize=(10, 6))
    plt.plot(range(EPOCHS), costs)
    plt.title("Cost Function over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Cost")
    plt.grid(True)
    plt.savefig("cost_over_epochs.png")
    print("Plot saved as 'cost_over_epochs.png'")

    # Forward pass for validation set
    val_forward = ann.forward_prop(VALIDATION_set_normalized, ACTIVATION_FUNCTION)

    # Denormalize the validation set input and predictions
    VALIDATION_set_denorm = (VALIDATION_set_normalized - (-NORMALIZE_TO)) * (
        X_input_max - X_input_min
    ) / (2 * NORMALIZE_TO) + X_input_min
    val_forward_denorm = (val_forward - (-NORMALIZE_TO)) * (
        Y_output_max - Y_output_min
    ) / (2 * NORMALIZE_TO) + Y_output_min

    # Sort validation set for proper line plot
    sort_indices = np.argsort(VALIDATION_set_denorm.flatten())
    VALIDATION_set_denorm = VALIDATION_set_denorm.flatten()[sort_indices]
    val_forward_denorm = val_forward_denorm.flatten()[sort_indices]

    plt.figure(figsize=(12, 8))

    # Plot original sin wave
    plt.plot(X_input, Y_output, label="True Sin(x)", color="blue")

    # Plot validation set prediction
    plt.scatter(
        VALIDATION_set_denorm,
        val_forward_denorm,
        color="red",
        s=20,
        alpha=0.6,
        label="ANN Prediction",
    )

    plt.title("Sin() Wave and ANN Prediction")
    plt.xlabel("x")
    plt.ylabel("Sin(x)")
    plt.grid(True)
    plt.axhline(y=0, color="k", linestyle="--")
    plt.axvline(x=0, color="k", linestyle="--")
    plt.legend()

    plt.savefig("sin_wave_with_prediction.png")
    print("Plot saved as 'sin_wave_with_prediction.png'")
    plt.show()


if __name__ == "__main__":
    main()
