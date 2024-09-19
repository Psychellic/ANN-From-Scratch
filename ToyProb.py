import math

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")  # Use non-interactive backend
import numpy as np

EPOCHS = 1000

INPUT_SIZE = 1000
VALIDATION_SIZE = 300
TESTING_SIZE = 100

NORMALIZE_TO = 1

ANN_MODEL = [1, 10, 10, 10, 1]
BATCH_SIZE = 100  # Full batch is 1000
ACTIVATION_FUNCTION = "tanh"
COST_FUNCTION = "MSE"

LEARNING_RATE = 0.01


# plotting


def Cal_Activation_func(X, func):
    if func == "logistic":
        return 1 / (1 + np.exp(-X))
    elif func == "tanh":
        return np.tanh(X)
    elif func == "relu":
        return X * (X > 0)
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


class NeuralNetwork:
    def __init__(self, architecture):
        self.architecture = architecture
        self.num_layers = len(architecture)
        self.weights = []
        self.biases = []

        for i in range(0, self.num_layers - 1):
            self.weights.append(
                np.random.randn(self.architecture[i + 1], self.architecture[i])
            )
            self.biases.append(np.random.rand(self.architecture[i + 1], 1))

    def forward_prop(self, X, Activation_func):

        X = X.reshape(-1, 1)
        self.activation = [X]

        for i in range(self.num_layers - 2):  # Use activation for hidden layers
            v = np.dot(self.weights[i], self.activation[-1].T).T + self.biases[i].T
            activation = Cal_Activation_func(v, Activation_func)

            self.activation.append(activation)

        # Linear activation for output layer
        v = np.dot(self.weights[-1], self.activation[-1].T).T + self.biases[-1].T
        self.activation.append(v)  # Linear activation for output

        return self.activation[-1]

    def backward_prop(self, Y, Activation_func):

        Y = Y.reshape(-1, 1)  # Ensure Y is (batch_size, 1)
        P = self.activation[-1]
        self.delta = []

        # Output layer
        delta_output = (Y - P) * Dif_Activation(self.activation[-1], "linear")
        # Assuming linear output activation
        self.delta.append(delta_output)

        # Update weights and biases for the last layer
        self.weights[-1] += (
            LEARNING_RATE * np.dot(delta_output.T, self.activation[-2]) / BATCH_SIZE
        )
        self.biases[-1] += (
            LEARNING_RATE * np.mean(delta_output, axis=0, keepdims=True).T
        )

        # Hidden layers
        for i in range(self.num_layers - 2, 0, -1):
            delta = np.dot(delta_output, self.weights[i]) * Dif_Activation(
                self.activation[i], Activation_func
            )
            self.delta.append(delta)

            # Update weights and biases
            self.weights[i - 1] += (
                LEARNING_RATE * np.dot(delta.T, self.activation[i - 1]) / BATCH_SIZE
            )
            self.biases[i - 1] += (
                LEARNING_RATE * np.mean(delta, axis=0, keepdims=True).T
            )

            delta_output = delta  # For the next iteration

        # Reverse the delta list so it's in forward order
        self.delta = self.delta[::-1]


def main():

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

    # Normalization
    X_input = (NORMALIZE_TO - (-NORMALIZE_TO)) * (
        (X_input - np.min(X_input)) / (np.max(X_input) - np.min(X_input))
    ) + (-NORMALIZE_TO)

    Y_output_min = np.min(Y_output)
    Y_output_max = np.max(Y_output)
    Y_output = (NORMALIZE_TO - (-NORMALIZE_TO)) * (
        (Y_output - Y_output_min) / (Y_output_max - Y_output_min)
    ) + (-NORMALIZE_TO)

    ann = NeuralNetwork(ANN_MODEL)

    costs = []  # List to store cost values

    # Batch
    for i in range(EPOCHS):
        for j in range(0, INPUT_SIZE, BATCH_SIZE):
            end = min(j + BATCH_SIZE, INPUT_SIZE)
            # Forward Propagation
            ann.forward_prop(X_input[j:end], ACTIVATION_FUNCTION)

            # Backward Propagation
            ann.backward_prop(Y_output[j:end], ACTIVATION_FUNCTION)

        # Calculate and store cost for the entire epoch
        forward = ann.forward_prop(X_input, ACTIVATION_FUNCTION)
        epoch_cost = Cost_func(forward, Y_output.reshape(-1, 1), COST_FUNCTION)
        costs.append(epoch_cost)

        if i % 10 == 0:  # Print every 10 epochs
            print(f"EPOCH {i}, Cost function = {epoch_cost}")

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(range(EPOCHS), costs)
    plt.title("Cost Function over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Cost")
    plt.grid(True)

    # Save the plot
    plt.savefig("cost_over_epochs.png")
    print("Plot saved as 'cost_over_epochs.png'")

    VALIDATION_set_normalized = (NORMALIZE_TO - (-NORMALIZE_TO)) * (
        (VALIDATION_set - np.min(VALIDATION_set))
        / (np.max(VALIDATION_set) - np.min(VALIDATION_set))
    ) + (-NORMALIZE_TO)

    # Forward pass for validation set
    val_forward = ann.forward_prop(VALIDATION_set_normalized, ACTIVATION_FUNCTION)

    # Denormalize the validation set input
    VALIDATION_set_denorm = (VALIDATION_set_normalized - (-NORMALIZE_TO)) * (
        np.max(X_input) - np.min(X_input)
    ) / (2 * NORMALIZE_TO) + np.min(X_input)

    # Denormalize the validation set predictions
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


if __name__ == "__main__":
    main()
