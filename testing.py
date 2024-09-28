import json

import numpy as np
import pandas as pd

# Constants
ANN_MODEL = [4, 10, 10, 10, 1]
NORMALIZE_TO = 1
ACTIVATION_FUNCTION = "relu"


def Cal_Activation_func(X, func):
    if func == "relu":
        return np.maximum(0, X)
    elif func == "logistic":
        return 1 / (1 + np.exp(-X))
    elif func == "tanh":
        return np.tanh(X)
    elif func == "linear":
        return X


class NeuralNetwork:
    def __init__(self, architecture):
        self.architecture = architecture
        self.num_layers = len(architecture)
        self.weights = []
        self.biases = []

    def load_weights(self, filename):
        with open(filename, "r") as f:
            weights_data = json.load(f)
        self.weights = [np.array(w) for w in weights_data["weights"]]
        self.biases = [np.array(b) for b in weights_data["biases"]]

    def forward_prop(self, X, activation_func):
        self.activation = [X]
        for i in range(self.num_layers - 1):
            Z = np.dot(self.weights[i], self.activation[-1]) + self.biases[i]
            if i == self.num_layers - 2:  # Last layer (output layer)
                A = Z  # Linear activation for regression
            else:
                A = Cal_Activation_func(Z, activation_func)
            self.activation.append(A)
        return self.activation[-1]


def calculate_mape(actual, predicted, epsilon=0.001):
    n = len(actual)
    mape = (100 / n) * np.sum(np.abs((actual - predicted) / (np.abs(actual) + epsilon)))
    return mape


def main():
    # Load saved datasets and normalization parameters
    with np.load("datasets.npz") as data:
        X_train, Y_train = data["X_train"], data["Y_train"]
        X_val, Y_val = data["X_val"], data["Y_val"]
        X_test, Y_test = data["X_test"], data["Y_test"]
        X_min, X_max = data["X_min"], data["X_max"]
        Y_min, Y_max = data["Y_min"], data["Y_max"]

    # Normalize the data
    X_val_norm = (NORMALIZE_TO - (-NORMALIZE_TO)) * (
        (X_val - X_min) / (X_max - X_min)
    ) + (-NORMALIZE_TO)
    X_test_norm = (NORMALIZE_TO - (-NORMALIZE_TO)) * (
        (X_test - X_min) / (X_max - X_min)
    ) + (-NORMALIZE_TO)

    # Load model and weights
    model = NeuralNetwork(ANN_MODEL)
    model.load_weights("final_weights.json")

    # Forward propagation for validation and test sets
    Y_val_pred_norm = model.forward_prop(X_val_norm, ACTIVATION_FUNCTION)
    Y_test_pred_norm = model.forward_prop(X_test_norm, ACTIVATION_FUNCTION)

    # Denormalize predictions
    Y_val_pred = Y_min + (Y_val_pred_norm - (-NORMALIZE_TO)) * (Y_max - Y_min) / (
        NORMALIZE_TO - (-NORMALIZE_TO)
    )
    Y_test_pred = Y_min + (Y_test_pred_norm - (-NORMALIZE_TO)) * (Y_max - Y_min) / (
        NORMALIZE_TO - (-NORMALIZE_TO)
    )

    # Calculate MAPE
    mape_val = calculate_mape(Y_val.flatten(), Y_val_pred.flatten())
    mape_test = calculate_mape(Y_test.flatten(), Y_test_pred.flatten())

    print(f"Validation MAPE: {mape_val:.2f}%")
    print(f"Test MAPE: {mape_test:.2f}%")

    # Calculate RMS error for test data
    rms_test = np.sqrt(np.mean((Y_test - Y_test_pred) ** 2))
    print(f"Test RMS Error: {rms_test:.4f}")


if __name__ == "__main__":
    main()
