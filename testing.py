import json

import matplotlib

matplotlib.use("module://matplotlib-backend-kitty")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Constants
ANN_MODEL = [4, 45, 45, 45, 1]
NORMALIZE_TO = 1
ACTIVATION_FUNCTION = "relu"
OUTPUT_ACTIVATION = "tanh"


def Cal_Activation_func(X, func):
    if func == "relu":
        return np.maximum(0, X)
    elif func == "logistic":
        clip_value = 709  # log(np.finfo(np.float64).max)
        X_clipped = np.clip(X, -clip_value, clip_value)
        return 1 / (1 + np.exp(-X_clipped))
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
        self.output_activation = OUTPUT_ACTIVATION

    def load_weights(self, filename):
        with open(filename, "r") as f:
            weights_data = json.load(f)
        self.weights = [np.array(w) for w in weights_data["weights"]]
        self.biases = [np.array(b) for b in weights_data["biases"]]

    def forward_prop(self, X, hidden_activation):
        self.activation = [X]
        for i in range(self.num_layers - 1):
            Z = np.dot(self.weights[i], self.activation[-1]) + self.biases[i]
            if i == self.num_layers - 2:  # Last layer (output layer)
                A = Cal_Activation_func(Z, self.output_activation)
            else:
                A = Cal_Activation_func(Z, hidden_activation)
            self.activation.append(A)
        return self.activation[-1]


def calculate_mape(actual, predicted, epsilon=0.001):
    n = len(actual)
    mape = (100 / n) * np.sum(np.abs((actual - predicted) / (np.abs(actual) + epsilon)))
    return mape


def calculate_r_squared(actual, predicted):
    ssr = np.sum((actual - predicted) ** 2)
    sst = np.sum((actual - np.mean(actual)) ** 2)
    r_squared = 1 - (ssr / sst)
    return r_squared


def plot_r_squared_scatter(
    Y_val, Y_val_pred, Y_test, Y_test_pred, r_squared_val, r_squared_test
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Validation set scatter plot
    ax1.scatter(Y_val, Y_val_pred, alpha=0.5)
    ax1.plot([Y_val.min(), Y_val.max()], [Y_val.min(), Y_val.max()], "r--", lw=2)
    ax1.set_xlabel("Actual Values")
    ax1.set_ylabel("Predicted Values")
    ax1.set_title(f"Validation Set (R² = {r_squared_val:.4f})")
    ax1.text(
        0.05,
        0.95,
        f"R² = {r_squared_val:.4f}",
        transform=ax1.transAxes,
        verticalalignment="top",
    )

    # Test set scatter plot
    ax2.scatter(Y_test, Y_test_pred, alpha=0.5)
    ax2.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], "r--", lw=2)
    ax2.set_xlabel("Actual Values")
    ax2.set_ylabel("Predicted Values")
    ax2.set_title(f"Test Set (R² = {r_squared_test:.4f})")
    ax2.text(
        0.05,
        0.95,
        f"R² = {r_squared_test:.4f}",
        transform=ax2.transAxes,
        verticalalignment="top",
    )

    plt.tight_layout()
    plt.savefig("r_squared_scatter_plot.png")
    plt.show()
    plt.close()


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

    # Calculate R-squared
    r_squared_val = calculate_r_squared(Y_val.flatten(), Y_val_pred.flatten())
    r_squared_test = calculate_r_squared(Y_test.flatten(), Y_test_pred.flatten())

    print(f"Validation MAPE: {mape_val:.2f}%")
    print(f"Test MAPE: {mape_test:.2f}%")
    print(f"Validation R-squared: {r_squared_val:.4f}")
    print(f"Test R-squared: {r_squared_test:.4f}")

    # Calculate RMS error for test data
    rms_test = np.sqrt(np.mean((Y_test - Y_test_pred) ** 2))
    print(f"Test RMS Error: {rms_test:.4f}")

    # Plot and save R-squared scatter plot
    plot_r_squared_scatter(
        Y_val.flatten(),
        Y_val_pred.flatten(),
        Y_test.flatten(),
        Y_test_pred.flatten(),
        r_squared_val,
        r_squared_test,
    )
    print("R-squared scatter plot saved as 'r_squared_scatter_plot.png'")


if __name__ == "__main__":
    main()
