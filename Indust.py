import json

import matplotlib

matplotlib.use("module://matplotlib-backend-kitty")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Constants
ANN_MODEL = [4, 10, 10, 10, 1]
EPOCHS = 500
BATCH_SIZE = 128
LEARNING_RATE = 0.001
PRINT = 50
NORMALIZE_TO = 1
ACTIVATION_FUNCTION = "relu"
COST_FUNCTION = "MSE"

OPTIMIZER = "adam"
BETA = 0.9
BETA2 = 0.999
EPSILON = 1e-8
LAMBDA = 0.01
DECAY_RATE = 0.001  # Continuous decay rate

PATIENCE = 100


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
    elif func == "MAPE":
        return (1 / BATCH_SIZE) * (np.mean((abs(X - Y) * 100) / abs(Y) + 0.001))


def lr_schedule(epoch, initial_lr):
    # Continuous learning rate decay
    return initial_lr / (1 + DECAY_RATE * epoch)


def save_data(filename, **kwargs):
    with open(filename, "wb") as f:
        np.savez(f, **kwargs)


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
        self.t = 0  # Adam timestep

        for i in range(self.num_layers - 1):
            # He initialization for ReLU activations
            limit = np.sqrt(2 / self.architecture[i])
            self.weights.append(
                np.random.uniform(
                    -limit, limit, (self.architecture[i + 1], self.architecture[i])
                )
            )
            self.biases.append(np.random.randn(self.architecture[i + 1], 1) * 0.01)

        # Initialize moment estimates for Adam/RMSprop/SGD with momentum
        self.v_weights = [np.zeros_like(w) for w in self.weights]
        self.v_biases = [np.zeros_like(b) for b in self.biases]
        self.s_weights = [np.zeros_like(w) for w in self.weights]
        self.s_biases = [np.zeros_like(b) for b in self.biases]

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

    def backward_prop(self, Y, activation_func, cost_function):
        m = Y.shape[1]
        Y = Y.reshape(self.activation[-1].shape)  # Ensure the shapes match

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

        # Calculate cost and include L2 regularization term
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
        self.t += 1  # Increment the timestep for bias correction
        for i in range(len(self.weights)):
            self.v_weights[i] = self.beta * self.v_weights[i] + (1 - self.beta) * dW[i]
            self.v_biases[i] = self.beta * self.v_biases[i] + (1 - self.beta) * db[i]

            self.s_weights[i] = self.beta2 * self.s_weights[i] + (1 - self.beta2) * (
                dW[i] ** 2
            )
            self.s_biases[i] = self.beta2 * self.s_biases[i] + (1 - self.beta2) * (
                db[i] ** 2
            )

            v_weights_corrected = self.v_weights[i] / (1 - self.beta**self.t)
            v_biases_corrected = self.v_biases[i] / (1 - self.beta**self.t)

            s_weights_corrected = self.s_weights[i] / (1 - self.beta2**self.t)
            s_biases_corrected = self.s_biases[i] / (1 - self.beta2**self.t)

            self.weights[i] -= (
                self.learning_rate / (np.sqrt(s_weights_corrected + self.epsilon))
            ) * v_weights_corrected
            self.biases[i] -= (
                self.learning_rate / (np.sqrt(s_biases_corrected + self.epsilon))
            ) * v_biases_corrected

    def save_weights(self, filename):
        weights_data = {
            "weights": [w.tolist() for w in self.weights],
            "biases": [b.tolist() for b in self.biases],
        }
        with open(filename, "w") as f:
            json.dump(weights_data, f)


def main():
    # Load data from the .ods file
    data = pd.read_excel(
        "CCPP/data.ods", engine="odf", sheet_name=0
    )  # Using the first sheet

    # Separate features (X) and target (Y)
    X = data.iloc[:, : ANN_MODEL[0]].values.T  # Shape: (4, num_samples)
    Y = data.iloc[:, ANN_MODEL[0]].values.reshape(1, -1)  # Shape: (1, num_samples)

    # Data splitting (72:18:10 for training:validation:testing)
    total_samples = X.shape[1]
    test_size = int(0.1 * total_samples)
    remaining_size = total_samples - test_size  # This will be split into 72:18
    val_size = int(0.18 * total_samples)  # 18% of total samples for validation
    train_size = remaining_size - val_size

    # Split the data for test (last 10% chunk without shuffling)
    X_test, Y_test = X[:, -test_size:], Y[:, -test_size:]

    # Remaining data (90%) for training and validation
    X_remaining, Y_remaining = X[:, :-test_size], Y[:, :-test_size]

    # Shuffle the remaining data
    indices = np.arange(remaining_size)
    np.random.shuffle(indices)
    X_remaining = X_remaining[:, indices]
    Y_remaining = Y_remaining[:, indices]

    # Initialize arrays for training and validation data
    X_train = []
    X_val = []
    Y_train = []
    Y_val = []

    # Interspersing validation data (indices divisible by 5 go to validation)
    for i in range(remaining_size):
        if i % 5 == 0 and len(X_val) < val_size:
            X_val.append(X_remaining[:, i])
            Y_val.append(Y_remaining[:, i])
        else:
            X_train.append(X_remaining[:, i])
            Y_train.append(Y_remaining[:, i])

    # Convert lists back to numpy arrays
    X_train = np.array(X_train).T
    Y_train = np.array(Y_train).T
    X_val = np.array(X_val).T
    Y_val = np.array(Y_val).T

    # Normalization for input and output
    X_min, X_max = np.min(X_train, axis=1, keepdims=True), np.max(
        X_train, axis=1, keepdims=True
    )
    Y_min, Y_max = np.min(Y_train), np.max(Y_train)

    X_train_norm = (NORMALIZE_TO - (-NORMALIZE_TO)) * (
        (X_train - X_min) / (X_max - X_min)
    ) + (-NORMALIZE_TO)
    X_val_norm = (NORMALIZE_TO - (-NORMALIZE_TO)) * (
        (X_val - X_min) / (X_max - X_min)
    ) + (-NORMALIZE_TO)
    Y_train_norm = (NORMALIZE_TO - (-NORMALIZE_TO)) * (
        (Y_train - Y_min) / (Y_max - Y_min)
    ) + (-NORMALIZE_TO)
    Y_val_norm = (NORMALIZE_TO - (-NORMALIZE_TO)) * (
        (Y_val - Y_min) / (Y_max - Y_min)
    ) + (-NORMALIZE_TO)

    # Save the datasets and normalization parameters
    save_data(
        "datasets.npz",
        X_train=X_train,
        Y_train=Y_train,
        X_val=X_val,
        Y_val=Y_val,
        X_test=X_test,
        Y_test=Y_test,
        X_min=X_min,
        X_max=X_max,
        Y_min=Y_min,
        Y_max=Y_max,
    )
    # Initialize model
    model = NeuralNetwork(ANN_MODEL)
    costs = []  # To store training costs
    val_costs = []  # To store validation costs

    # Training loop with batch processing and validation
    for epoch in range(EPOCHS):
        epoch_cost = 0
        for j in range(0, train_size, BATCH_SIZE):
            end = min(j + BATCH_SIZE, train_size)
            X_batch = X_train_norm[:, j:end]
            Y_batch = Y_train_norm[:, j:end]

            # Forward and Backward Propagation
            model.forward_prop(X_batch, ACTIVATION_FUNCTION)
            batch_cost = model.backward_prop(
                Y_batch, ACTIVATION_FUNCTION, COST_FUNCTION
            )
            epoch_cost += batch_cost

        # Calculate full training cost for the entire epoch (not just batches)
        train_output = model.forward_prop(X_train_norm, ACTIVATION_FUNCTION)
        train_cost = Cost_func(train_output, Y_train_norm, COST_FUNCTION)
        costs.append(train_cost)  # Append full training cost

        # Validation cost (without regularization)
        val_output = model.forward_prop(X_val_norm, ACTIVATION_FUNCTION)
        val_cost = Cost_func(val_output, Y_val_norm, COST_FUNCTION)
        val_costs.append(val_cost)

        # Print progress every PRINT epochs
        if epoch % PRINT == 0:
            print(f"Epoch {epoch}/{EPOCHS} - Cost: {train_cost}, Val Cost: {val_cost}")

        # Shuffle training data at the start of each epoch
        indices = np.arange(train_size)
        np.random.shuffle(indices)
        X_train_norm = X_train_norm[:, indices]
        Y_train_norm = Y_train_norm[:, indices]

        # Update learning rate with decay
        model.learning_rate = lr_schedule(epoch, LEARNING_RATE)

    # Plot the cost over epochs with log scale
    plt.plot(costs, label="Training Cost")
    plt.plot(val_costs, label="Validation Cost")
    plt.xlabel("Epochs")
    plt.ylabel("Cost (Log Scale)")
    plt.yscale("log")  # Set y-axis to log scale
    plt.legend()
    plt.title("Val_vs_Train_cost")
    plt.show()
    plt.savefig("Val_vs_Train_cost(MSE).png")

    model = NeuralNetwork(ANN_MODEL)
    costs = []  # To store training costs
    val_costs = []  # To store validation costs
    best_val_cost = float("inf")
    wait = 0
    best_epoch = 0

    # Training loop with batch processing and validation
    for epoch in range(EPOCHS):
        epoch_cost = 0
        for j in range(0, train_size, BATCH_SIZE):
            end = min(j + BATCH_SIZE, train_size)
            X_batch = X_train_norm[:, j:end]
            Y_batch = Y_train_norm[:, j:end]
            # Forward and Backward Propagation
            model.forward_prop(X_batch, ACTIVATION_FUNCTION)
            batch_cost = model.backward_prop(
                Y_batch, ACTIVATION_FUNCTION, COST_FUNCTION
            )
            epoch_cost += batch_cost

        # Calculate full training cost for the entire epoch (not just batches)
        train_output = model.forward_prop(X_train_norm, ACTIVATION_FUNCTION)
        train_cost = Cost_func(train_output, Y_train_norm, "MAPE")
        costs.append(train_cost)  # Append full training cost

        # Validation cost (without regularization)
        val_output = model.forward_prop(X_val_norm, ACTIVATION_FUNCTION)
        val_cost = Cost_func(val_output, Y_val_norm, "MAPE")
        val_costs.append(val_cost)

        # Print progress every PRINT epochs
        if epoch % PRINT == 0:
            print(f"Epoch {epoch}/{EPOCHS} - Cost: {train_cost}, Val Cost: {val_cost}")

        # Check if the validation cost improved
        if val_cost < best_val_cost:
            best_val_cost = val_cost
            wait = 0
        else:
            wait += 1

        # Early stopping with patience
        if wait >= PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            best_epoch = epoch
            break

        # Shuffle training data at the start of each epoch
        indices = np.arange(train_size)
        np.random.shuffle(indices)
        X_train_norm = X_train_norm[:, indices]
        Y_train_norm = Y_train_norm[:, indices]

        # Update learning rate with decay
        model.learning_rate = lr_schedule(epoch, LEARNING_RATE)

    # Plot the cost over epochs with log scale
    plt.plot(costs, label="Training Cost")
    plt.plot(val_costs, label="Validation Cost")
    plt.xlabel("Epochs")
    plt.ylabel("Cost (Log Scale)")
    plt.yscale("log")  # Set y-axis to log scale
    plt.legend()
    plt.title("Val_vs_Train_cost_with_early_stop")
    plt.show()
    plt.savefig("Val_vs_Train_cost_with_early_stop(MAPE).png")

    model = NeuralNetwork(ANN_MODEL)
    costs = []  # To store training costs
    val_costs = []  # To store validation costs

    for epoch in range(best_epoch):
        epoch_cost = 0
        for j in range(0, train_size, BATCH_SIZE):
            end = min(j + BATCH_SIZE, train_size)
            X_batch = X_train_norm[:, j:end]
            Y_batch = Y_train_norm[:, j:end]

            # Forward and Backward Propagation
            model.forward_prop(X_batch, ACTIVATION_FUNCTION)
            batch_cost = model.backward_prop(
                Y_batch, ACTIVATION_FUNCTION, COST_FUNCTION
            )
            epoch_cost += batch_cost

        # Calculate full training cost for the entire epoch (not just batches)
        train_output = model.forward_prop(X_train_norm, ACTIVATION_FUNCTION)
        train_cost = Cost_func(train_output, Y_train_norm, COST_FUNCTION)
        costs.append(train_cost)  # Append full training cost

        # Validation cost (without regularization)
        val_output = model.forward_prop(X_val_norm, ACTIVATION_FUNCTION)
        val_cost = Cost_func(val_output, Y_val_norm, COST_FUNCTION)
        val_costs.append(val_cost)

        # Print progress every PRINT epochs
        if epoch % PRINT == 0:
            print(f"Epoch {epoch}/{EPOCHS} - Cost: {train_cost}, Val Cost: {val_cost}")

        # Shuffle training data at the start of each epoch
        indices = np.arange(train_size)
        np.random.shuffle(indices)
        X_train_norm = X_train_norm[:, indices]
        Y_train_norm = Y_train_norm[:, indices]

        # Update learning rate with decay
        model.learning_rate = lr_schedule(epoch, LEARNING_RATE)

    model.save_weights("final_weights.json")

    print("Training completed. Weights saved to 'final_weights.json'")


if __name__ == "__main__":
    main()
