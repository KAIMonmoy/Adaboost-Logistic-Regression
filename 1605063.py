import enum
from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score


# ----------------------------------------------------------------------------
#  Strategy
# ----------------------------------------------------------------------------

class ActivationStrategy(enum.Enum):
    SIGMOID = 0
    TANH = 1


# ----------------------------------------------------------------------------
#  Logistic Regression
# ----------------------------------------------------------------------------

class LogisticRegression:
    def __init__(self, W: np.ndarray, activation_strategy: ActivationStrategy):
        self.W = W
        if activation_strategy == ActivationStrategy.TANH:
            self.activation_function = tanh_activation
            self.cost_function = tanh_cost
            self.gradient_function = tanh_gradient
        else:
            self.activation_function = sigmoid_activation
            self.cost_function = sigmoid_cost
            self.gradient_function = sigmoid_gradient

    def train(self, features: np.ndarray, target: np.ndarray,
              learning_rate: float = 1e-3, epoch: int = 100, earlystop_acc: float = 1.0) -> Dict:
        X = np.insert(features, 0, 1.0, axis=1)
        Y = target

        if self.W is None:
            self.W = np.random.random((X.shape[1] + 1, 1))

        assert X.shape[1] == self.W.shape[0]

        history = {
            'cost': [],
            'accuracy': []
        }

        for i in range(epoch):
            A = self.activation_function(X @ self.W)

            cost = self.cost_function(Y, A)
            accuracy = accuracy_score(Y, A)

            if accuracy > earlystop_acc:
                print(f"Early Stopping: Accuracy ({accuracy}) > EarlyStop Accuracy({earlystop_acc})")
                break

            gradient = self.gradient_function(X, Y, A)
            self.W -= (learning_rate * gradient)

            if (i + 1) % 5 == 0:
                print(f"Epoch {i + 1}: Accuracy={accuracy}, Loss={cost}")

            history['cost'].append(cost)
            history['accuracy'].append(accuracy)

            return history

    def predict(self, features: np.ndarray) -> np.ndarray:
        assert self.W

        X = np.insert(features, 0, 1.0, axis=1)
        assert X.shape[1] == self.W.shape[0]

        return self.activation_function(X @ self.W)

    def save_weights(self, filename: str):
        assert bool(filename) and type(filename) is str
        assert filename.endswith(".npy")

        with open(filename, "wb") as file:
            np.save(file, self.W)

    def load_weights(self, filename: str):
        assert bool(filename) and type(filename) is str
        assert filename.endswith(".npy")

        with open(filename, "rb") as file:
            self.W = np.load(file)


# ----------------------------------------------------------------------------
#  Activation Functions
# ----------------------------------------------------------------------------

def tanh_activation(tensor: np.ndarray) -> np.ndarray:
    return np.tanh(tensor)


def sigmoid_activation(tensor: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-tensor))


# ----------------------------------------------------------------------------
#  Cost Functions
# ----------------------------------------------------------------------------

def sigmoid_cost(y_real: np.ndarray, y_pred: np.ndarray) -> float:
    assert y_real.shape == y_pred.shape
    return -1 * np.mean(y_real * np.log(y_pred) + (1 - y_real) * np.log(y_pred))


def tanh_cost(y_real: np.ndarray, y_pred: np.ndarray) -> float:
    assert y_real.shape == y_pred.shape
    return -1 * np.mean((1 + y_real) / 2 * np.log((1 + y_pred) / 2)) + ((1 - y_real) / 2 * np.log((1 - y_pred) / 2))


# ----------------------------------------------------------------------------
#  Gradient Functions
# ----------------------------------------------------------------------------

def sigmoid_gradient(X: np.ndarray, Y: np.ndarray, A: np.ndarray) -> np.ndarray:
    assert Y.shape == A.shape
    assert X.shape[0] == Y.shape[0]

    return np.mean(X * (A - Y), axis=0).reshape(-1, 1)


def tanh_gradient(X: np.ndarray, Y: np.ndarray, A: np.ndarray) -> np.ndarray:
    assert Y.shape == A.shape
    assert X.shape[0] == Y.shape[0]

    return np.mean(X * (A - Y), axis=0).reshape(-1, 1)
