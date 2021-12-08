import numpy as np
import json


# ----------------------------------------------------------------------------
#  Logistic Regression
# ----------------------------------------------------------------------------

class LogisticRegression:
    def __init__(self, W: np.ndarray, b: float):
        self.W = W
        self.b = b

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert X.shape[1] == self.W.shape[0]
        return (X @ self.W) + self.b

    def save_weights(self, filename: str):
        assert bool(filename)

        weight = {
            'W': self.W.tolist(),
            'b': self.b
        }

        with open(filename, "w") as file:
            json.dump(weight, file)

    def load_weights(self, filename: str):
        assert bool(filename)

        with open(filename, "r") as file:
            weight = json.load(file)

        self.W = np.array(weight['W'])
        self.b = weight['b']


# ----------------------------------------------------------------------------
#  Activation Functions
# ----------------------------------------------------------------------------

def tanh(tensor: np.ndarray) -> np.ndarray:
    return np.tanh(tensor)


def sigmoid(tensor: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-tensor))


# ----------------------------------------------------------------------------
#  Cost Functions
# ----------------------------------------------------------------------------

def sigmoid_cost(y_real: np.ndarray, y_pred: np.ndarray) -> float:
    assert y_real.shape == y_pred.shape
    return (-1 / len(y_real)) * np.sum(y_real * np.log(y_pred) + (1 - y_real) * np.log(y_pred))


def tanh_cost(y_real: np.ndarray, y_pred: np.ndarray) -> float:
    assert y_real.shape == y_pred.shape
    return (-1 / len(y_real)) * \
           np.sum((1 + y_real) / 2 * np.log((1 + y_pred) / 2)) + ((1 - y_real) / 2 * np.log((1 - y_pred) / 2))
