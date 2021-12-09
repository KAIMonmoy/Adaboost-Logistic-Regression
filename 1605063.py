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
    def __init__(self, activation_strategy: ActivationStrategy = ActivationStrategy.SIGMOID):
        self.W = None
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
            self.W = np.random.random((X.shape[1], 1))

        assert X.shape[1] == self.W.shape[0]

        history = {
            'cost': [],
            'accuracy': []
        }

        for i in range(epoch):
            A = self.activation_function(X @ self.W)

            cost = self.cost_function(Y, A)
            accuracy = accuracy_score(Y, np.round(A).astype(int))

            if (i + 1) % 25 == 0:
                print(f"Epoch {i + 1}: Accuracy={accuracy}, Loss={cost}")

            if accuracy > earlystop_acc:
                print(f"\nEarly Stopping @ {i+1}th Epoch\nAccuracy ({accuracy}) > EarlyStopAccuracy({earlystop_acc})\n")
                break

            gradient = self.gradient_function(X, Y, A)
            self.W -= (learning_rate * gradient)

            history['cost'].append(cost)
            history['accuracy'].append(accuracy)

        return history

    def predict(self, features: np.ndarray) -> np.ndarray:
        assert self.W is not None

        X = np.insert(features, 0, 1.0, axis=1)
        assert X.shape[1] == self.W.shape[0]

        return np.round(self.activation_function(X @ self.W)).astype(int)

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
    assert y_real.shape == y_pred.shape, f"{y_real.shape}, {y_pred.shape}"
    return -1 * np.mean(y_real * np.log(y_pred) + (1 - y_real) * np.log(1 - y_pred))


def tanh_cost(y_real: np.ndarray, y_pred: np.ndarray) -> float:
    assert y_real.shape == y_pred.shape, f"{y_real.shape}, {y_pred.shape}"
    return -1 * np.mean((1 + y_real) / 2 * np.log((1 + y_pred) / 2) + (1 - y_real) / 2 * np.log((1 - y_pred) / 2))


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


# ----------------------------------------------------------------------------
#  Main
# ----------------------------------------------------------------------------

def main():
    from sklearn import datasets
    feat, tgt = datasets.make_classification(200, 4, random_state=94)
    tgt = tgt.reshape(-1, 1)

    f_t, t_t = feat[:150], tgt[:150]
    f_v, t_v = feat[150:], tgt[150:]

    learner = LogisticRegression(ActivationStrategy.TANH)

    history = learner.train(f_t, t_t, epoch=5000, earlystop_acc=.8)
    y_p = learner.predict(f_v)
    print('Validation Accuracy', accuracy_score(t_v, y_p))

    import matplotlib.pyplot as plt
    plt.plot(history['accuracy'])
    plt.plot(history['cost'])
    plt.show()


if __name__ == "__main__":
    main()
