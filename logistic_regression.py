import numpy as np
import json


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
