import numpy as np


def tanh(tensor: np.ndarray) -> np.ndarray:
    return np.tanh(tensor)


def sigmoid(tensor: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-tensor))
