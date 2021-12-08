import numpy as np


def sigmoid_cost(y_real: np.ndarray, y_pred: np.ndarray) -> float:
    assert y_real.shape == y_pred.shape
    return (-1 / len(y_real)) * np.sum(y_real * np.log(y_pred) + (1 - y_real) * np.log(y_pred))


def tanh_cost(y_real: np.ndarray, y_pred: np.ndarray) -> float:
    assert y_real.shape == y_pred.shape
    return (-1 / len(y_real)) * \
        np.sum((1 + y_real) / 2 * np.log((1 + y_pred) / 2)) + ((1 - y_real) / 2 * np.log((1 - y_pred) / 2))
