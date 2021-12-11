import enum
from typing import Dict, List
import random
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

SEED = 0
random.seed(SEED)
np.random.seed(SEED)


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
        self.activation_strategy = activation_strategy
        if activation_strategy == ActivationStrategy.TANH:
            self.activation_function = tanh_activation
            self.cost_function = tanh_cost
            self.gradient_function = tanh_gradient
        else:
            self.activation_function = sigmoid_activation
            self.cost_function = sigmoid_cost
            self.gradient_function = sigmoid_gradient

    def train(self, features: np.ndarray, target: np.ndarray,
              learning_rate: float = 1e-3, epoch: int = 100, earlystop_error: float = 0.0) -> Dict:
        X = np.insert(features, 0, 1.0, axis=1)
        Y = target

        if self.W is None:
            self.W = np.zeros((X.shape[1], 1))

        assert X.shape[1] == self.W.shape[0]

        history = {
            'cost': [],
            'accuracy': []
        }

        for i in range(epoch):
            A = self.activation_function(X @ self.W)

            cost = self.cost_function(Y, A)

            if self.activation_strategy == ActivationStrategy.TANH:
                accuracy = accuracy_score(Y, np.where(A >= 0, 1, -1))
            else:
                accuracy = accuracy_score(Y, np.where(A >= 0.5, 1, 0))

            if (i + 1) % 100 == 0:
                print(f"Epoch {i + 1}: Accuracy={accuracy}, Loss={cost}")

            if cost < earlystop_error:
                print(
                    f"\nEarly Stopping @ {i + 1}th Epoch\n"
                    f"Error ({cost}) < EarlyStopError ({earlystop_error})\n"
                )
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

        if self.activation_strategy == ActivationStrategy.TANH:
            return np.where(self.activation_function(X @ self.W) >= 0, 1, -1)

        return np.where(self.activation_function(X @ self.W) >= 0.5, 1, 0)

    def predict_probability(self, features: np.ndarray) -> np.ndarray:
        assert self.W is not None

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
#  AdaBoost
# ----------------------------------------------------------------------------

class AdaBoost:
    def __init__(self, l_weak, activation_strategy):
        self.WeakLearner = l_weak
        self.activation_strategy = activation_strategy

        self.weights = None
        self.predictors = None

    def train(self, X: np.ndarray, Y: np.ndarray, n_round: int = 5, learning_rate: float = 1e-2):
        self.predictors = []
        self.weights = []

        n = len(X)
        w = np.full(n, 1 / n)
        all_ids = np.arange(n)

        for i in range(n_round):
            print(f"Adaboost Round: {i + 1}")
            # Resample Data based on "w" probability
            sampled_ids = np.random.choice(all_ids, n, p=w)
            print('Unique IDs', len(np.unique(sampled_ids)))
            X_sampled = X[sampled_ids]
            Y_sampled = Y[sampled_ids]

            # Train Weak Learner on Sampled Data
            learner = self.WeakLearner(self.activation_strategy)
            learner.train(X_sampled, Y_sampled, learning_rate=learning_rate, epoch=10000, earlystop_error=0.5)

            # Test Weak Learner's Performance on Training Dataset
            Y_predicted = learner.predict(X)
            correct = (Y == Y_predicted).ravel().astype(float)
            wrong = (1 - correct)

            # Calculate Error
            error = np.sum(wrong * w)
            print(f'\nError @ Round-{i + 1}: {error}\n')
            if error > 0.5:
                continue

            # Update Weight
            w_m = error / (1 - error)
            w = (w_m * correct * w) + (wrong * w)
            w /= w.sum()

            # Store Learner and Learner's Weight
            self.predictors.append(learner)
            self.weights.append(np.log(1 / w_m))

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.predictors is not None
        assert self.weights is not None
        assert len(self.predictors) == len(self.weights)

        predictions = np.full((len(X), 1), 0.0)

        k = len(self.predictors)
        for i in range(k):
            predictions += (self.weights[i] * self.predictors[i].predict_probability(X))

        return np.where(predictions >= 0, 1, -1)


# ----------------------------------------------------------------------------
#  Dataset
# ----------------------------------------------------------------------------


class CustomDataset(ABC):
    @abstractmethod
    def get_training_set(self):
        pass

    @abstractmethod
    def get_testing_set(self):
        pass


def calculate_binary_entropy(x: float) -> float:
    if x == 0.0 or x == 1.0:
        return 0.0
    return -x * np.log2(x) - (1 - x) * np.log2((1 - x))


def calculate_information_gain(dataframe: pd.DataFrame, feature: str, target: str) -> float:
    t = len(dataframe)
    y = len(dataframe[dataframe[target] == 1])

    entropy = calculate_binary_entropy(y / t)

    remainder = 0
    for value in dataframe[feature].unique():
        kt = len(dataframe[dataframe[feature] == value])
        ky = len(dataframe[(dataframe[feature] == value) & (dataframe[target] == 1)])
        remainder += ((kt / t) * calculate_binary_entropy(ky / kt))

    return entropy - remainder


def process_target_feature(dataframe: pd.DataFrame, target: str, p_val, n_val) -> np.ndarray:
    dataframe[target].replace(p_val, 1, inplace=True)
    dataframe[target].replace(n_val, -1, inplace=True)
    dataframe[target] = dataframe[target].astype(int)
    return dataframe[target].values.reshape(-1, 1)


def encode_feature_labels(dataframe: pd.DataFrame, categorical_feature: List):
    for feature in categorical_feature:
        dataframe[feature] = dataframe[feature].astype('category').cat.codes


def convert_continuous_features_to_categorical(dataframe: pd.DataFrame, continuous_features: List[str]):
    for feature in continuous_features:
        feature_info = dataframe[feature].describe()
        feature_bins = [
            feature_info['min'] - 1.0,
            feature_info['25%'],
            feature_info['50%'],
            feature_info['75%'],
            feature_info['max'] + 1.0,
        ]
        feature_labels = [bin_value for bin_value in range(len(feature_bins) - 1)]
        dataframe[feature] = pd.cut(dataframe[feature], feature_bins, labels=feature_labels)
        dataframe[feature] = dataframe[feature].astype(int)


def get_selected_features(df: pd.DataFrame, info_gain_threshold: float) -> List:
    return [feature for feature in df.columns[:-1] \
            if calculate_information_gain(df, feature, df.columns[-1]) > info_gain_threshold]


class TelcoChurnDataset(CustomDataset):
    def __init__(self, filename: str):
        df = pd.read_csv(filename)

        # Handle Null Values
        df.dropna(inplace=True)

        # Convert Target Feature
        Y = process_target_feature(df, 'Churn', 'Yes', 'No')

        # Dataset Specific Preprocessing
        df.drop('customerID', axis=1, inplace=True)

        df.TotalCharges = pd.to_numeric(df.TotalCharges, errors='coerce')
        df.TotalCharges = df.TotalCharges.fillna(0)

        # Encode Labels of Features
        categorical_features = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
                                'MultipleLines', 'InternetService', 'OnlineSecurity',
                                'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                                'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']
        encode_feature_labels(df, categorical_features)

        # Convert Continuous Features to Categorical Features
        continuous_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
        convert_continuous_features_to_categorical(df, continuous_features)

        # Feature Selection based on Information Gain
        selected_features = get_selected_features(df, info_gain_threshold=0.03999)
        df = df[selected_features]

        # One-hot Encoding
        one_hot_encodable_features = [feature for feature in df.columns if len(df[feature].unique()) > 2]
        df = pd.get_dummies(df, prefix=one_hot_encodable_features, columns=one_hot_encodable_features, drop_first=True)

        X = df.to_numpy(dtype=float)

        # Clear Memory
        del df
        del categorical_features
        del continuous_features
        del selected_features
        del one_hot_encodable_features

        # Stratified Train-Test Split
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X, Y,
                                                                                test_size=0.2,
                                                                                random_state=SEED,
                                                                                stratify=Y.ravel())

    def get_training_set(self):
        return self.X_train, self.Y_train

    def get_testing_set(self):
        return self.X_test, self.Y_test


class AdultIncomeDataset(CustomDataset):
    def __init__(self):
        self.X_train, self.X_test, self.Y_train, self.Y_test = None, None, None, None

    def get_training_set(self):
        return self.X_train, self.Y_train

    def get_testing_set(self):
        return self.X_test, self.Y_test


class CreditCardFraudDataset(CustomDataset):
    def __init__(self):
        self.X_train, self.X_test, self.Y_train, self.Y_test = None, None, None, None

    def get_training_set(self):
        return self.X_train, self.Y_train

    def get_testing_set(self):
        return self.X_test, self.Y_test


# ----------------------------------------------------------------------------
#  Main
# ----------------------------------------------------------------------------

def main():
    t_dataset = TelcoChurnDataset('data/Telco Churn/TelcoChurn.csv')

    f_t, t_t = t_dataset.get_training_set()
    f_v, t_v = t_dataset.get_testing_set()

    # learner = LogisticRegression(activation_strategy=ActivationStrategy.TANH)
    #
    # learner.train(f_t, t_t, learning_rate=1e-3, epoch=10000, earlystop_error=0.1)
    # print('LR Test Accuracy', accuracy_score(t_v, learner.predict(f_v)), '\n')

    adaboost = AdaBoost(LogisticRegression, ActivationStrategy.TANH)
    adaboost.train(f_t, t_t, 5, learning_rate=7.5e-4)

    print('AdaBoost Test Accuracy', accuracy_score(t_v, adaboost.predict(f_v)), '\n')


if __name__ == "__main__":
    main()
