import h5py
import numpy as np
from sklearn.metrics import accuracy_score, log_loss
from tqdm import tqdm


def load_data():
    train_dataset = h5py.File("datasets/trainset.hdf5", "r")
    X_train = np.array(train_dataset["X_train"][:])  # your train set features
    y_train = np.array(train_dataset["Y_train"][:])  # your train set labels

    test_dataset = h5py.File("datasets/testset.hdf5", "r")
    X_test = np.array(test_dataset["X_test"][:])  # your train set features
    y_test = np.array(test_dataset["Y_test"][:])  # your train set labels

    return X_train, y_train, X_test, y_test


def display_pet(value: float) -> str:
    return "cat" if value == 0 else "dog"


def normalize_and_flat(data):
    return (data / 255).reshape((data.shape[0], -1))


def initialisation(dataset):
    """Initialize neuron"""
    weights = np.random.randn(dataset.shape[1], 1)
    bias = np.random.randn(1)
    return weights, bias


def model(dataset, weights, bias):
    """Process model"""
    Z = dataset.dot(weights) + bias
    A = 1 / (1 + np.exp(-Z))
    return A


def gradients(A, dataset, expected):
    """Computes the gradients"""
    gradient_w = 1 / len(expected) * np.dot(dataset.T, A - expected)
    gradient_b = 1 / len(expected) * np.sum(A - expected)
    return gradient_w, gradient_b


def update(gradient_w, gradient_b, weights, bias, learning_rate):
    """Update model"""
    weights = weights - learning_rate * gradient_w
    bias = bias - learning_rate * gradient_b
    return weights, bias


def predict(dataset, weights, bias):
    """Predict"""
    A = model(dataset, weights, bias)
    return A >= 0.5


def model_training(
    dataset, expected, dataset_test, expected_test, learning_rate=0.1, n_iter=100
):
    """Model training"""
    weights, bias = initialisation(dataset)
    loss = []
    acc = []
    loss_test = []
    acc_test = []
    for i in tqdm(range(n_iter)):
        A = model(dataset, weights, bias)
        if i % 100 == 0:
            loss.append(log_loss(expected, A))
            y_pred = predict(dataset, weights, bias)
            acc.append(accuracy_score(expected, y_pred))

            A_test = model(dataset_test, weights, bias)
            loss_test.append(log_loss(expected_test, A_test))
            y_pred = predict(dataset_test, weights, bias)
            acc_test.append(accuracy_score(expected_test, y_pred))
        gradient_w, gradient_b = gradients(A, dataset, expected)
        weights, bias = update(gradient_w, gradient_b, weights, bias, learning_rate)
    return weights, bias, loss, acc, loss_test, acc_test
