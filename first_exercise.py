import numpy as np
from sklearn.datasets import make_blobs


def create_dataset():
    """Create a dataset"""
    dataset, expected = make_blobs(
        n_samples=100, n_features=2, centers=2, random_state=0
    )
    expected = expected.reshape((expected.shape[0], 1))
    return dataset, expected


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


def log_loss(A, expected):
    """Log loss"""
    return (
        1
        / len(expected)
        * np.sum(-expected * np.log(A) - (1 - expected) * np.log(1 - A))
    )


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


def model_training(dataset, expected, learning_rate=0.1, n_iter=100):
    """Model training"""
    weights, bias = initialisation(dataset)
    loss = []
    for _ in range(n_iter):
        A = model(dataset, weights, bias)
        loss.append(log_loss(A, expected))
        gradient_w, gradient_b = gradients(A, dataset, expected)
        weights, bias = update(gradient_w, gradient_b, weights, bias, learning_rate)
    return weights, bias, loss
