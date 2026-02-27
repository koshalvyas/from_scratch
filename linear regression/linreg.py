import numpy as np


def prediction(X, W, b):
    y_hat = np.dot(X, W) + b
    return y_hat


def mse_loss(y, y_hat):
    n = len(y)
    L = (1/n) * np.sum((y_hat - y)**2)
    return L


def initialize_params(n_features):
    W = np.random.randn(n_features) * 0.01
    b = 0
    return W, b


def compute_gradients(X, y, y_hat):
    n = len(y)
    dLW = (2/n) * np.dot(X.T, (y_hat - y))
    dLb = (2/n) * np.sum(y_hat - y)
    return dLW, dLb


def update_params(W, b, dLW, dLb, lr):
    W = W - lr * dLW
    b = b - lr * dLb
    return W, b


def train(X, y, lr, n_epochs):
    W, b = initialize_params(X.shape[1])
    for epoch in range(n_epochs):
        y_hat = prediction(X, W, b)
        dLW, dLb = compute_gradients(X, y, y_hat)
        W, b = update_params(W, b, dLW, dLb, lr)
        if epoch % 100 == 0:
            print(f"Epoch {epoch} | Loss: {mse_loss(y, y_hat):.4f}")
    return W, b


def predict(X, W, b):
    return prediction(X, W, b)
