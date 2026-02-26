import numpy as np


def sigmoid(i):
    return 1 / (1 + np.exp(-i))


def cross_entropy_loss(y, y_hat):
    n = len(y)
    return -(1 / n) * (np.sum((y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))))


def initialize_params(n_feature):
    w = np.zeros(n_feature)
    b = 0
    return w, b


def forward(X, W, b):
    z = np.dot(X, W) + b
    y_hat = sigmoid(z)
    return y_hat


def compute_gradients(X, y, y_hat):
    n = len(y)
    dW = (1 / n) * np.dot(X.T, (y_hat - y))
    db = (1 / n) * np.sum(y_hat - y)
    return dW, db


def update_params(W, b, dW, db, lr):
    W = W - (lr * dW)
    b = b - (lr * db)
    return W, b


def train(X, y, lr, n_epochs):
    W, b = initialize_params(X.shape[1])
    for epoch in range(n_epochs):
        y_hat = forward(
            X,
            W,
            b
        )
        dW, db = compute_gradients(
            X,
            y,
            y_hat
        )
        W, b = update_params(
            W,
            b,
            dW,
            db,
            lr
        )
        if epoch % 100 == 0:
            print(f"Epoch {epoch} | Loss: {cross_entropy_loss(y, y_hat):.4f}")
    return W, b


def predict(X, W, b):
    y_hat = forward(X, W, b)
    y_hat = y_hat >= 0.5
    return y_hat.astype(int)