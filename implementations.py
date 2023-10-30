import numpy as np
from matplotlib import pyplot as plt

# helper functions


def compute_loss(y, tx, w):
    """compute the loss"""
    e = y - tx.dot(w)
    return np.squeeze(e.T.dot(e) / (2 * len(y)))


def compute_loss_likelihood(y, tx, w):
    """compute the cost by negative log likelihood."""
    assert y.shape[0] == tx.shape[0]
    assert tx.shape[1] == w.shape[0]

    pred = sigmoid(tx.dot(w))
    n = y.shape[0]
    return (-1 / n) * np.sum(y * np.log(pred) + (1 - y) * np.log(1 - pred))


def compute_gradient(y, tx, w):
    """compute the gradient of linear regression"""
    e = y - tx.dot(w)
    return (-1) / len(y) * tx.T.dot(e)


def compute_log_gradient(y, tx, w):
    """compute the gradient of logistic regression"""
    N = y.shape[0]
    pred = sigmoid(tx.dot(w))
    grad = 1 / N * tx.T.dot(pred - y)
    return grad


def compute_reg_log_gradient(y, tx, w, lambda_):
    """compute the gradient with L2 regularization"""
    N = y.shape[0]
    pred = sigmoid(tx.dot(w))
    grad = 1 / N * tx.T.dot(pred - y) + 2 * lambda_ * w
    return grad


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """Get a mini batch iterator"""
    data_size = len(y)

    if shuffle:
        idx_shuffled = np.random.permutation(np.arange(data_size))
        y_shuffled = y[idx_shuffled]
        tx_shuffled = tx[idx_shuffled]
    else:
        y_shuffled = y
        tx_shuffled = tx

    for i in range(num_batches):
        st = i * batch_size
        ed = min((i + 1) * batch_size, data_size)
        if st != ed:
            yield y_shuffled[st:ed], tx_shuffled[st:ed]


def sigmoid(z):
    """apply sigmoid function on z."""
    return 1 / (1 + np.exp(-z))


# implementations


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent"""
    w = initial_w
    loss = compute_loss(y, tx, w)

    for _ in range(max_iters):
        grad = compute_gradient(y, tx, w)
        w = w - gamma * grad
        loss = compute_loss(y, tx, w)

    return w, loss


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using sgd"""
    w = initial_w
    loss = compute_loss(y, tx, w)

    for _ in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, 1):
            grad = compute_gradient(y_batch, tx_batch, w)
            w = w - gamma * grad
            loss = compute_loss(y_batch, tx_batch, w)

    return w, loss


def least_squares(y, tx):
    """compute the least squares solution.
    returns mse, and optimal weights.
    """
    A = tx.T.dot(tx)
    B = tx.T.dot(y)
    w = np.linalg.solve(A, B)
    mse = compute_loss(y, tx, w)

    return w, mse


def ridge_regression(y, tx, lambda_):
    """
    Implement ridge regression using normal equations.
    """
    N = tx.shape[0]
    D = tx.shape[1]
    I = np.eye(D)
    A = tx.T.dot(tx) + 2 * N * lambda_ * I
    B = tx.T.dot(y)
    w = np.linalg.solve(A, B)
    # print(A[2,2], lambda_)
    # w, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    loss = compute_loss(y, tx, w)

    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Logistic regression using gradient descent
    """
    w = initial_w
    loss = compute_loss_likelihood(y, tx, w)

    for _ in range(max_iters):
        grad = compute_log_gradient(y, tx, w)
        w = w - gamma * grad
        loss = compute_loss_likelihood(y, tx, w)
    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Regularized logistic regression using gradient descent
    """
    w = initial_w
    loss = compute_loss_likelihood(y, tx, w)

    for _ in range(max_iters):
        grad = compute_reg_log_gradient(y, tx, w, lambda_)
        w = w - gamma * grad
        loss = compute_loss_likelihood(y, tx, w)
    return w, loss
