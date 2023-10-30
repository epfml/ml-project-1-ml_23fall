import numpy as np
from matplotlib import pyplot as plt
from implementations import *


def compute_loss_likelihood_v2(y, tx, w):
    """compute the cost by negative log likelihood for labels 1 and -1."""
    assert y.shape[0] == tx.shape[0]
    assert tx.shape[1] == w.shape[0]

    # Convert labels from -1,1 to 0,1 since logistic regression expects labels in the 0,1 range for loss computation.
    y_converted = (y + 1) / 2
    pred = sigmoid(tx.dot(w))
    pred = pred.squeeze()
    n = y.shape[0]
    return (-1 / n) * np.sum(
        y_converted * np.log(pred) + (1 - y_converted) * np.log(1 - pred)
    )


def compute_reg_log_gradient_v2(y, tx, w, lambda_):
    """compute the gradient with L2 regularization for labels 1 and -1."""
    N = y.shape[0]
    pred = sigmoid(tx.dot(w)).squeeze()

    gradient_part = tx.T.dot(pred - y).reshape(-1, 1)
    regularization_part = 2 * lambda_ * w
    grad = 1 / N * gradient_part + regularization_part

    return grad


def reg_logistic_regression_v2(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Regularized logistic regression using gradient descent for labels 1 and -1.
    """
    w = initial_w
    loss = compute_loss_likelihood_v2(y, tx, w)

    for _ in range(max_iters):
        grad = compute_reg_log_gradient_v2(y, tx, w, lambda_)
        w = w - gamma * grad
        loss = compute_loss_likelihood_v2(y, tx, w)
    return w, loss


def build_k_fold(y, k, seed):
    """Build index for k-fold"""
    n = y.shape[0]
    interval = n // k
    np.random.seed(seed)
    idx = np.random.permutation(n)  # shuffle
    k_fold = [idx[i * interval : (i + 1) * interval] for i in range(k)]
    return np.array(k_fold)


def poly_extension(x, d):
    """extend the features by (x, x^2, ..., x^d) and add a all-1 column at first"""
    extension = np.column_stack([x**i for i in range(1, d + 1)])
    tmp = extension.reshape(x.shape[0], -1)
    x_ext = np.c_[np.ones(tmp.shape[0]), tmp]
    return x_ext


def predict_labels(weights, data):
    # For ridge regression
    # y_pred = np.dot(data, weights)
    # y_pred[np.where(y_pred <= 0)] = -1
    # y_pred[np.where(y_pred > 0)] = 1

    # For logistic regression
    y_pred_proba = sigmoid(np.dot(data, weights))
    y_pred = np.ones(y_pred_proba.shape)
    y_pred[y_pred_proba <= 0.5] = -1

    return y_pred


def f1_score(y_true, y_pred):
    """Compute the F1 score."""
    y_pred = y_pred.squeeze()
    # True positive
    TP = np.sum((y_true == 1) & (y_pred == 1))

    # False positive
    FP = np.sum((y_true == -1) & (y_pred == 1))

    # False negative
    FN = np.sum((y_true == 1) & (y_pred == -1))

    # Precision and Recall
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)

    # F1 score
    f1 = 2 * precision * recall / (precision + recall)

    return f1


def acc(y_true, y_pred):
    """Compute the accuracy."""
    y_pred = y_pred.squeeze()
    # True positive
    t1 = np.sum((y_true == 1) & (y_pred == 1))
    t2 = np.sum((y_true == -1) & (y_pred == -1))

    return (t1 + t2) / y_pred.shape[0]


def cross_validation(y, x_ext, k_fold, lambda_):
    """perform k-fold cross validation and return the train/test loss"""
    train_loss = []
    test_loss = []
    train_f1 = []
    test_f1 = []
    w = []
    # x_ext = poly_extension(x, degree)
    k = len(k_fold)

    for i in range(0, k):
        test_idx = k_fold[i]
        x_test = x_ext[test_idx, :]
        y_test = y[test_idx]

        train_idx = np.array([i not in test_idx for i in range(x_ext.shape[0])])
        x_train = x_ext[train_idx, :]
        y_train = y[train_idx]

        num_features = x_train.shape[1]
        initial_w = np.zeros((num_features, 1))
        w_k, train_loss_k = reg_logistic_regression_v2(
            y_train, x_train, lambda_, initial_w, 2000, 0.001
        )
        test_loss_k = compute_loss_likelihood_v2(y_test, x_test, w_k)

        y_train_pred = predict_labels(w_k, x_train)
        y_test_pred = predict_labels(w_k, x_test)
        train_f1.append(f1_score(y_train, y_train_pred))
        test_f1.append(f1_score(y_test, y_test_pred))

        train_loss.append(train_loss_k)
        test_loss.append(test_loss_k)
        w.append(w_k)

    loss_tr = sum(train_loss) / k
    loss_te = sum(test_loss) / k
    f1_train = sum(train_f1) / k
    f1_test = sum(test_f1) / k
    ww = np.array(w)
    weight = np.mean(ww, axis=0)

    return loss_tr, loss_te, f1_train, f1_test, weight


def cross_validation_rr(y, x_ext, k_fold, lambda_):
    """perform k-fold cross validation and return the train/test loss of ridge_regression"""
    train_loss = []
    test_loss = []
    train_f1 = []
    test_f1 = []
    w = []
    # x_ext = poly_extension(x, degree)
    k = len(k_fold)

    for i in range(0, k):
        test_idx = k_fold[i]
        x_test = x_ext[test_idx, :]
        y_test = y[test_idx]

        train_idx = np.array([i not in test_idx for i in range(x_ext.shape[0])])
        x_train = x_ext[train_idx, :]
        y_train = y[train_idx]

        num_features = x_train.shape[1]
        initial_w = np.zeros((num_features, 1))
        w_k, train_loss_k = ridge_regression(y_train, x_train, lambda_)
        test_loss_k = compute_loss(y_test, x_test, w_k)

        y_train_pred = predict_labels(w_k, x_train)
        y_test_pred = predict_labels(w_k, x_test)
        train_f1.append(f1_score(y_train, y_train_pred))
        test_f1.append(f1_score(y_test, y_test_pred))

        train_loss.append(train_loss_k)
        test_loss.append(test_loss_k)
        w.append(w_k)

    loss_tr = sum(train_loss) / k
    loss_te = sum(test_loss) / k
    f1_train = sum(train_f1) / k
    f1_test = sum(test_f1) / k
    ww = np.array(w)
    weight = np.mean(ww, axis=0)

    return loss_tr, loss_te, f1_train, f1_test, weight


def demo(y, x, seed, k, deg_l, lbd_l):
    """a demo for visualization of loss"""
    k_fold = build_k_fold(y, k, seed)
    train_loss = []
    test_loss = []
    f1_train = []
    f1_test = []
    for d in deg_l:
        for i, lbd in enumerate(lbd_l):
            x_ext = poly_extension(x, d)
            loss_tr, loss_te, f1_tr, f1_te, _ = cross_validation(y, x_ext, k_fold, lbd)
            train_loss.append(loss_tr)
            test_loss.append(loss_te)
            f1_train.append(f1_tr)
            f1_test.append(f1_te)

            print(
                "reg_logistic  extension degree %d, Lambda value %f, train_loss = %f, test_loss = %f, train_f1 = %f, test_f1 = %f"
                % (d, lbd, loss_tr, loss_te, f1_tr, f1_te)
            )

            if i == len(lbd_l) - 1:
                plt.clf()
                fig, axs = plt.subplots(2, 1, figsize=(10, 10))

                axs[0].semilogx(
                    lbd_l[0 : i + 1],
                    train_loss,
                    marker=".",
                    color="b",
                    label="train error",
                )
                axs[0].semilogx(
                    lbd_l[0 : i + 1],
                    test_loss,
                    marker=".",
                    color="r",
                    label="test error",
                )
                axs[0].set_xlabel("lambda")
                axs[0].set_ylabel("loss")
                axs[0].set_title("cross validation loss")
                axs[0].legend(loc=2)
                axs[0].grid(True)

                axs[1].semilogx(
                    lbd_l[0 : i + 1], f1_train, marker=".", color="b", label="train f1"
                )
                axs[1].semilogx(
                    lbd_l[0 : i + 1], f1_test, marker=".", color="r", label="test f1"
                )
                axs[1].set_xlabel("lambda")
                axs[1].set_ylabel("F1 score")
                axs[1].set_title("cross validation F1 score")
                axs[1].legend(loc=2)
                axs[1].grid(True)

                plt.tight_layout()
                plt.show()
