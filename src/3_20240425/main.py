import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def preprocess():
    x_train = np.load("data/x_train.npy")
    y_train = np.load("data/y_train.npy")

    x_test = np.load("data/x_test.npy")

    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train, x_test = x_train.reshape(x_train.shape[0], -1), x_test.reshape(x_test.shape[0], -1)
    y_train = np.eye(10)[y_train.astype("int32").flatten()]

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=10000, random_state=42)
    return x_train, y_train, x_val, y_val, x_test


def np_log(x):
    return np.log(np.clip(x, 1e-10, 1e10))


def create_batch(data, batch_size):
    """
    :param data: np.ndarray, input data
    :param batch_size: int, batch size
    """
    num_batches, mod = divmod(data.shape[0], batch_size)
    batched_data = np.split(data[: num_batches * batch_size], num_batches)
    if mod:
        batched_data.append(data[num_batches * batch_size :])

    return batched_data


def relu(x):
    return np.maximum(0, x)


def deriv_relu(x):
    return (x > 0).astype(x.dtype)


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=1)[:, np.newaxis]


def deriv_softmax(x): ...


def cross_entropy_loss(t, y):
    return -np.sum(t * np_log(y)) / t.shape[0]


class Dense:
    def __init__(self, units, activation=relu):
        self.w = np.random.randn(units[0], units[1])
        self.b = np.random.randn(units[1])
        self.units = units
        self.activation = activation

    def forward(self, x):
        return self.activation(x @ self.w + self.b)

    def backward(self, grad):
        grad = self.activation(grad)


class Model:
    def __init__(self):
        pass


def train_model(mlp, x_train, y_train, x_val, y_val, epochs, batch_size, lr):
    for epoch in range(epochs):
        losses_train = []
        losses_val = []
        train_num = 0
        train_true_num = 0
        val_num = 0
        val_true_num = 0

        x_train, y_train = shuffle(x_train, y_train)
        x_train_batches, y_train_batches = create_batch(x_train, batch_size), create_batch(y_train, batch_size)

        x_val, y_val = shuffle(x_val, y_val)
        x_val_batches, y_val_batches = create_batch(x_val, batch_size), create_batch(y_val, batch_size)

        # train
        for x, y in zip(x_train_batches, y_train_batches):
            # forward
            y_pred = mlp.forward(x)

            # loss
            loss = cross_entropy_loss(y, y_pred)
            losses_train.append(loss)

            # update
            mlp.backward(y)

            # accuracy
            accuracy = accuracy_score(y.argmax(axis=1), y_pred.argmax(axis=1), normalize=False)
            train_num += x.shape[0]
            train_true_num += accuracy

        # val
        for x, y in zip(x_val_batches, y_val_batches):
            # forward
            y_pred = mlp.forward(x)

            # loss
            loss = cross_entropy_loss(y, y_pred)
            losses_val.append(loss)

            # accuracy
            accuracy = accuracy_score(y.argmax(axis=1), y_pred.argmax(axis=1), normalize=False)
            val_num += x.shape[0]
            val_true_num += accuracy

        print(
            f"EPOCH: {epoch + 1}, Train [Loss: {np.mean(losses_train):.3f}, Accuracy: {train_true_num / train_num:.3f}], Val [Loss: {np.mean(losses_val):.3f}, Accuracy: {val_true_num / val_num:.3f}]"
        )


if __name__ == "__main__":
    # hyperparameters
    lr = 0.1
    epochs = 100
    batch_size = 100

    save_pred = True

    mlp = Model()
    x_train, y_train, x_val, y_val, x_test = preprocess()
    train_model(mlp, x_train, y_train, x_val, y_val, epochs, batch_size, lr)

    y_pred = []
    for x in x_test:
        # forward
        x = x[np.newaxis, :]
        y = mlp(x)

        pred = y.argmax(axis=1).tolist()

        y_pred.extend(pred)

    submission = pd.Series(y_pred, name="label")
    if save_pred:
        submission.to_csv("result/submission.csv", header=True, index_label="id")
