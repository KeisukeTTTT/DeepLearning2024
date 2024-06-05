import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def load_fashionmnist(data_dir="./data"):
    x_train = np.load(f"{data_dir}/x_train.npy")
    y_train = np.load(f"{data_dir}/y_train.npy")

    x_test = np.load(f"{data_dir}/x_test.npy")

    x_train = x_train.reshape(-1, 784).astype("float32") / 255
    y_train = np.eye(10)[y_train.astype("int32")]
    x_test = x_test.reshape(-1, 784).astype("float32") / 255

    return x_train, y_train, x_test


def softmax(x):
    x -= x.max(axis=1, keepdims=True)
    return np.exp(x) / np.exp(x).sum(axis=1, keepdims=True)


def train(x, t, eps=1.0):
    global W, b
    batch_size = x.shape[0]
    y_hat = softmax(np.dot(x, W) + b)
    cost = (-t * np.log(y_hat)).sum(axis=1).mean()
    delta = y_hat - t
    dW = np.dot(x.T, delta) / batch_size
    db = delta.mean(axis=0)
    W -= eps * dW
    b -= eps * db
    return cost


def valid(x, t):
    y_hat = softmax(np.dot(x, W) + b)
    cost = (-t * np.log(y_hat)).sum(axis=1).mean()
    return cost, y_hat


if __name__ == "__main__":
    x_train, y_train, x_test = load_fashionmnist()
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1)
    print("train size: ", x_train.shape[0])

    # hyperparameters
    epochs = 100
    batch_size = 100
    learning_rate = 0.1

    # weights
    W = np.random.uniform(low=-0.08, high=0.08, size=(784, 10)).astype("float32")
    b = np.zeros(shape=(10,), dtype="float32")

    iter = x_train.shape[0] // batch_size

    train_losses = []
    valid_losses = []
    accuracies = []
    for epoch in range(epochs):
        idxs = np.random.permutation(x_train.shape[0])
        idxs = np.array_split(idxs, iter)
        for idx in idxs:
            x_batch = x_train[idx]
            y_batch = y_train[idx]
            train_loss = train(x_batch, y_batch, eps=learning_rate)
            train_losses.append(train_loss)
        valid_loss, y_pred = valid(x_valid, y_valid)
        valid_losses.append(valid_loss)
        accuracy = accuracy_score(y_valid.argmax(axis=1), y_pred.argmax(axis=1))
        accuracies.append(accuracy)
        if (epoch + 1) % 10 == 0:
            print(f"epoch {epoch + 1}, train loss: {train_loss:.3f}, accuracy: {accuracy:.3f}")

    y_pred = softmax(np.dot(x_test, W) + b).argmax(axis=1)
    submission = pd.Series(y_pred, name="label")

    save_csv = True
    if save_csv:
        os.makedirs("./result", exist_ok=True)
        submission.to_csv("./result/submission_pred.csv", header=True, index_label="id")

    plot = True
    if plot:
        plt.figure(figsize=(10, 6))

        # Plot train_loss and valid_loss on the left y-axis
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label="train_loss")
        plt.plot(valid_losses, label="valid_loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid()

        # Plot accuracies on the right y-axis
        plt.subplot(1, 2, 2)
        plt.plot(accuracies, label="accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid()

        plt.tight_layout()
        plt.show()
