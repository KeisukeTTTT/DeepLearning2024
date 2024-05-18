import inspect
import os

import numpy as np
import pandas as pd
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

nn_except = ["Module", "Parameter", "Sequential", "modules", "functional"]
for m in inspect.getmembers(nn):
    if not m[0] in nn_except and m[0][0:2] != "__":
        delattr(nn, m[0])

seed = 1234
torch.manual_seed(seed)
np.random.seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 学習データ
data_dir = "./data/"
x_train = np.load(os.path.join(data_dir, "x_train.npy"))
y_train = np.load(os.path.join(data_dir, "y_train.npy"))

# テストデータ
x_test = np.load(os.path.join(data_dir, "x_test.npy"))


class train_dataset(torch.utils.data.Dataset):
    def __init__(self, x_train, y_train):
        self.x_train = x_train.reshape(-1, 784).astype("float32") / 255
        self.y_train = y_train

    def __len__(self):
        return self.x_train.shape[0]

    def __getitem__(self, idx):
        return torch.tensor(self.x_train[idx], dtype=torch.float), torch.tensor(self.y_train[idx], dtype=torch.long)


class test_dataset(torch.utils.data.Dataset):
    def __init__(self, x_test):
        self.x_test = x_test.reshape(-1, 784).astype("float32") / 255

    def __len__(self):
        return self.x_test.shape[0]

    def __getitem__(self, idx):
        return torch.tensor(self.x_test[idx], dtype=torch.float)


train_val_data = train_dataset(x_train, y_train)
test_data = test_dataset(x_test)

# 多層パーセプトロンの実装
batch_size = 32

val_size = 10000
train_size = len(train_val_data) - val_size

train_data, val_data = torch.utils.data.random_split(train_val_data, [train_size, val_size])

dataloader_train = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

dataloader_valid = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True)

dataloader_test = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)


def relu(x):
    return torch.relu(x)


def softmax(x):
    return torch.softmax(x, dim=1)


class Dense(nn.Module):  # nn.Moduleを継承する
    def __init__(self, in_dim, out_dim, function=lambda x: x):
        super().__init__()
        # He Initialization
        # in_dim: 入力の次元数、out_dim: 出力の次元数
        self.W = nn.Parameter(
            torch.tensor(np.random.uniform(low=-np.sqrt(6 / in_dim), high=np.sqrt(6 / in_dim), size=(in_dim, out_dim)).astype("float32"))
        )
        self.b = nn.Parameter(torch.tensor(np.zeros([out_dim]).astype("float32")))
        self.function = function

    def forward(self, x):  # forwardをoverride
        return self.function(torch.matmul(x, self.W) + self.b)


class MLP(nn.Module):  # nn.Moduleを継承する
    def __init__(self, in_dim, hid_dim, out_dim):
        super(MLP, self).__init__()
        self.layer1 = Dense(in_dim, hid_dim, relu)
        self.layer2 = Dense(hid_dim, out_dim, softmax)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


in_dim = 784
hid_dim = 200
out_dim = 10
lr = 0.001
n_epochs = 10

mlp = MLP(in_dim, hid_dim, out_dim).to(device)

optimizer = optim.Adam(mlp.parameters(), lr=lr)
# criterion = nn.CrossEntropyLoss()

for epoch in range(n_epochs):
    losses_train = []
    losses_valid = []
    train_num = 0
    train_true_num = 0
    valid_num = 0
    valid_true_num = 0

    mlp.train()  # 訓練時には勾配を計算するtrainモードにする
    for x, t in dataloader_train:
        x, t = x.to(device), t.to(device)
        optimizer.zero_grad()
        y_pred = mlp.forward(x)
        # loss = criterion(y_pred, t)
        loss = -torch.log(y_pred[range(len(t)), t]).mean()
        loss.backward()
        optimizer.step()
        losses_train.append(loss.item())

        acc = (y_pred.argmax(dim=1) == t).sum().item()
        train_num += t.size(0)
        train_true_num += acc

    mlp.eval()  # 評価時には勾配を計算しないevalモードにする
    with torch.no_grad():
        for x, t in dataloader_valid:
            x, t = x.to(device), t.to(device)
            y_pred = mlp.forward(x)
            # loss = criterion(y_pred, t)
            loss = -torch.log(y_pred[range(len(t)), t]).mean()
            losses_valid.append(loss.item())

            acc = (y_pred.argmax(dim=1) == t).sum().item()
            valid_num += t.size(0)
            valid_true_num += acc

    print(
        "EPOCH: {}, Train [Loss: {:.3f}, Accuracy: {:.3f}], Valid [Loss: {:.3f}, Accuracy: {:.3f}]".format(
            epoch, np.mean(losses_train), train_true_num / train_num, np.mean(losses_valid), valid_true_num / valid_num
        )
    )

mlp.eval()

t_pred = []
with torch.no_grad():
    for x in dataloader_test:
        x = x.to(device)
        y = mlp.forward(x)
        pred = y.argmax(1).tolist()
        t_pred.extend(pred)

submission = pd.Series(t_pred, name="label")
submission.to_csv("submission_pred.csv", header=True, index_label="id")
