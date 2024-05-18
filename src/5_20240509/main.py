import os
import random

import numpy as np
import pandas as pd
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision import transforms
from tqdm import tqdm_notebook as tqdm

# 学習データ
data_dir = "./data"
x_train = np.load(os.path.join(data_dir, "x_train.npy"))
y_train = np.load(os.path.join(data_dir, "t_train.npy"))

# テストデータ
x_test = np.load(os.path.join(data_dir, "x_test.npy"))


class train_dataset(torch.utils.data.Dataset):
    def __init__(self, x_train, y_train):
        data = x_train.astype("float32")
        self.x_train = []
        for i in range(data.shape[0]):
            self.x_train.append(Image.fromarray(np.uint8(data[i])))
        self.y_train = y_train
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.x_train)

    def __getitem__(self, idx):
        return self.transform(self.x_train[idx]), torch.tensor(y_train[idx], dtype=torch.long)


class test_dataset(torch.utils.data.Dataset):
    def __init__(self, x_test):
        data = x_test.astype("float32")
        self.x_test = []
        for i in range(data.shape[0]):
            self.x_test.append(Image.fromarray(np.uint8(data[i])))
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.x_test)

    def __getitem__(self, idx):
        return self.transform(self.x_test[idx])


trainval_data = train_dataset(x_train, y_train)
test_data = test_dataset(x_test)


# 畳み込みニューラルネットワークの実装
def fix_seed(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


fix_seed(seed=42)


class gcn:
    # gcn: Global Contrast Normalization
    def __init__(self):
        pass

    def __call__(self, x):
        mean = torch.mean(x)
        std = torch.std(x)
        return (x - mean) / (std + 10 ** (-6))  # 0除算を防ぐ


class ZCAWhitening:
    def __init__(self, epsilon=1e-4, device="cuda" if torch.cuda.is_available() else "cpu"):  # 計算が重いのでGPUを用いる
        self.epsilon = epsilon
        self.device = device

    def fit(self, images):  # 変換行列と平均をデータから計算
        x = images[0][0].reshape(1, -1)
        self.mean = torch.zeros([1, x.size()[1]]).to(self.device)
        con_matrix = torch.zeros([x.size()[1], x.size()[1]]).to(self.device)
        for i in range(len(images)):  # 各データについての平均を取る
            x = images[i][0].reshape(1, -1).to(self.device)
            self.mean += x / len(images)
            con_matrix += torch.mm(x.t(), x) / len(images)
            if i % 10000 == 0:
                print("{0}/{1}".format(i, len(images)))
        self.E, self.V = torch.linalg.eigh(con_matrix)  # 固有値分解
        self.E = torch.max(self.E, torch.zeros_like(self.E))  # 誤差の影響で負になるのを防ぐ
        self.ZCA_matrix = torch.mm(torch.mm(self.V, torch.diag((self.E.squeeze() + self.epsilon) ** (-0.5))), self.V.t())
        print("completed!")

    def __call__(self, x):
        size = x.size()
        x = x.reshape(1, -1).to(self.device)
        x -= self.mean
        x = torch.mm(x, self.ZCA_matrix.t())
        x = x.reshape(tuple(size))
        x = x.to("cpu")
        return x


# (datasetのクラスを自作したので，このあたりの処理が少し変わっています)

zca = ZCAWhitening()
zca.fit(trainval_data)

val_size = 3000
train_data, val_data = torch.utils.data.random_split(trainval_data, [len(trainval_data) - val_size, val_size])  # 訓練データと検証データに分割


# 前処理を定義
transform_train = transforms.Compose([transforms.ToTensor(), gcn(), zca])
transform = transforms.Compose([transforms.ToTensor(), gcn(), zca])

# データセットに前処理を設定
trainval_data.transform = transform_train
test_data.transform = transform

batch_size = 64

dataloader_train = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

dataloader_valid = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True)

dataloader_test = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

rng = np.random.RandomState(1234)
random_state = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


conv_net = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(4096, 512),
    nn.ReLU(),
    nn.Linear(512, 10),
)


def init_weights(m):  # Heの初期化
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        torch.nn.init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0.0)


conv_net.apply(init_weights)


n_epochs = 5
lr = 0.01
device = "cuda" if torch.cuda.is_available() else "cpu"

conv_net.to(device)
optimizer = optim.Adam(conv_net.parameters(), lr=lr)
loss_function = nn.CrossEntropyLoss()

for epoch in range(n_epochs):
    losses_train = []
    losses_valid = []

    conv_net.train()
    n_train = 0
    acc_train = 0
    for x, t in dataloader_train:
        x, t = x.to(device), t.to(device)
        y = conv_net(x)
        loss = loss_function(y, t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = y.argmax(1)
        acc_train += (pred == t).float().sum().item()
        n_train += len(x)
        losses_train.append(loss.tolist())

    conv_net.eval()
    n_val = 0
    acc_val = 0
    for x, t in dataloader_valid:
        x, t = x.to(device), t.to(device)
        y = conv_net(x)
        loss = loss_function(y, t)
        pred = y.argmax(1)
        acc_val += (pred == t).float().sum().item()
        n_val += len(x)
        losses_valid.append(loss.tolist())

    print(
        "EPOCH: {}, Train [Loss: {:.3f}, Accuracy: {:.3f}], Valid [Loss: {:.3f}, Accuracy: {:.3f}]".format(
            epoch, np.mean(losses_train), acc_train / n_train, np.mean(losses_valid), acc_val / n_val
        )
    )

conv_net.eval()

t_pred = []
for x in dataloader_test:

    x = x.to(device)

    # 順伝播
    y = conv_net.forward(x)

    # モデルの出力を予測値のスカラーに変換
    pred = y.argmax(1).tolist()

    t_pred.extend(pred)

submission = pd.Series(t_pred, name="label")
submission.to_csv("submission_pred.csv", header=True, index_label="id")
