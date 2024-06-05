import os
import random
import re
import string
from collections import Counter
from typing import List, Union

import numpy as np
import pandas as pd
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchtext
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchtext import datasets
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import vocab

torchtext.disable_torchtext_deprecation_warning()

seed = 1234
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


# 学習データ
data_dir = "./data"
x_train = np.load(os.path.join(data_dir, "x_train.npy"), allow_pickle=True)
y_train = np.load(os.path.join(data_dir, "t_train.npy"), allow_pickle=True)

# 検証データを取る
x_train, x_valid, t_train, t_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=seed)

# テストデータ
x_test = np.load(os.path.join(data_dir, "x_test.npy"), allow_pickle=True)


def text_transform(text: List[int], max_length=256):
    # <BOS>はすでに1で入っている．<EOS>は2とする．
    text = text[: max_length - 1] + [2]

    return text, len(text)


def collate_batch(batch):
    label_list, text_list, len_seq_list = [], [], []

    for sample in batch:
        if isinstance(sample, tuple):
            label, text = sample

            label_list.append(label)
        else:
            text = sample.copy()

        text, len_seq = text_transform(text)
        text_list.append(torch.tensor(text))
        len_seq_list.append(len_seq)

    # NOTE: 宿題用データセットでは<PAD>は3です．
    return torch.tensor(label_list), pad_sequence(text_list, padding_value=3).T, torch.tensor(len_seq_list)


word_num = np.concatenate(np.concatenate((x_train, x_test))).max() + 1
print(f"単語種数: {word_num}")

batch_size = 128

train_dataloader = DataLoader(
    [(t, x) for t, x in zip(t_train, x_train)],
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_batch,
)
valid_dataloader = DataLoader(
    [(t, x) for t, x in zip(t_valid, x_valid)],
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_batch,
)
test_dataloader = DataLoader(
    x_test,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_batch,
)


def torch_log(x):
    return torch.log(torch.clamp(x, min=1e-10))


class Embedding(nn.Module):
    def __init__(self, emb_dim, vocab_size):
        super().__init__()
        self.embedding_matrix = nn.Parameter(torch.rand((vocab_size, emb_dim), dtype=torch.float))

    def forward(self, x):
        return F.embedding(x, self.embedding_matrix)


class SequenceTaggingNet(nn.Module):
    def __init__(self, word_num, emb_dim, hid_dim):
        super().__init__()
        self.emb = Embedding(emb_dim, word_num)
        # self.rnn = nn.LSTM(emb_dim, hid_dim, batch_first=True)
        self.rnn = nn.RNN(emb_dim, hid_dim, batch_first=True)
        self.linear = nn.Linear(hid_dim, 1)

    def forward(self, x, len_seq_max=0, len_seq=None, init_state=None):
        h = self.emb(x)
        h = self.rnn(h, len_seq_max, init_state)
        if len_seq is not None:
            h = h[len_seq - 1, list(range(len(x))), :]
        else:
            h = h[-1]
        y = self.linear(h)
        return y


emb_dim = 100
hid_dim = 50
n_epochs = 10
device = "mps"

net = SequenceTaggingNet(word_num, emb_dim, hid_dim)
net.to(device)
optimizer = optim.Adam(net.parameters())

for epoch in range(n_epochs):
    losses_train = []
    losses_valid = []

    net.train()
    n_train = 0
    acc_train = 0
    for label, line, len_seq in train_dataloader:
        # label : (batch_size,)
        # line : (batch_size, max_length)
        # len_seq : (batch_size,)
        net.zero_grad()

        t = label.to(device)
        x = line.to(device)
        len_seq = len_seq.to(device)
        import pdb

        pdb.set_trace()

        h = net(x, torch.max(len_seq), len_seq)
        y = torch.sigmoid(h).squeeze()

        loss = -torch.mean(t * torch_log(y) + (1 - t) * torch_log(1 - y))

        loss.backward()
        optimizer.step()

        losses_train.append(loss.tolist())

        n_train += t.size()[0]
        acc_train += torch.sum((y.round() == t).float()).item()

    # Valid
    t_valid = []
    y_pred = []
    net.eval()
    for label, line, len_seq in valid_dataloader:

        t = label.to(device)
        x = line.to(device)
        len_seq = len_seq.to(device)

        h = net(x, torch.max(len_seq), len_seq)
        y = torch.sigmoid(h).squeeze()

        loss = -torch.mean(t * torch_log(y) + (1 - t) * torch_log(1 - y))

        pred = y.round().squeeze()  # 0.5以上の値を持つ要素を正ラベルと予測する

        t_valid.extend(t.tolist())
        y_pred.extend(pred.tolist())

        losses_valid.append(loss.tolist())

    print(
        "EPOCH: {}, Train Loss: {:.3f}, Valid Loss: {:.3f}, Validation F1: {:.3f}".format(
            epoch, np.mean(losses_train), np.mean(losses_valid), f1_score(t_valid, y_pred, average="macro")
        )
    )

net.eval()

y_pred = []
for _, line, len_seq in test_dataloader:

    x = line.to(device)
    len_seq.to(device)

    h = net(x, torch.max(len_seq), len_seq)
    y = torch.sigmoid(h).squeeze()

    pred = y.round().squeeze()  # 0.5以上の値を持つ要素を正ラベルと予測する

    y_pred.extend(pred.tolist())


submission = pd.Series(y_pred, name="label")
submission.to_csv("drive/MyDrive/Colab Notebooks/DLBasics2023_colab/Lecture06/submission_pred.csv", header=True, index_label="id")
