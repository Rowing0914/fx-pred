import torch
from torch import nn
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from gen_rl.commons.seeds import set_randomSeed
from set_summariser import RNNFamily, DeepSet, BiLSTM, Transformer

pd.options.mode.chained_assignment = None

set_randomSeed(seed=2023)

# df = yf.download("USDJPY=X", start="2021-01-01", end="2023-02-24")
# df.index = pd.to_datetime(df.index)
# df.to_csv("./data.csv")
# print(df.columns)
# print(df.tail())

df = pd.read_csv("./data.csv")
print(df.tail())

seq_len = 7

df_close = df[["Close"]]
cols = list()
for i in range(1, seq_len + 1):
    df_close[f"Close{i}"] = df_close["Close"].shift(i)
    cols.append(f"Close{i}")

df_close = df_close.dropna().reset_index(drop=True)
df_close["if_up"] = ((df_close["Close"] - df_close["Close1"]) > 0.0).astype(int)
print(df_close.tail())

total_size = len(df_close) - 1
train_size = int(total_size * 0.8)
train_ids = np.random.randint(low=0, high=total_size, size=train_size)
test_ids = np.asarray([i for i in range(total_size) if i not in train_ids])

X, Y = df_close[cols].values, df_close["if_up"].values
X_train, X_test = X[train_ids, :], X[test_ids, :]
Y_train, Y_test = Y[train_ids], Y[test_ids]

# test that dimensions are as expected
# model = RNNFamily(num_layers=2, model_type="lstm")
# model = DeepSet()
# model = BiLSTM()
model = Transformer()

device = "cpu"
# criterion = nn.MSELoss()
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
batch_size = 128
num_epochs = 500 * 3
log_freq = 100

model.train()
for epoch in range(num_epochs):
    ids = np.random.randint(low=0, high=X_train.shape[0], size=batch_size)
    x = torch.tensor(X_train[ids].astype(np.float32), device=device)[..., None]
    y = torch.tensor(Y_train[ids].astype(np.float32), device=device)[:, None]
    pred = model(x)
    loss_train = criterion(pred, y)

    optimizer.zero_grad()
    loss_train.backward()
    optimizer.step()

    # display loss and predictions
    if (epoch % log_freq) == 0:
        model.eval()
        with torch.no_grad():
            ids = np.random.randint(low=0, high=X_test.shape[0], size=batch_size)
            x = torch.tensor(X_test[ids].astype(np.float32), device=device)[..., None]
            y = Y_test[ids].astype(np.float32)
            pred = np.where(model(x).detach().cpu().numpy() > 0.5, 1, 0).astype(np.float32)
            loss_test = (y == pred).mean()
            print(f"[Loss] Epoch: {epoch}, Train: {loss_train.item()}, Test: {loss_test}")
        model.train()

with torch.no_grad():
    recent_x_days = 3
    x, y = X[-recent_x_days:], Y[-recent_x_days:]
    x = torch.tensor(x.astype(np.float32), device=device)[..., None]
    pred = np.where(model(x).detach().cpu().numpy() > 0.5, 1, 0).flatten()
    for t in range(1, recent_x_days + 1):
        _i = recent_x_days - t
        print(f"T - {_i + 1}: {x[_i].detach().cpu().numpy().flatten()} | {pred.flatten()[_i]}")

    x_yesterday = np.asarray([df_close["Close"].values[-1]])
    x = np.concatenate([x_yesterday, X[-1, :-1].flatten()])
    x = torch.tensor(x.astype(np.float32), device=device)[:, None, None]
    pred = np.where(model(x).detach().cpu().numpy() > 0.5, 1, 0).flatten()
    print(f"Tomorrow: {x.detach().cpu().numpy().flatten()} | {pred.flatten()[0]}")

""" todo
- sentiment analysis: https://qiita.com/THERE2/items/8b7c94787911fad8daa6
- image based fundamentals analysis: https://qiita.com/ryosao/items/32e30baa7374f78aeaf0
"""
