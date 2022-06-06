# 1) design model (input, output size, forward pass)
# 2) construct loss and optimizer
# 3) Training Loop
##   - forward pass: computer prediction
##   - backward Pass: gradients
##   - update weights
##   - Set Grad to Zero

from pickletools import optimize
import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 0) prepare data
dataset = datasets.load_breast_cancer()
X, y = dataset.data, dataset.target

n_samples, n_features = X.shape
# print(n_samples, n_features)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# scale and transform data (Standard Scalaer Required when using Logistic regression)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

y_train = y_train.view(y_train.shape[0],1)
y_test = y_test.view(y_test.shape[0],1)

# 1) model
class LogisticRegression(nn.Module):
    def __init__(self, n_input_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        y_predited = torch.sigmoid(self.linear(x))
        return y_predited

model = LogisticRegression(n_features)
# 2) loss and optimizer
learning_rate = 0.02
criterian = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)
epochs = 1000

# 3) training loop!!
for epoch in range(epochs):
    # Prediction
    predicted = model(X_train)

    # loss
    loss = criterian(predicted, y_train)
    loss.backward()
    # optimize
    optimizer.step()

    # grads to Zero
    optimizer.zero_grad()

    if (epoch+1) % 100 == 0:
        print(f"epoch: {epoch-1} loss: {loss.item():.4f}")

with torch.no_grad():
    predicted = model(X_test)
    predicted = predicted.round()
    acc = predicted.eq(y_test).sum() / float(y_test.shape[0])
    print(f'accuracy = {acc:.4f}')


