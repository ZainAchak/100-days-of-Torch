import numpy as np

X = np.array([1,2,3,4], dtype=np.float32)
y = np.array([2,4,6,8], dtype=np.float32)

w = 0.0

# model Forward
def forward(x):
    return w * x

# loss = MSE
def loss(y, y_pred):
    return ((y_pred-y)**2).mean()

# Gradient descent
def gradient(x, y, y_predicted):
    return np.dot(2*x, y_predicted-y).mean()


print(f'Predicted before training: f(5) = {forward(5):.3f}')

# Training 
learing_rate = 0.01
n_iters = 10

for epochs in range(n_iters):
    # Forward Pass
    y_pred = forward(X)

    # loss
    l = loss(y, y_pred)

    # gradient
    dw = gradient(X, y, y_pred)

    # Update weight
    w -= learing_rate * dw

    if epochs % 1 == 0:
        print(f'epoch {epochs+1}: w = {w:.3f}, loss = {l:.8f}')

print(f'Prediction after training: f(5) = {forward(5):.3f}')