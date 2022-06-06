# 1) design model (input, output size, forward pass)
# 2) construct loss and optimizer
# 3) Training Loop
##   - forward pass: computer prediction
##   - backword Pass: gradients
##   - update weights

import torch

X = torch.tensor([1,2,3,4], dtype=torch.float32)
y = torch.tensor([2,4,6,8], dtype=torch.float32)

w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

# model Forward
def forward(x):
    return w * x

# loss = MSE
def loss(y, y_pred):
    return ((y_pred-y)**2).mean()


print(f'Predicted before training: f(5) = {forward(5):.3f}')

# Training 
learing_rate = 0.01
n_iters = 1000

for epochs in range(n_iters):
    # Forward Pass
    y_pred = forward(X)

    # loss
    l = loss(y, y_pred)

    # gradient
    l.backward() # dl/dw

    # Update weight
    with torch.no_grad():
        w -= learing_rate * w.grad

    # Zero Gradient
    w.grad.zero_()

    if epochs % 10 == 0:
        print(f'epoch {epochs+1}: w = {w:.3f}, loss = {l:.8f}')

print(f'Prediction after training: f(5) = {forward(5):.3f}')