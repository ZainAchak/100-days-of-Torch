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

X = torch.tensor([[1],[2],[3],[4]], dtype=torch.float32)
y = torch.tensor([[2],[4],[6],[8]], dtype=torch.float32)

X_test = torch.tensor([5], dtype=torch.float32)
n_samples, n_features = X.shape
print(n_samples, n_features)

input_size = n_features
output_size = n_features

# model Forward
model = nn.Linear(input_size,output_size)

learing_rate = 0.01
# loss = MSE
loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learing_rate)


print(f'Predicted before training: f(5) = {model(X_test).item():.3f}')

# Training 
n_iters = 100

for epochs in range(n_iters):
    # Forward Pass
    y_pred = model(X)

    # loss
    l = loss(y, y_pred)

    # gradient
    l.backward() # dl/dw

    # Update weight
    optimizer.step()

    # Zero Gradient
    optimizer.zero_grad()

    if epochs % 10 == 0:
        [w, b] = model.parameters()
        print(f'epoch {epochs+1}: w = {w[0][0].item():.3f}, loss = {l:.8f}')

print(f'Predicted before training: f(5) = {model(X_test).item():.3f}')