import re
import torch

x = torch.tensor(1.0)
y = torch.tensor(2.0)

w = torch.tensor(1.0, requires_grad=True)

# Forward Pass
y_hat = w * x
loss = (y_hat - y)**2

print(loss)

# Backword Pass
loss.backward()

print(w.grad)

# Update weights
# Next forward and Backword Prop