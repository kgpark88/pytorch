import torch

a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)

x = 2*a + 3*b
y = 2*a*a*a + 3*b*b
z = 3*x + 2*y
z.backward()

print(a.grad)
