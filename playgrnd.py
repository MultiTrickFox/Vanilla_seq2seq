import torch

a = torch.ones([1, 3], requires_grad=True)
b = torch.Tensor([1, 3, 4])

result = torch.matmul(a, b)
result = result.sum()

result.backward()

print(a.grad)
print(b.grad)
