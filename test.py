import torch

t = torch.rand(3,5)
a = t.sort(-1,descending=True)
print(t)
print()
print(a)