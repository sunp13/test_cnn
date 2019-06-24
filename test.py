import torch
import torch.nn

input = torch.randn(5, 3)
print(input)
print("-"*10)

box = torch.nn.Linear(3, 2)
print(box)
print(box.weight)
print("-"*10)

res = box(input)
print(res)
