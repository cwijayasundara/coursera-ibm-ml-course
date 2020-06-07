import torch

q = torch.tensor(1.0,requires_grad=False)
fq=2*q ** 3 + q
fq.backward()
q.grad

