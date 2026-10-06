import torch
def green_list(secret, V):
    g = torch.Generator().manual_seed(int(secret)); return torch.rand(V, generator=g) < 0.5
