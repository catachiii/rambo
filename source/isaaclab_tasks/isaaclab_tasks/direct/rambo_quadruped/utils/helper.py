import torch


def to_torch(x, dtype=torch.float, device='cuda:0', requires_grad=False):
    if isinstance(x, torch.Tensor):
        return x.clone().detach().to(device).requires_grad_(requires_grad)
    else:
        return torch.tensor(x, dtype=dtype, device=device, requires_grad=requires_grad)
