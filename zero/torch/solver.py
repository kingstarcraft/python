import numpy as np

import torch
import torch_dct


def possion(laplacian, mean=None):
    f = torch_dct.dct_2d(laplacian, 'ortho')
    shape = laplacian.shape[-2:]
    ky, kx = torch.meshgrid(
        torch.arange(shape[0], device=laplacian.device),
        torch.arange(shape[1], device=laplacian.device),
        indexing='ij'
    )
    u = f / (2 * (torch.cos(torch.pi * ky / (shape[0] + 2)) + torch.cos(torch.pi * kx / (shape[-1] + 2)) - 2))
    u[..., 0, 0] = 0
    reconst = torch_dct.idct_2d(u, 'ortho')
    if mean is None:
        return reconst
    else:
        normal = reconst - reconst.mean(dim=[-2, -1])[..., None, None]
        if isinstance(mean, (np.ndarray, list, tuple)):
            mean = torch.tensor(mean, dtype=normal.dtype, device=normal.device)[..., None, None]
        elif isinstance(mean, torch.Tensor):
            mean = mean[..., None, None]
        else:
            raise NotImplemented
        return normal + mean
