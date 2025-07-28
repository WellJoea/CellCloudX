import torch as th
import numpy as np
from typing import Optional, Union
import scipy.sparse as ssp

def spsparse_to_thsparse(X):
    XX = X.tocoo()
    values = XX.data
    indices = np.vstack((XX.row, XX.col))
    i = th.LongTensor(indices)
    v = th.tensor(values, dtype=th.float64)
    shape = th.Size(XX.shape)
    return th.sparse_coo_tensor(i, v, shape)

def thsparse_to_spsparse(X):
    XX = X.to_sparse_coo().coalesce()
    values = XX.values().detach().cpu().numpy()
    indices = XX.indices().detach().cpu().numpy()
    shape = XX.shape
    return ssp.csr_array((values, indices), shape=shape)

def to_tensor( X, dtype=None, device=None, todense=False, clone=True):
    if th.is_tensor(X):
        X = X.clone()
    elif ssp.issparse(X):
        X = spsparse_to_thsparse(X)
        if todense:
            X = X.to_dense()
    else:
        try:
            X = th.tensor(X, dtype=dtype)
        except:
            raise ValueError(f'{type(X)} cannot be converted to tensor')
    if clone:
        return X.clone().to(dtype).to(device)
    else:
        return X.to(dtype).to(device)

def normalize_total_torch(
    X,
    target_sum: Optional[float] = 1e4,
    exclude_highly_expressed: bool = False,
    max_fraction: float = 0.05,
    dtype=None, device=None,
) :

    X_tensor = to_tensor(X, dtype=dtype, device=device, todense=False)

    counts_per_cell = X_tensor.sum(dim=1)

    gene_mask = None
    if exclude_highly_expressed:
        fraction = X_tensor / counts_per_cell.view(-1, 1)
        gene_mask = (fraction > max_fraction).any(dim=0)
        counts_per_cell = X_tensor[:, ~gene_mask].sum(dim=1)
    
    non_zero_counts = counts_per_cell[counts_per_cell > 0]
    target = target_sum if target_sum else th.median(non_zero_counts).item()

    scaling_factors = counts_per_cell / target
    scaling_factors[scaling_factors == 0] = 1  
    X_tensor = X_tensor / scaling_factors.view(-1, 1)
    return X_tensor
