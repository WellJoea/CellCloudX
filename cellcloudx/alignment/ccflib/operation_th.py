import numpy as np
import torch as th
import os
import random
import scipy as sci
from scipy.sparse import issparse
from .neighbors_ensemble import Neighbors
from .operation_expectation import (centerlize, normalize, scaling, 
                                    spsparse_to_thsparse, thsparse_to_spsparse,
                                    sigma_square_cos, sigma_square)
from ...utilis._clean_cache import clean_cache

class thopt():
    def __init__(self, device=None, device_pre=None, floatx=None, 
                 floatxx=None,
                 eps=1e-8, seed=200504, ):
        super().__init__()
        self.xp = th
        self.device = (self.xp.device('cuda' if self.xp.cuda.is_available() else 'cpu') 
                       if device is None else device)
        self.device_pre = (self.xp.device('cuda' if self.xp.cuda.is_available() else 'cpu') 
                            if device_pre is None else device_pre)
        self.device_index = self.xp.device(self.device).index
        self.floatx  = eval(f'self.xp.{ floatx  or "float32"}')
        self.floatxx = eval(f'self.xp.{ floatxx or "float64"}')
        self.seed_torch(seed)
        self.eps = self.to_tensor( min(self.xp.finfo(self.floatx).eps, eps),
                                  dtype=self.floatx, device=self.device)
        self.spsparse_to_thsparse = spsparse_to_thsparse
        self.thsparse_to_spsparse = thsparse_to_spsparse
        self.centerlize = centerlize
        self.normalize = normalize
        self.scaling = scaling
        self.sigma_square_cos = sigma_square_cos
        self.sigma_square = sigma_square
    
        self.clean_cache = clean_cache
        self.dvinfo = [self.get_memory(idv) for idv in [self.device, self.device_pre]]

    def get_memory(self, device):
        if hasattr(device, 'type'):
            device = device.type
        if 'cuda' in str(device):
            tm = self.xp.cuda.get_device_properties(device).total_memory/(1024**3)
            #am = self.xp.cuda.memory_allocated(device)
            rm = self.xp.cuda.memory_reserved(device)/(1024**3)
            am = tm - rm
            return f'{device} T={tm:.2f};A={am:.2f}GB'
        else:
            # gm = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")/(1024**3)
            import psutil
            tm = psutil.virtual_memory().total/(1024**3)
            am = psutil.virtual_memory().available/(1024**3)
            return f'cpu T={tm:.2f};A={am:.2f}GB'

    def is_null(self, X):
        if X is None:
            return True
        elif len(X) == 0:
            return True
        elif isinstance(X, (list, tuple)):
            return all([ self.is_null(i) for i in X])
        else:
            return False

    def is_sparse(self, X):
        if issparse(X):
            return True, 'scipy'
        elif th.is_tensor(X):
            return X.is_sparse, 'torch'
        else:
            return False, 'numpy'

    def tensor2numpy(self, x):
        if x is None:
            return x
        issp, spty = self.is_sparse(x)
        if (spty=='torch'):
            if issp:
                value = self.thsparse_to_spsparse(x)
            else:
                value = x.detach().cpu().numpy()
            value = value.astype(np.float64)
        elif th.is_tensor(x):
            value = x.detach().cpu().numpy().astype(np.float64)

        elif isinstance(x, (list, tuple)):
            value = [ self.tensor2numpy(v) for v in x ]
        else:
            value = x
        return value

    def tensordetach(self, x):
        if x is None:
            return x
        issp, spty = self.is_sparse(x)
        if (spty=='torch'):
            value = x.detach().cpu()
        elif th.is_tensor(x):
            value = x.detach().cpu()
        elif isinstance(x, (list, tuple)):
            value = [ self.tensordetach(v) for v in x ]
        else:
            value = th.asarray(x, device='cpu')
        return value

    def speye(self, n, dtype=None, device=None, value=1.0):
        indices = self.xp.arange(n).repeat(2, 1)
        values = self.xp.ones(n)*value
        sparse_eye = self.xp.sparse_coo_tensor(indices, values, (n, n))
        return sparse_eye.to(device, dtype=dtype)

    def scalar2vetor(self, x, L, default = None, force=False):
        if force:
            xs = [ x for i in range(L) ]
        else:
            if ((type(x) in [str, float, int, bool]) 
                or isinstance(x, (str, bytes))
                or np.isscalar(x)
                or (x is None)):
                xs = [ x for i in range(L) ]
            else:
                assert len(x) == L, f'len(x)={len(x)} != L={L}'
                xs = x
        if default is not None:
            xs = [ default if x is None else x for x in xs ]
        return xs

    def scalar2list(self, x, L):
        if ((type(x) in [str, float, int, bool]) 
            or isinstance(x, (str, bytes))
            or np.isscalar(x)
            or (x is None)):
            xs = [ [ x for j in range(i) ] for i in L ]
        else:
            for ix, il in zip(x, L):
                iv = ix
                if ((type(x) in [str, float, int, bool]) 
                    or isinstance(x, (str, bytes))
                    or np.isscalar(x)
                    or (x is None)) and il == 1:
                    iv = [ix]
                else:
                    iv = ix
                assert len(iv) == il, f'len(x)={len(iv)} != L={il}'
            xs = x
        return xs

    def scalar2matrix(self, x, L, device=None, dtype=None):
        '''x:scalar, L:list of int'''
        assert not x is None
        x = self.xp.asarray(x)
        if x.ndim ==1 and x.shape[0] == L[0]:
            x = x.expand(*L).T.clone()
        else:
            x = x.expand(*L).clone()
        return x.to(device, dtype=dtype)

    def to_tensor(self, X, dtype=None, device=None, todense=False, requires_grad =False):
        if th.is_tensor(X):
            X = X.clone()
        elif issparse(X):
            X = self.spsparse_to_thsparse(X)
            if todense:
                X = X.to_dense()
        else:
            try:
                X = th.asarray(X, dtype=dtype)
            except:
                raise ValueError(f'{type(X)} cannot be converted to tensor')
        X = X.clone().to(device=device, dtype=dtype)
        X.requires_grad_(requires_grad)
        return X

    def flatten_list(self, lst):
        for item in lst:
            if isinstance(item, list):
                yield from self.flatten_list(item) 
            elif item is None:
                pass
            else:
                yield item

    def cos_distance(self, X, Y, device=None, use_pair = False,):
        xp = self.xp
        device = X.device if device is None else device

        X = X.clone().to(device)
        Y = Y.clone().to(device)
        for x in [X, Y]:
            l2x = xp.linalg.norm(x, ord=None, dim=1, keepdim=True)
            l2x[l2x == 0] = 1
            x /= l2x

        if use_pair:
            assert len(X) == len(Y), 'len(X) != len(Y)'
            D = xp.sum((X-Y)**2, axis=-1, keepdims=True)
        else:
            D = th.cdist(X, Y, p=2) # TODO KeOps
            D.pow_(2)
        return D

    def js_distance(self, X, Y, normalize=True, eps=1e-10, device=None):
        pass
    
    def euclidean_distance(self, X, Y,  use_pair = False, device=None):
        xp = self.xp
        device = X.device if device is None else device

        X = X.clone().to(device)
        Y = Y.clone().to(device)

        if use_pair:
            assert len(X) == len(Y), 'len(X) != len(Y)'
            D = xp.sum((X-Y)**2, axis=-1, keepdims=True)
        else:
            D = th.cdist(X, Y, p=2) # TODO KeOps
            D.pow_(2)
        return D

    def kl_distance(self, X, Y, normalize=True, eps=1e-10, use_pair = False,
                    negative_handling ='softmax', device=None):
        xp = self.xp
        device = X.device if device is None else device

        X = X.clone().to(device)
        Y = Y.clone().to(device)
        if xp.any(X < 0) or xp.any(Y < 0):
            if negative_handling =='softmax':
                if xp.__name__ == 'torch':
                    X = xp.exp(X - xp.max(X, axis=1, keepdims=True)[0])
                    Y = xp.exp(Y - xp.max(Y, axis=1, keepdims=True)[0])
                else:
                    X = xp.exp(X - xp.max(X, axis=1, keepdims=True))
                    Y = xp.exp(Y - xp.max(Y, axis=1, keepdims=True))

            elif negative_handling == 'shift':
                if xp.__name__ == 'torch':
                    X = X - xp.min(X, axis=1, keepdims=True)[0]
                    Y = Y - xp.min(Y, axis=1, keepdims=True)[0]
                else:
                    X = X - xp.min(X, axis=1, keepdims=True)
                    Y = Y - xp.min(Y, axis=1, keepdims=True)
            else:
                raise ValueError('X and Y must be non-negative')

        if normalize:
            X_norm = xp.sum(X, axis=1, keepdims=True)
            Y_norm = xp.sum(Y, axis=1, keepdims=True)
            X = xp.where(X_norm > eps, X / X_norm, X)
            Y = xp.where(Y_norm > eps, Y / Y_norm, Y)

        X = xp.clip(X, eps, None)
        Y = xp.clip(Y, eps, None)

        log_X = xp.log(X)
        log_Y = xp.log(Y)

        if use_pair:
            assert len(X) == len(Y), 'len(X) != len(Y)'
            D = xp.sum(X * (log_X - log_Y), axis=-1, keepdims=True) 
        else:
            # D = xp.sum(X[:, None] * (log_X[:, None] - log_Y[None, :]), axis=2) # TODO KeOps
            X_log_X = xp.sum(X * log_X, axis=1, keepdims=True)
            D = X_log_X - X @ log_Y.T
        return D

    def dist2prob(self, dist2, sigma2, D, kernel='gmm', dfs=1, temp=1):
        if kernel == 'gmm':
            D = D/sigma2
            D = D**dfs
            D = D/temp
        return D

    def kernel_xmm(self, X, Y, sigma2=None,
                 temp=1, dfs=1, kernel='gmm'):
        (N, D) = X.shape
        M = Y.shape[0]
        dist2 = self.xp.cdist(Y, X, p=2)
        dist2.pow_(2)
        if sigma2 is None:
            sigma2 = self.xp.sum(dist2) / (D*N*M)
        R = self.dist2prob(dist2, sigma2, D, 
                            kernel=kernel, dfs=dfs, temp=temp)
        return dist2, R, sigma2

    def cdist_k(self, X, Y, sigma2=None, method='cunn',
                    metric='euclidean', 
                    knn=50, n_jobs=-1, **kargs):
        (N, D) = X.shape
        M = Y.shape[0]
        snn = Neighbors(method=method, metric=metric, 
                        device_index=self.device_index,
                        n_jobs=n_jobs)
        snn.fit(Y.detach().cpu().numpy(), **kargs)
        ckdout = snn.transform(X.detach().cpu().numpy(), knn=knn)
        nnidx = ckdout[1]

        src = self.xp.LongTensor(nnidx.flatten('C')).to(X.device)
        # dst = self.xp.LongTensor.repeat_interleave(self.xp.arange(N), knn, device=X.device)
        dst = self.xp.LongTensor(np.repeat(np.arange(N), knn)).to(X.device)
        dist2 = self.xp.tensor((ckdout[0].flatten('C'))**2, dtype=X.dtype, device=X.device)

        if sigma2 is None:
            sigma2 = self.xp.sum(dist2) / (D*N*knn)
        dist2 = self.xp.sparse_coo_tensor( self.xp.vstack([src, dst]), dist2, 
                                        size=(M, N), 
                                        dtype=self.floatx)
        return dist2, sigma2
    
    def kernel_xmm_k(self, X, Y, sigma2=None, method='cunn',
                    metric='euclidean', 
                    knn=50, n_jobs=-1,
                    temp=1, dfs=1, kernel='gmm', **kargs):
        (N, D) = X.shape
        M = Y.shape[0]
        snn = Neighbors(method=method, metric=metric, 
                        device_index=self.device_index,
                        n_jobs=n_jobs)
        snn.fit(Y.detach().cpu().numpy(), **kargs)
        ckdout = snn.transform(X.detach().cpu().numpy(), knn=knn)
        nnidx = ckdout[1]

        src = self.xp.LongTensor(nnidx.flatten('C')).to(X.device)
        # dst = self.xp.LongTensor.repeat_interleave(self.xp.arange(N), knn, device=X.device)
        dst = self.xp.LongTensor(np.repeat(np.arange(N), knn)).to(X.device)
        dist2 = self.xp.tensor((ckdout[0].flatten('C'))**2, dtype=X.dtype, device=X.device)

        if sigma2 is None:
            sigma2 = self.xp.sum(dist2) / (D*N*knn)
        R = self.dist2prob(dist2, sigma2, D, 
                    kernel=kernel, dfs=dfs, temp=temp)

        P = self.xp.sparse_coo_tensor( self.xp.vstack([src, dst]), dist2, 
                                        size=(M, N), 
                                        dtype=self.floatx)
        return P,R, sigma2

    def kernel_xmm_p(self, X, Y, pairs, sigma2=None, temp=1,
                        dfs=1, kernel='gmm'):
        assert X.shape[1] == Y.shape[1]
        (N, D) = X.shape
        M = Y.shape[0]

        dist2 = self.xp.square(X[pairs[0]] - Y[pairs[1]])
        dist2.sum_(-1)
        sigma2 = (self.xp.mean(dist2)/D).item() if sigma2 is None else sigma2

        R = self.dist2prob(dist2, sigma2, D, 
                    kernel=kernel, dfs=dfs, temp=temp)
        P = self.xp.sparse_coo_tensor( (pairs[0], pairs[1]), dist2, shape=(N, M), 
                                        dtype=self.floatx)
        return P,R, sigma2

    def dist2prob(self, dist2, sigma2, D, kernel='gmm', dfs=2, temp=1):

        if kernel == 'gmm':
            dist2.div_(-2 * sigma2 * temp)
            dist2.exp_()
            R = (2 * self.xp.pi * sigma2) ** (-0.5 * D)
            # R = self.xp.asarray(R, dtype=dist2.dtype, device=dist2.device)
            return R

        elif kernel == 'smm':
            #th.exp(torch.lgamma((dfs + D) / 2.0))
            R  = sci.special.gamma((dfs + D) / 2.0) 
            R /= sci.special.gamma(dfs/2.0) * (sigma2**0.5)
            R *= np.power(np.pi * dfs, -D/ 2.0)
            R = self.xp.asarray(R, dtype=dist2.dtype, device=dist2.device)

            dist2.div_(sigma2*dfs)
            dist2.add_(1.0)
            dist2.pow_( -(dfs + D) / 2.0 )
            return R

        elif kernel is None:
            R = self.xp.asarray(1, dtype=dist2.dtype, device=dist2.device)
            return R
    
    def seed_torch(self, seed=200504):
        if not seed is None:
            random.seed(seed)
            os.environ['PYTHONHASHSEED'] = str(seed)
            np.random.seed(seed)
            th.manual_seed(seed)
            th.cuda.manual_seed(seed)
            th.cuda.manual_seed_all(seed)
            th.mps.manual_seed(seed)
            th.backends.cudnn.deterministic = True
            th.backends.cudnn.benchmark = False
