import numpy as np
import torch as th
import os
import random
import scipy as sci
from scipy.sparse import issparse
from ...tools._neighbors import Neighbors
from .xp_utility import spsparse_to_thsparse, thsparse_to_spsparse
from ...utilis._clean_cache import clean_cache

class thopt():
    def __init__(self, device=None, device_pre='cpu', floatx=None, seed=None, ):
        self.xp = th
        self.device = (self.xp.device('cuda' if self.xp.cuda.is_available() else 'cpu') 
                       if device is None else device)
        self.device_pre = device_pre
        self.device_index = self.xp.device(self.device).index
        self.floatx = eval(f'self.xp.{ floatx or "float32"}')
        self.seed_torch(seed)
        self.eps = self.to_tensor(self.xp.finfo(self.floatx).eps,
                                  dtype=self.floatx, device=self.device)
        self.spsparse_to_thsparse = spsparse_to_thsparse
        self.thsparse_to_spsparse = thsparse_to_spsparse
        self.clean_cache = clean_cache
    
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

    def is_sparse(self, X):
        if issparse(X):
            return True, 'scipy'
        elif th.is_tensor(X):
            return X.is_sparse, 'torch'
        else:
            return False, 'numpy'

    def to_tensor(self, X, dtype=None, device=None):
        device = self.device_pre if device is None else device
        dtype = self.floatx if dtype is None else dtype
        if th.is_tensor(X):
            X = X.clone()
        elif issparse(X):
            X = self.spsparse_to_thsparse(X)
        else:
            X = th.tensor(X, dtype=dtype)
        return X.to(dtype).to(device)

    def centerlize(self, X, Xm=None, Xs=None, device=None):
        device = self.device_pre if device is None else device
        if X.is_sparse: 
            X = X.to_dense()

        X = X.clone().to(device)
        N,D = X.shape
        Xm = self.xp.mean(X, 0) if Xm is None else Xm.to(device)

        X -= Xm
        Xs = self.xp.sqrt(self.xp.sum(self.xp.square(X))/N) if Xs is None else Xs.to(device)  #(N*D) 
        X /= Xs
        Xf = self.xp.eye(D+1, dtype=self.floatx, device=device)
        Xf[:D,:D] *= Xs
        Xf[:D, D] = Xm
        return [X, Xm, Xs, Xf]

    def normalize(self, X, device=None):
        device = self.device_pre if device is None else device
        if X.is_sparse: 
            X = X.to_dense()

        X = X.clone().to(device)
        l2x = self.xp.linalg.norm(X, ord=None, dim=1, keepdim=True)

        l2x[l2x == 0] = 1
        return X/l2x

    def scaling(self, X, anis_var=False, zero_center = True, device=None):
        device = self.device_pre if device is None else device
        if X.is_sparse: 
            X = X.to_dense()

        X = X.clone().to(device)
        mean = X.mean(dim=0, keepdim=True)

        if anis_var:
            std  = X.std(dim=0, keepdim=True)
            std[std == 0] = 1
        else:
            std =  X.std() or 1
        if zero_center:
            X -= mean
        X /=  std
        return X

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
            R = self.xp.asarray(R, dtype=dist2.dtype, device=dist2.device)
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

    def sigma_square(self, X, Y):
        [N, D] = X.shape
        [M, D] = Y.shape
        # sigma2 = (M*np.trace(np.dot(np.transpose(X), X)) + 
        #           N*np.trace(np.dot(np.transpose(Y), Y)) - 
        #           2*np.dot(np.sum(X, axis=0), np.transpose(np.sum(Y, axis=0))))
        sigma2 = (M*self.xp.sum(X * X) + 
                  N*self.xp.sum(Y * Y) - 
                  2* self.xp.sum(self.xp.sum(X, 0) * self.xp.sum(Y, 0)))
        sigma2 /= (N*M*D)
        return sigma2

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
