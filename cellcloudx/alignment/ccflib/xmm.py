# from .xp_operation import xpopt
import numpy as np
from scipy.spatial import distance as scipy_distance
import  scipy as sci 
from scipy.sparse import issparse, csr_array, csc_array, diags, linalg
import scipy.sparse as ssp
from sklearn.metrics.pairwise import euclidean_distances

from ...tools._neighbors import Neighbors

def kernel_xmm(X_emb, Y_emb, sigma2=None, temp=1,
               dfs=1, shift=0, kernel='gmm',
               xp = None, floatx=np.float64):
    if xp is None:
        xp = np
    assert X_emb.shape[1] == Y_emb.shape[1]
    (N, D) = X_emb.shape
    M = Y_emb.shape[0]

    # Dist2 = scipy_distance.cdist(Y_emb, X_emb, "sqeuclidean")
    Dist2 = euclidean_distances(Y_emb, X_emb, squared=True)
    Dist2 = Dist2.astype(floatx)
    if sigma2 is None:
        sigma2 = np.sum(Dist2) / (D*N*M)

    P,R = dist2prob(Dist2, sigma2, D, 
                   kernel=kernel, shift=shift, dfs=dfs, temp=temp)
    return P,R, sigma2

def kernel_xmm_p(X_emb, Y_emb, pairs, sigma2=None, temp=1,
               dfs=1, shift=0, kernel='gmm',
               xp = None, floatx=np.float64):
    if xp is None:
        xp = np
    assert X_emb.shape[1] == Y_emb.shape[1]
    (N, D) = X_emb.shape
    M = Y_emb.shape[0]

    Dist2 = np.square(X_emb[pairs[0]] - Y_emb[pairs[1]]).astype(floatx)
    Dist2 = np.sum(Dist2, axis=-1)
    sigma2 = np.mean(Dist2)/D if sigma2 is None else sigma2

    P,R = dist2prob(Dist2, sigma2, D, 
                  kernel=kernel, shift=shift, dfs=dfs, temp=temp)
    P = csr_array((P, (pairs[0], pairs[1])), shape=(N, M), dtype=floatx)
    return P,R, sigma2

def kernel_xmm_k(X, Y, sigma2=None, temp=1,
                dfs=1, shift=0, kernel='gmm', method='hnsw',
                metric='euclidean', 
                knn=50, radius=None, n_jobs=-1,
                xp = None, floatx=np.float64, **kargs):
    if xp is None:
        xp = np
    assert X.shape[1] == Y.shape[1]
    M,D = Y.shape
    N = X.shape[0]

    snn = Neighbors(method=method, metric=metric, n_jobs=n_jobs)
    snn.fit(Y, **kargs)
    ckdout = snn.transform(X, knn=knn, radius=radius)
    nnidx = ckdout[1]

    src = np.concatenate(nnidx, axis=0)
    dst = np.repeat(np.arange(len(nnidx)), list(map(len, nnidx)))
    Dist2 = np.concatenate(ckdout[0], axis=0)**2

    if sigma2 is None:
        # sigma2 = np.sum(Dist2) / (D*N*knn)
        sigma2 = np.mean(Dist2)/D

    P,R = dist2prob(Dist2, sigma2, D, 
                  kernel=kernel, shift=shift, dfs=dfs, temp=temp)
    P = csr_array((P, (src, dst)), shape=(M, N), dtype=floatx)
    return P,R, sigma2

def dist2prob(Dist2, sigma2, D, kernel='gmm', shift=0, dfs=2, temp=1):
    # Calculate the value of the numerator
    if kernel == 'gmm':
        P = np.exp( (-Dist2) / (2 * sigma2 * temp))
        R = (2 * np.pi * sigma2) ** (-0.5 * D)
        return P, R
    elif kernel == 'smm':
        R  = sci.special.gamma((dfs + D) / 2.0)
        R /= sci.special.gamma(dfs/2.0) * (sigma2**0.5)
        R *= np.power(np.pi * dfs, -D/ 2.0) 
        P = np.power(1.0 + Dist2 / (sigma2*dfs), -(dfs + D) / 2.0)
        return P, R
    elif kernel is None:
        P = Dist2
        R = 1
        return P, R

def lle_W(Y, Y_index=None, kw=15, rl_w=None,  method='sknn',
           eps=np.finfo(np.float64).eps):
    M, D = Y.shape

    #D2 =np.sum(np.square(Y[:, None, :]- Y[None, :, :]), axis=-1)
    #D3 = D2 + np.eye(D2.shape[0])*D2.max()
    #cidx = np.argpartition(D3, self.knn)[:, :self.knn]

    eps = np.finfo(np.float64).eps
    if rl_w is None:
        rl_w = 1e-3 if(kw>D) else 0

    snn = Neighbors(method=method)
    snn.fit(Y)
    ckdout = snn.transform(Y, knn=kw+1)
    kdx = ckdout[1][:,1:]
    L = []
    for i in range(M):
        kn = kdx[i]
        z = (Y[kn] - Y[i]) #K*D
        G = z @ z.T # K*K
        Gtr = np.trace(G)
        if Gtr>0:
            G = G +  np.eye(kw) * rl_w* Gtr
        w = np.sum(np.linalg.inv(G), axis=1) #K*1
        #w = solve(G, v, assume_a="pos")
        w = w/ np.sum(w).clip(eps, None)
        L.append(w)
    src = kdx.flatten('C')
    dst = np.repeat(np.arange(kdx.shape[0]), kdx.shape[1])
    L = ssp.csr_array((np.array(L).flatten(), (dst, src)), shape=(M, M))
    if Y_index is not None:
        L = L[Y_index][:, Y_index]
    L  = ssp.eye(M) - L
    return L

def gl_w(Y, Y_index=None, Y_feat=None, kw=50, method='sknn'):
    '''
    Y_feat : L2 normal
    '''
    M, D = Y.shape
    snn = Neighbors(method=method)
    snn.fit(Y)
    ckdout = snn.transform(Y, knn=kw+1)
    kdx = ckdout[1][:,1:]
    src = kdx.flatten('C')
    dst = np.repeat(np.arange(kdx.shape[0]), kdx.shape[1])
    if not Y_feat is None: #TODO
        pass
    else:
        A = np.ones_like(dst)
    L = ssp.csr_array((A, (dst, src)), shape=(M, M))
    if Y_index is not None:
        L = L[Y_index][:, Y_index]

    D = ssp.diags((L.sum(1) )**(-0.5))
    A = D @ L @ D
    K  = ssp.eye(A.shape[0]) - A
    return K

def sigma_square(X, Y):
    [N, D] = X.shape
    [M, D] = Y.shape

    sigma2 = (M*np.sum(X * X) + 
            N*np.sum(Y * Y) - 
            2*np.dot(np.sum(X, axis=0), np.transpose(np.sum(Y, axis=0))))
    sigma2 /= (N*M*D)
    return sigma2
    
def low_rank_eigen(G, num_eig):
    S, Q = np.linalg.eigh(G)
    eig_indices = list(np.argsort(np.abs(S))[::-1][:num_eig])
    Q = Q[:, eig_indices]  # eigenvectors
    S = S[eig_indices]  # eigenvalues.
    return Q, S

def low_rank_eigen_grbf(X, h, num_eig, sw_h=0):
    from probreg import gauss_transform as gt
    M = X.shape[0]
    k = min(M-1, num_eig)
    trans = gt.GaussTransform(X, h, sw_h=sw_h)
    def matvec(x):
        return trans.compute(X, x.T)
    lo = sci.sparse.linalg.LinearOperator((M,M), matvec)
    S, Q = sci.sparse.linalg.eigs(lo, k=k, which='LM')

    eig_indices = list(np.argsort(np.abs(S))[::-1][:num_eig])
    Q = Q[:, eig_indices]  # eigenvectors
    S = S[eig_indices]  # eigenvalues.
    return np.real(Q), np.real(S)

def low_rank_eigen_sp(G, num_eig, floatx=np.float64):
    k = min(G.shape[0]-1, num_eig)
    S, Q = sci.sparse.linalg.eigs(G.astype(floatx), k=k)

    eig_indices = list(np.argsort(np.abs(S))[::-1][:num_eig])
    Q = Q[:, eig_indices]  # eigenvectors
    S = S[eig_indices]  # eigenvalues.
    return np.real(Q), np.real(S)

def Nystrom_low_rank(Y, G=None, rank=None, Beta=2, seed=200504): #TODO close to singular in reality
    np.random.seed(seed)
    if G is None:
        G = kernel_xmm(Y, Y, sigma2=Beta, temp=1)[0]

    M = G.shape[0]
    if rank is None:
        rank = int(max([M**.4, 1000,]))
    uidx = np.where((G>0.99).sum(1)>1)[0]  # Kss error 
    rank = min(rank, uidx.shape[0])
    idx = np.random.choice(uidx, rank, replace=False)

    Kns = G[:, idx]
    Kss = Kns[idx,:]
    assert not np.allclose(np.linalg.det(Kss), 0) # ERROR in singular
    #from numpy.linalg import cond
    #cond(Kss)       
    K = Kns @ np.linalg.inv(Kss) @ Kns.T 
    return csc_array(K, shape=(M,M))

def Nystrom_low_rank1(Y,  num_eig, rank=None, Beta=2, seed=200504): #TODO
    np.random.seed(seed)
    M, D = Y.shape
    if rank is None:
        rank = int(max([M**.4, 500, num_eig]))

    uidx = np.where((G>0.99).sum(1)>1)[0]  # Kss error 
    rank = min(rank, uidx.shape[0])
    idx = np.random.choice(uidx, rank, replace=False)

    Ys = Y[idx]
    YN = kernel_xmm(Y, Ys, sigma2=Beta)[0]
    NN = kernel_xmm(Ys, Ys, sigma2=Beta)[0]
    assert not np.allclose(np.linalg.det(NN), 0) # ERROR in singular
    K = YN @ np.linalg.inv(NN) @ YN.T
    K = csc_array(K)

def Nystrom_low_rank_eigen(Y, num_eig, rank=None, Beta=1, seed=200504): #TODO
    K = Nystrom_low_rank(Y,  rank=max(num_eig, rank), Beta=Beta, seed=seed)
    return low_rank_eigen_sp(K, num_eig)

def WoodburyB(Av, U, Cv, V):
    UCv = np.linalg.inv(Cv  + V @ Av @ U)
    return  Av - (Av @ U) @ UCv @ (V @ Av)

def WoodburyA(A, U, C, V):
    Av = np.linalg.inv(A)
    Cv = np.linalg.inv(C)
    UCv = np.linalg.inv(Cv  + V @ Av @ U)
    return  Av - (Av @ U) @ UCv @ (V @ Av)
