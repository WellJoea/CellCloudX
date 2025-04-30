import numpy as np
import scipy.sparse as ssp
from ...tools._neighbors import Neighbors

def lle_w(Y, kw=15, use_unique=False, rl_w=None,  method='sknn'): #TODO 
    #D2 =np.sum(np.square(Y[:, None, :]- Y[None, :, :]), axis=-1)
    #D3 = D2 + np.eye(D2.shape[0])*D2.max()
    #cidx = np.argpartition(D3, self.knn)[:, :self.knn]

    if hasattr(Y, 'detach'):
        uY = Y.detach().cpu().numpy()
        is_tensor = True
        device = Y.device
        dtype = Y.dtype
    else:
        uY = Y
        is_tensor = False

    if use_unique:
        uY, Yidx = np.unique(uY, return_inverse=True,  axis=0)
    eps = np.finfo(uY.dtype).eps
    M, D = uY.shape
    Mr =  Y.shape[0]

    if rl_w is None:
        rl_w = 1e-5 if(kw>D) else 0

    snn = Neighbors(method=method)
    snn.fit(uY)
    ckdout = snn.transform(uY, knn=kw+1)
    kdx = ckdout[1][:,1:]
    L = []
    for i in range(M):
        kn = kdx[i]
        z = (uY[kn] - uY[i]) #K*D
        G = z @ z.T # K*K
        Gtr = np.trace(G)
        if Gtr != 0:
            G = G +  np.eye(kw) * rl_w* Gtr
        else:
            G = G +  np.eye(kw) * rl_w
        w = np.sum(np.linalg.inv(G), axis=1) #K*1
        #w = solve(G, v, assume_a="pos")
        w = w/ np.sum(w).clip(eps, None)
        L.append(w)
    src = kdx.flatten('C')
    dst = np.repeat(np.arange(kdx.shape[0]), kdx.shape[1])
    L = ssp.csr_array((np.array(L).flatten(), (dst, src)), shape=(M, M))
    if use_unique:
        L = L[Yidx][:, Yidx]
    L  = ssp.eye(Mr) - L
    if is_tensor:
        L = spsparse_to_thsparse(L).to(dtype).to(device)
    return L

def gl_w(Y, Y_feat=None, kw=15, use_unique=False, rl_w=None,  method='sknn'): #TODO 
    if hasattr(Y, 'detach'):
        uY = Y.detach().cpu().numpy()
        is_tensor = True
        device = Y.device
        dtype = Y.dtype
    else:
        uY = Y
        is_tensor = False

    if use_unique:
        uY, Yidx = np.unique(uY, return_inverse=True,  axis=0)

    M, D = uY.shape

    if rl_w is None:
        rl_w = 1e-3 if(kw>D) else 0

    snn = Neighbors(method=method)
    snn.fit(uY)
    ckdout = snn.transform(uY, knn=kw+1)
    kdx = ckdout[1][:,1:]
    src = kdx.flatten('C')
    dst = np.repeat(np.arange(kdx.shape[0]), kdx.shape[1])

    if not Y_feat is None: #TODO
        pass
    else:
        A = np.ones_like(dst)

    L = ssp.csr_array((A, (dst, src)), shape=(M, M))
    if use_unique:
        L = L[Yidx][:, Yidx]

    D = ssp.diags((L.sum(1) )**(-0.5))
    A = D @ L @ D
    K  = ssp.eye(A.shape[0]) - A

    if is_tensor:
        K = spsparse_to_thsparse(K).to(dtype).to(device)
    return K

def low_rank_eigen_grbf(X, h, num_eig, sw_h=0.0, eps=1e-10):
    from ...third_party._ifgt_warp import GaussTransform, GaussTransform_fgt
    # from probreg.gauss_transform import GaussTransform as GaussTransform_fgt

    if hasattr(X, 'detach'):
        is_tensor = True
        device = X.device
        dtype = X.dtype
        X = X.detach().cpu().numpy()
    else:
        is_tensor = False
    # X = X.copy().astype(np.float32)
    M = X.shape[0]
    k = min(M-1, num_eig)
    trans = GaussTransform_fgt(X, h, sw_h=sw_h, eps=eps) #XX*s/h, s
    def matvec(x):
        return trans.compute(X, x.T)
    lo = ssp.linalg.LinearOperator((M,M), matvec)
    S, Q = ssp.linalg.eigs(lo, k=k, which='LM')

    eig_indices = np.argsort(-np.abs(S))[:k]
    Q = np.real(Q[:, eig_indices])  # eigenvectors
    S = np.real(S[eig_indices])  # eigenvalues.

    if is_tensor:
        import torch as th
        Q = th.tensor(Q, dtype=dtype, device=device)
        S = th.tensor(S, dtype=dtype, device=device)
    return Q, S

def low_rank_svd_grbf(X, Y, h, num_eig, sw_h=0.0, use_complete=True, eps =1e-10):
    from CellCloudX.cellcloudx.third_party._ifgt_warp import GaussTransform, GaussTransform_fgt

    if hasattr(X, 'detach'):
        is_tensor = True
        device = X.device
        dtype = X.dtype
        X = X.detach().cpu().numpy()
        Y = Y.detach().cpu().numpy()
    else:
        is_tensor = False
    # X = X.copy().astype(np.float32)
    N, M = X.shape[0], Y.shape[0]
    k = min(M-1, N-1, num_eig)

    if use_complete:
        tran1 = GaussTransform_fgt(X, h, sw_h=sw_h, eps=eps)
        tran2 = GaussTransform_fgt(Y, h, sw_h=sw_h, eps=eps)
        def matvec(x):
            return tran2.compute(X, x.T)
        def rmatvec(y):
            return tran1.compute(Y, y.T)
    else: #not recommended
        tran1 = GaussTransform_fgt(X, h, sw_h=sw_h, eps=eps)
        def matvec(x):
            return tran1.compute(X, x.T)
        def rmatvec(y):
            return tran1.compute(Y, y.T)

    lo = ssp.linalg.LinearOperator((N,M), matvec, rmatvec)
    U, S, Vh = ssp.linalg.svds(lo, k=k, which='LM')

    eig_indices = np.argsort(-np.abs(S))[:k]
    U = np.real(U[:, eig_indices])
    S = np.real(S[eig_indices])
    Vh = np.real(Vh[eig_indices,:])
    if is_tensor:
        import torch as th
        U = th.tensor(U, dtype=dtype, device=device)
        S = th.tensor(S, dtype=dtype, device=device)
        Vh = th.tensor(Vh, dtype=dtype, device=device)
    return U, S, Vh

def low_rank_eigen(G, num_eig, xp=None): #TODO KernelPCA
    if hasattr(G, 'detach'):
        import torch as th
        is_tensor = True
        device = G.device
        dtype = G.dtype
        xp = th
    else:
        is_tensor = False
        xp = np if xp is None else xp
    if G.shape[0] == G.shape[1]:
        S, Q = xp.linalg.eigh(G) #only for symmetric 
        eig_indices = xp.argsort(-xp.abs(S))[:num_eig]
        Q = Q[:, eig_indices]
        S = S[eig_indices]
        if is_tensor:
            Q = Q.clone().to(dtype).to(device)
            S = S.clone().to(dtype).to(device)
        return Q, S
    else:
        U, S, Vh = xp.linalg.svd(G)
        eig_indices = xp.argsort(-xp.abs(S))[:num_eig]
        U = U[:, eig_indices]
        S = S[eig_indices]
        Vh = Vh[eig_indices,:]
        if is_tensor:
            U = U.clone().to(dtype).to(device)
            S = S.clone().to(dtype).to(device)
            Vh = Vh.clone().to(dtype).to(device)
        return U, S, Vh

def compute_svd( X, Y, h2, num_eig, use_ifgt = 1e7, xp=None):
    if (X.shape[0] * Y.shape[0]) >= use_ifgt:
        U, S, Vh = low_rank_svd_grbf(X, Y, h2**0.5, num_eig)
    else:
        G = xp.cdist(X, Y)
        G.pow_(2).div_(-h2)
        G.exp_()
        U, S, Vh = low_rank_eigen(G, num_eig)
    return U, S, Vh

def compute_eigen(X, h2, num_eig, use_ifgt = 1e9, xp=None):
    if (X.shape[0] **2) >= use_ifgt:
        Q, S = low_rank_eigen_grbf(X, h2**0.5, num_eig)
    else:
        G = xp.cdist(X, X)
        G.pow_(2).div_(-h2)
        G.exp_()
        Q, S = low_rank_eigen(G, num_eig)
    return Q, S

def WoodburyC(Av, U, Cv, V, xp=np):
    UCv = xp.linalg.inv(Cv  + Av * (V @ U))
    Fc = -(Av * Av) * (U @ UCv @ V)
    Fc.diagonal().add_(Av)
    return Fc

def WoodburyB(Av, U, Cv, V, xp=np):
    UCv = xp.linalg.inv(Cv  + V @ Av @ U)
    return  Av - (Av @ U) @ UCv @ (V @ Av)

def WoodburyA(A, U, C, V, xp=np):
    Av = xp.linalg.inv(A)
    Cv = xp.linalg.inv(C)
    UCv = xp.linalg.inv(Cv  + V @ Av @ U)
    return  Av - (Av @ U) @ UCv @ (V @ Av)

def spsparse_to_thsparse(X):
    import torch as th
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