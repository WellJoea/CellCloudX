import numpy as np
from scipy.spatial import distance_matrix
from scipy.spatial import distance as scipy_distance
from scipy.sparse import issparse, csr_array, csc_array, diags, linalg
import scipy.sparse as ssp

def centerlize(X, xp, Xm=None, Xs=None,):
    X = xp.array(X)
    N,D = X.shape
    Xm = np.mean(X, axis=0) if Xm is None else Xm
    X -= Xm
    Xs = np.sqrt(np.sum(np.square(X))/N) if Xs is None else Xs  #(N*D) 
    X /= Xs
    Xf = np.eye(D+1, dtype=np.float64)
    Xf[:D,:D] *= Xs
    Xf[:D, D] = Xm
    return [X, Xm, Xs, Xf]
    
def scale_array( X,
                zero_center = True,
                anis_var = False,
                axis = 0,
    ):
    if issparse(X):
        X = X.toarray()
    X = X.copy()
    N,D = X.shape

    mean = np.expand_dims(np.mean(X, axis=axis), axis=axis)

    if anis_var:
        std  = np.expand_dims(np.std(X, axis=axis, ddof=0), axis=axis)
        std[std == 0] = 1
    else:
        std = np.std(X)

    if zero_center:
        X -= mean
    X /=  std

    mean = np.squeeze(mean)
    std  = np.squeeze(std)
    Xf = np.eye(D+1, dtype=np.float64)
    Xf[:D,:D] *= std
    Xf[:D, D] = mean

    return X, mean, std, Xf

def is_positive_semi_definite(R):
    if not isinstance(R, (np.ndarray, np.generic)):
        raise ValueError('Encountered an error while checking if the matrix is positive semi definite. \
            Expected a numpy array, instead got : {}'.format(R))
    return np.all(np.linalg.eigvals(R) > 0)

def sigma_square(X, Y):
    [N, D] = X.shape
    [M, D] = Y.shape
    # sigma2 = (M*np.trace(np.dot(np.transpose(X), X)) + 
    #           N*np.trace(np.dot(np.transpose(Y), Y)) - 
    #           2*np.dot(np.sum(X, axis=0), np.transpose(np.sum(Y, axis=0))))
    sigma2 = (M*np.sum(X * X) + 
              N*np.sum(Y * Y) - 
              2*np.dot(np.sum(X, axis=0), np.transpose(np.sum(Y, axis=0))))
    sigma2 /= (N*M*D)
    return sigma2

def gaussian_kernel(X, beta, Y=None):
    if Y is None:
        Y = X
    diff = X[:, None, :] - Y[None, :,  :]
    diff = np.square(diff)
    diff = np.sum(diff, 2)
    return np.exp(-diff / (2 * beta**2))

def Gmm(X_emb, Y_emb, norm=False, sigma2=None, temp=1, shift=0, xp = None):
    if xp is None:
        xp = np
    assert X_emb.shape[1] == Y_emb.shape[1]
    (N, D) = X_emb.shape
    M = Y_emb.shape[0]

    if norm:
        # X_emb = (X_emb - np.mean(X_emb, axis=0)) / np.std(X_emb, axis=0)
        # Y_emb = (Y_emb - np.mean(Y_emb, axis=0)) / np.std(Y_emb, axis=0)
        X_l2 =  X_emb/np.linalg.norm(X_emb, ord=None, axis=1, keepdims=True)
        Y_l2 =  Y_emb/np.linalg.norm(Y_emb, ord=None, axis=1, keepdims=True)
    else:
        X_l2 = X_emb
        Y_l2 = Y_emb
    
    Dist = dist_matrix(X_l2, Y_l2)
    if sigma2 is None:
        sigma2 =np.sum(Dist) / (D*N*M)
    P = np.exp( (shift-Dist) / (2 * sigma2 * temp))
    return P, sigma2

def kGmm(X_emb, Y_emb, col, row, temp=1):
    assert X_emb.shape[1] == Y_emb.shape[1]
    (N, D) = X_emb.shape
    M = Y_emb.shape[0]

    Dist = (X_emb[row] - Y_emb[col])**2
    Dist = np.sum(Dist, axis=-1)

    sigma2 =np.mean(Dist) / D
    # sigma2 =np.sum(Dist) / (D*N*M)
    P = np.exp( -Dist / (2 * sigma2 * temp))
    return P, sigma2

# def dist_matrix(X, Y):
#     dist = scipy_distance.cdist(X, Y, "sqeuclidean")
#     return dist.T

def dist_matrix(X, Y, p=2, threshold=1000000):
    dist = distance_matrix(X, Y, p=p, threshold=threshold).T #D,M -> M,D
    return dist ** 2
    # (N, D) = X.shape
    # (M, _) = Y.shape
    # if chunck is None:
    #     diff = X[None, :, :] - Y[:, None, :] # (1, N, D) - (M ,1, D)
    #     dist = diff ** 2
    #     return np.sum(dist, axis=-1)
    # else:
    #     C = min(chunck, N)
    #     splits = np.arange(0, N+C, C).clip(0, N)
    #     Xb = X[None, :, :] 
    #     Yb = Y[:, None, :] 
    #     dist = []
    #     for idx in range(len(splits)-1):
    #         islice = slice(splits[idx], splits[idx+1])
    #         idist = np.sum( (Xb[:,islice,:] - Yb)** 2, axis=-1)
    #         dist.append(idist)
    #     return np.concatenate(dist, axis=0)

def normalAdj(A):
    d1 = np.sqrt(np.sum(A, axis = 1))
    d1[(np.isnan(d1) | (np.isinf(d1) ))] = 0
    d1 = np.diag(1/d1)

    d2 = np.sqrt(np.sum(A, axis = 0))
    d2[(np.isnan(d2) | (np.isinf(d2) ))] = 0
    d2 = np.diag(1/d2)

    return d1 @ A @ d2

def lowrankQS(G, beta, num_eig, eig_fgt=False):
    """
    Calculate eigenvectors and eigenvalues of gaussian matrix G.
    
    !!!
    This function is a placeholder for implementing the fast
    gauss transform. It is not yet implemented.
    !!!

    Attributes
    ----------
    G: numpy array
        Gaussian kernel matrix.
    
    beta: float
        Width of the Gaussian kernel.
    
    num_eig: int
        Number of eigenvectors to use in lowrank calculation of G
    
    eig_fgt: bool
        If True, use fast gauss transform method to speed up. 
    """

    # if we do not use FGT we construct affinity matrix G and find the
    # first eigenvectors/values directly

    if eig_fgt is False:
        S, Q = np.linalg.eigh(G)
        eig_indices = list(np.argsort(np.abs(S))[::-1][:num_eig])
        Q = Q[:, eig_indices]  # eigenvectors
        S = S[eig_indices]  # eigenvalues.

        return Q, S

    elif eig_fgt is True:
        raise Exception('Fast Gauss Transform Not Implemented!')

def low_rank_eigen(G, num_eig):
    """
    Calculate num_eig eigenvectors and eigenvalues of gaussian matrix G.
    Enables lower dimensional solving.
    """
    S, Q = np.linalg.eigh(G)
    eig_indices = list(np.argsort(np.abs(S))[::-1][:num_eig])
    Q = Q[:, eig_indices]  # eigenvectors
    S = S[eig_indices]  # eigenvalues.
    return Q, S
