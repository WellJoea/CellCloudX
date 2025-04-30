import torch as th
from ._neighbors import ko_knn

def swnn(X, Y, Xf, Yf, mnn=10, snn=30, fnn=60, temp=1.0,
        scale_locs=True, scale_feats=True, use_soft=False,
        lower = 0.01, upper=0.995,
        min_score= 0.35,
        max_pairs=5e4,
        min_pairs=100,
        version=1,
        device=None, dtype=th.float32, verbose=False):

    device = X.device if device is None else device
    dtype = X.dtype if dtype is None else dtype

    if scale_locs:
        X =centerlize(X.to(device, dtype=dtype).clone())
        Y =centerlize(Y.to(device, dtype=dtype).clone())
    else:
        X =X.to(device, dtype=dtype).clone()
        Y =Y.to(device, dtype=dtype).clone()
    if scale_feats:
        Xf =normlization(Xf.to(device, dtype=dtype).clone())
        Yf =normlization(Yf.to(device, dtype=dtype).clone())
    else:
        Xf =Xf.to(device, dtype=dtype).clone()
        Yf =Yf.to(device, dtype=dtype).clone()
    if use_soft:
        swnn = sswnn_adj(X, Y, Xf, Yf, mnn=mnn, snn=snn, fnn=fnn, temp=temp,)
    else:
        swnn = swnn_adj(X, Y, Xf, Yf, mnn=mnn, snn=snn, fnn=fnn, temp=temp, version=version)

    mscore = swnn_score(swnn, lower=lower, upper=upper, verbose=verbose)

    t_idx = th.where(mscore>=min_score)[0]
    verbose and print(f'sswnn pairs filtered by min_score: {t_idx.shape[0]}...')
    
    if (min_pairs is not None):
        min_pairs = int(min(min_pairs, mscore.shape[0]))
        kidx = th.topk(mscore, min_pairs).indices
        t_idx = th.unique(th.cat([kidx, t_idx]))

    if (max_pairs is not None):
        max_pairs = int(min(max_pairs, t_idx.shape[0]))
        mscore1 = mscore[t_idx]
        ind = th.topk(mscore1, max_pairs).indices
        t_idx = t_idx[ind]
        verbose and print(f'sswnn pairs filtered by max_pairs: {t_idx.shape[0]}...')
    
    pairs = swnn.indices()[:,t_idx]
    mscore = mscore[t_idx]
    return pairs, mscore

def swnn_score(swnn, lower = 0.01, upper=0.995, verbose=False):
    values  = swnn.values()
    # indices = swnn.indices()
    mhits = values[values>0]
    if len(mhits) < 100:
        if verbose:
            print('too few swnn pairs.')

    min_score = th.quantile(mhits, lower)
    max_score = th.quantile(mhits, upper)
    values = (values-min_score)/(max_score-min_score)
    values  = th.clip(values, 0, 1)
    return values

def sswnn_adj(X, Y, Xf, Yf, mnn=6, snn=30, fnn=60, temp=1.0,):
    qr = ko_knn(Xf, Yf, K=fnn)
    qri, qrd = qr.knn(return_distances=True)
    qrd.mul_(-0.5).add_(1.0).clip_(0, 1.0).sqrt_()
    qrs = qr.resparse(qri, qrd).to_sparse_coo()

    rq = ko_knn(Yf, Xf, K=fnn)
    rqi, rqd = rq.knn(return_distances=True)
    rqd.mul_(-0.5).add_(1.0).clip_(0, 1.0).sqrt_()
    rqs = rq.resparse(rqi, rqd).to_sparse_coo()

    grr = ko_knn(X, K=snn).knn(return_sparse=True, return_distances=True)[1]
    grr.values().div_(-2.0*temp).exp_()
    grr = grr.to_sparse_coo()

    gqq = ko_knn(Y, K=snn).knn(return_sparse=True, return_distances=True)[1]
    gqq.values().div_(-2.0*temp).exp_()
    gqq = gqq.to_sparse_coo()

    # Ssmm = th.sparse.mm(grr, qrs.T) * th.sparse.mm(rqs, gqq.T)
    try:
        A = th.sparse.mm(qrs, grr).T
        B = th.sparse.mm(rqs, gqq)
        Ssmm = A * B
        Ssmm = Ssmm.coalesce()
        Ssmm.values().sqrt_().div_(float(min(fnn, snn)))
    except:
        Ssmm = th.sparse.mm(qrs.to('cpu'), grr.to('cpu')).T * \
               th.sparse.mm(rqs.to('cpu'), gqq.to('cpu'))
        Ssmm = Ssmm.coalesce()
        Ssmm = th.sqrt(Ssmm) / float(min(fnn, snn))
        Ssmm = Ssmm.to(X.device)

    # Fmnn = (rqm * qrm.T)
    # Fmnn = Fmnn.coalesce()
    rqm = rq.resparse(rqi[:,:mnn], rqd[:,:mnn])
    qrm = qr.resparse(qri[:,:mnn], qrd[:,:mnn])
    Asswmm = th.sqrt(rqm * qrm.T * Ssmm).coalesce()
    return Asswmm

def swnn_adj(X, Y, Xf, Yf, mnn=6, snn=30, fnn=60, temp=1.0, version=1,):
    qr = ko_knn(Xf, Yf, K=fnn)
    qri, qrd = qr.knn(return_distances=True)
    qrs = qr.resparse(qri, None)

    rq = ko_knn(Yf, Xf, K=fnn)
    rqi, rqd = rq.knn(return_distances=True)
    rqs = rq.resparse(rqi, None)

    grr = ko_knn(X, K=snn).knn(return_sparse=True, return_distances=True)[0]
    gqq = ko_knn(Y, K=snn).knn(return_sparse=True, return_distances=True)[0]

    try:
        qrm = qr.resparse(qri[:,:mnn], None)
        rqm = rq.resparse(rqi[:,:mnn], None)

        if version == 1:
            A = qrm*(gqq @ qrs).coalesce()
            B = rqm*(grr @ rqs).coalesce()
        elif version == 2:
            A = qrm*(gqq @ qrs @ grr.T).coalesce()
            B = rqm*(grr @ rqs @ gqq.T).coalesce()
    
        Ssmm = A.T * B
        Ssmm = Ssmm.coalesce()

        if version == 1:
            Ssmm.values().sqrt_().div_(float(min(fnn, snn))).sqrt_()
        elif version == 2:
            Ssmm.values().sqrt_().div_(float(min(fnn, snn))**2).sqrt_()
        Aswmm = Ssmm
    except:
        qrm = qr.resparse(qri[:,:mnn], None).to('cpu')
        rqm = rq.resparse(rqi[:,:mnn], None).to('cpu')

        if version == 1:
            A = qrm*(gqq.to('cpu') @ qrs.to('cpu')).coalesce()
            B = rqm*(grr.to('cpu') @ rqs.to('cpu')).coalesce()
        elif version == 2:
            A = qrm*(gqq.to('cpu') @ qrs.to('cpu') @ grr.T.to('cpu')).coalesce()
            B = rqm*(grr.to('cpu') @ rqs.to('cpu') @ gqq.T.to('cpu')).coalesce()

        Ssmm = A.T * B
        Ssmm = Ssmm.coalesce()
        Ssmm = Ssmm.to(X.device)
        if version == 1:
            Ssmm.values().sqrt_().div_(float(min(fnn, snn))).sqrt_()
        elif version == 2:
            Ssmm.values().sqrt_().div_(float(min(fnn, snn))**2).sqrt_()
        Aswmm = Ssmm

    # Ssmm = th.sparse.mm(qrs, grr).T * th.sparse.mm(rqs, gqq)
    # Ssmm = th.sparse.mm(gqq, qrs).T * th.sparse.mm(grr, rqs)
    # Ssmm= (gqq @ qrs @ grr.T).T* (grr @ rqs @ gqq.T)
    # Ssmm = Ssmm.coalesce()
    # Ssmm = th.sqrt(Ssmm) / float(min(fnn, snn))
    return Aswmm

def swnn_adj0(X, Y, Xf, Yf, mnn=6, snn=30, fnn=60, temp=1.0,):
    qr = ko_knn(Xf, Yf, K=fnn)
    qri, qrd = qr.knn(return_distances=True)
    qrd.mul_(-0.5).add_(1.0).clip_(0, 1.0).sqrt_()
    qrm = qr.resparse(qri[:,:mnn], qrd[:,:mnn])
    qrs = qr.resparse(qri, qrd)

    rq = ko_knn(Yf, Xf, K=fnn)
    rqi, rqd = rq.knn(return_distances=True)
    rqd.mul_(-0.5).add_(1.0).clip_(0, 1.0).sqrt_()
    rqm = rq.resparse(rqi[:,:mnn], rqd[:,:mnn])
    rqs = rq.resparse(rqi, rqd)

    grr = ko_knn(X, K=snn).knn(return_sparse=True, return_distances=True)[1]
    grr.values().div_(-2.0*temp).exp_()

    gqq = ko_knn(Y, K=snn).knn(return_sparse=True, return_distances=True)[1]
    gqq.values().div_(-2.0*temp).exp_()

    Ssmm = th.sparse.mm(grr, qrs.T) * th.sparse.mm(rqs, gqq.T)
    Ssmm = Ssmm.coalesce()
    Ssmm = th.sqrt(Ssmm) / float(min(fnn, snn))
    # Fmnn = (rqm * qrm.T)
    # Fmnn = Fmnn.coalesce()
    Aswmm = th.sqrt(rqm*qrm.T*Ssmm).coalesce()
    return Aswmm

def centerlize(X, Xm=None, Xs=None, device=None, xp=th):
    device = X.device if device is None else device
    if X.is_sparse: 
        X = X.to_dense()

    X = X.clone().to(device)
    N,D = X.shape
    Xm = xp.mean(X, 0) if Xm is None else Xm.to(device)

    X -= Xm
    Xs = xp.sqrt(xp.sum(xp.square(X))/(N*D/2)) if Xs is None else Xs.to(device) # N
    X /= Xs
    Xf = xp.eye(D+1, device=device)
    Xf[:D,:D] *= Xs
    Xf[:D, D] = Xm
    return X

def normlization(X):
    l2x = th.linalg.norm(X, ord=None, axis=1, keepdims=True)
    l2x[l2x == 0] = 1
    return X / l2x

def sigma_square(X, Y, xp=th):
    [N, D] = X.shape
    [M, D] = Y.shape
    # sigma2 = (M*np.trace(np.dot(np.transpose(X), X)) + 
    #           N*np.trace(np.dot(np.transpose(Y), Y)) - 
    #           2*np.dot(np.sum(X, axis=0), np.transpose(np.sum(Y, axis=0))))
    sigma2 = (M*xp.sum(X * X) + 
                N*xp.sum(Y * Y) - 
                2* xp.sum(xp.sum(X, 0) * xp.sum(Y, 0)))
    sigma2 /= (N*M*D)
    return sigma2