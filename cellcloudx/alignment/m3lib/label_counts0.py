import numpy as np
from .neighbors_ensemble import Neighbors
from scipy.sparse import coo_matrix
from itertools import chain


def label_onehots(
    Ls,
    Xs=None,
    La=None,
    knn=0,
    radius=None,
    knn_method = 'sknn',
    normal='row',
    fuzzy_global=0,
    fuzzy_unique=0,
    include_self=True,
    tau=1.0,
    knn_qt=None,
    eps=1e-9,
    dtype=np.float32,
    n_jobs=-1,
):

    Ls = [np.asarray(L) for L in Ls]
    K = len(Ls)

    if La is None:
        La = np.unique(np.concatenate(Ls))
    La = np.asarray(La)
    J = len(La)

    if isinstance(knn, (int, np.integer)):
        knns = [int(knn)] * K
    elif len(knn) == K:
        knns = list(knn)
    else:
        raise ValueError("knn must have length K.")
    
    if isinstance(fuzzy_global, (list, tuple)):
        assert len(fuzzy_global) == K
    else:
        fuzzy_global = [fuzzy_global] * K
    if isinstance(fuzzy_unique, (list, tuple)):
        assert len(fuzzy_unique) == K
    else:
        fuzzy_unique = [fuzzy_unique] * K
    
    if radius is None:
        radii = [None] * K
    elif np.isscalar(radius):
        radii = [radius] * K
    elif len(radius) == K:
        radii = list(radius)
    else:
        raise ValueError("radius must have length K.")
    
    # if S is None:
    #     S = np.eye(J, dtype=dtype)
    # else:
    #     S = np.asarray(S, dtype=dtype).copy()
    #     if S.shape != (J, J):
    #         raise ValueError("S must have shape [J, J].")

    if Xs is None:
        Xs = [None] * K
    elif len(Xs) != K:
        raise ValueError("Xs must have length K.")

    present = np.stack([np.isin(La, np.unique(L)) for L in Ls])
    uu_ls = present & (present.sum(axis=0, keepdims=True) <= 1)
    As = []
    for k in range(K):
        A0, La0 = label_onehot(
            Ls[k],
            La=La,
            unique_idx=uu_ls[k],
            knn=knns[k],
            X=Xs[k],
            radius=radii[k],
            knn_method=knn_method,
            normal=normal,
            fuzzy_global=fuzzy_global[k],
            fuzzy_unique=fuzzy_unique[k],
            include_self=include_self,
            tau=tau,
            knn_qt=knn_qt,
            eps=eps,
            dtype=dtype,
            n_jobs=n_jobs,
        )
        As.append(A0)
    return As, La


def label_onehot(
    L,
    La=None,
    unique_idx=None,
    knn=0,
    radius=None,
    fuzzy_global = 0,
    fuzzy_unique= False,
    knn_method = 'sknn',
    X=None,
    normal=None,
    include_self=True,
    tau=1.0,
    knn_qt=0.995,
    eps=1e-8,
    dtype=np.float32,
    n_jobs = -1,
):
    L = np.asarray(L).squeeze()
    if La is None:
        La = np.unique(L)
    La = np.asarray(La).squeeze()
    
    if len(np.unique(La)) != len(La):
        raise ValueError("La contains duplicated labels.")

    N = L.shape[0]
    J = La.shape[0]
    knn = int(knn)

    u, inv = np.unique(L, return_inverse=True)
    _, iu, iLa = np.intersect1d(u, La, return_indices=True)

    mp = np.full(u.shape[0], -1, dtype=np.int64)
    mp[iu] = iLa
    ids = mp[inv]

    valid = ids >= 0
    if knn <= 0:
        A = np.zeros((N, J), dtype=dtype)
        A[np.arange(N)[valid], ids[valid]] = 1.0

    else:
        if X is None:
            raise ValueError("X must be provided when knn > 0.")

        X = np.asarray(X)
        k = min(knn + 1, N)

        # nn = NearestNeighbors(n_neighbors=k, n_jobs=n_jobs)
        # nn.fit(X)
        # dist, idx = nn.kneighbors(X, return_distance=True)
        snn = Neighbors(method=knn_method)
        snn.fit(X)
        dist, idx = snn.transform(X, knn=k, radius=radius)
    
        if include_self:
            idx = idx[:, :k]
            dist = dist[:, :k]
        else:
            idx = idx[:, 1:]
            dist = dist[:, 1:]
        keep = ( np.isfinite(dist) & (idx >= 0) & (idx < N))
        if knn_qt is not None and knn_qt < 1:
            dvalid = dist[keep & (dist > 0)]
            dcut = np.quantile(dvalid, knn_qt)
            keep &= (dist <= dcut)

        rows, pos = np.nonzero(keep)
        neighbor_ids = ids[idx[rows, pos]]
        label_valid = ( (neighbor_ids >= 0) & (neighbor_ids < J) )
        rows = rows[label_valid]
        neighbor_ids = neighbor_ids[label_valid]
        A = coo_matrix(
            ( np.ones(rows.size, dtype=dtype), (rows, neighbor_ids)),
            shape=(N, J),
            dtype=dtype,
        ).tocsr()
        A.sum_duplicates()
        A = A.toarray()

    if fuzzy_unique:
        unique_idx = np.asarray(unique_idx, dtype=bool).reshape(-1)

        kidx = valid & (np.count_nonzero(A, axis=1) == 1)

        rows = np.flatnonzero(kidx)
        kidx[rows] &= A[rows, ids[rows]] > 0

        fuzzy_idx = np.zeros(N, dtype=bool)
        fuzzy_idx[kidx] = unique_idx[ids[kidx]]
        fuzzy_col = np.isin(La, u) & (~unique_idx)
        A[np.ix_(fuzzy_idx, fuzzy_col)] += fuzzy_unique
        
    if (knn > 0) or (radius is not None):
        A += fuzzy_global


    if normal in [None, 'none', False]:
        return A, La

    rs = A.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1

    if normal in ['row', 'prob', True]:
        return (A / rs, La)

    if normal in ['temp', 'temperature']:
        A = A / rs
        A = np.where(A > 0, A ** (1.0 / float(tau)), 0)
        rs = A.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1
        A = A / rs
        return (A, La)

    raise ValueError("normal must be one of {'row', 'temp', 'none'}")

def label_onehotm(
    L,
    La=None,
    unique_idx=None,
    label_type='single',

    knn=0,
    radius=None,
    knn_method='sknn',
    include_self=True,
    knn_qt=0.995,

    fuzzy_global=0,
    fuzzy_unique=0,

    fuzzy_label=None,
    fuzzy_prior=0,

    X=None,
    normal='row',

    dtype=np.float32,
    n_jobs=-1,
):
    if label_type not in ['single', 'multi_label']:
        raise ValueError(
            "label_type must be one of {'single', 'multi_label'}."
        )

    if fuzzy_label is not None:
        assert type(fuzzy_label) in [bool, int, float, str, None]

    #TODO fuzzy_label : list

    if label_type == 'single':
        L = np.asarray(L, dtype=object).reshape(-1)
        N = L.size

        fuzzy_idx = (
            np.zeros(N, dtype=bool)
            if fuzzy_label is None
            else L == fuzzy_label
        )

        Lainf = np.asarray(
            list(dict.fromkeys(L[~fuzzy_idx])),
            dtype=object,
        )

        La = (
            Lainf
            if La is None
            else np.asarray(La, dtype=object).reshape(-1)
        )

        if len(set(La)) != len(La):
            raise ValueError("La contains duplicated labels.")

        if not set(Lainf).issubset(set(La)):
            unq = list(set(Lainf) - set(La))
            raise ValueError(f"L contains labels not found in La :{unq}.")

        if fuzzy_label is not None and fuzzy_label in set(La):
            raise ValueError(f"fuzzy_label '{fuzzy_label}' must not be included in La.")

        J = La.size
        label_map = {v: j for j, v in enumerate(La)}

        ids = np.fromiter(
            (label_map.get(v, -1) for v in L),
            dtype=np.int64,
            count=N,
        )
        valid = ids >= 0

        H = np.zeros((N, J), dtype=dtype)
        H[np.flatnonzero(valid), ids[valid]] = 1

    else:
        L = [v if isinstance(v, list) else [v] for v in L]
        N = len(L)

        lens = np.fromiter(map(len, L), dtype=np.int64, count=N)
        rows = np.repeat(np.arange(N, dtype=np.int64), lens)

        labels = np.fromiter(
            chain.from_iterable(L),
            dtype=object,
            count=int(lens.sum()),
        )

        fuzzy_idx = (
            np.zeros(N, dtype=bool)
            if fuzzy_label is None
            else np.fromiter(
                (fuzzy_label in v for v in L),
                dtype=bool,
                count=N,
            )
        )

        known = ~np.repeat(fuzzy_idx, lens)
        Lainf = np.asarray(
            list(dict.fromkeys(labels[known])),
            dtype=object,
        )

        La = (
            Lainf
            if La is None
            else np.asarray(La, dtype=object).reshape(-1)
        )

        if len(set(La)) != len(La):
            raise ValueError("La contains duplicated labels.")

        if not set(Lainf).issubset(set(La)):
            unq = list(set(Lainf) - set(La))
            raise ValueError(f"L contains labels not found in La: {unq}.")

        if fuzzy_label is not None and fuzzy_label in set(La):
            raise ValueError(f"fuzzy_label '{fuzzy_label}' must not be included in La.")

        J = La.size
        label_map = {v: j for j, v in enumerate(La)}

        cols = np.fromiter(
            (label_map.get(v, -1) for v in labels),
            dtype=np.int64,
            count=labels.size,
        )
        valid = cols >= 0

        H = np.zeros((N, J), dtype=dtype)
        H[rows[valid], cols[valid]] = 1

        H[fuzzy_idx] = 0

    knn = int(knn)
    if knn <= 0:
        A = H.copy()
        # A[fuzzy_idx] = 1.0/float(J)
    else:
        if X is None:
            raise ValueError("X must be provided when knn > 0.")
        X = np.asarray(X)
        if X.shape[0] != N:
            raise ValueError( f"X.shape[0]={X.shape[0]} != len(L)={N}." )

        k = min(knn + 1, N)
        snn = Neighbors(method=knn_method)
        snn.fit(X)
        dist, idx = snn.transform(X, knn=k, radius=radius) # radius TODO

        dist = np.asarray(dist)
        idx = np.asarray(idx)
        if include_self:
            dist, idx = dist[:, :k], idx[:, :k]
        else:
            dist, idx = dist[:, 1:], idx[:, 1:]


        keep = ( np.isfinite(dist) & (idx >= 0) & (idx < N) )
        if knn_qt is not None and knn_qt < 1:
            dvalid = dist[keep & (dist > 0)]
            keep &= dist <= np.quantile(dvalid, knn_qt)

        rows, pos = np.nonzero(keep)
        cols = idx[rows, pos].astype(np.int64, copy=False)

        W = coo_matrix(
            ( np.ones(rows.size, dtype=dtype), (rows, cols), ),
            shape=(N, N),
            dtype=dtype,
        ).tocsr()

        W.sum_duplicates()
        if W.nnz:
            W.data[:] = 1
        A = np.asarray(W @ H, dtype=dtype)

    # ============================================================
    # Unique-label fuzzy compatibility
    # ============================================================
    if fuzzy_unique:
        if unique_idx is None:
            raise ValueError( "unique_idx is required when fuzzy_unique > 0." )

        unique_idx = np.asarray(unique_idx, dtype=bool).reshape(-1)
        if unique_idx.size != J:
            raise ValueError(f"len(unique_idx)={unique_idx.size} != len(La)={J}." )

        rows = np.flatnonzero(
            np.count_nonzero(H, axis=1) == 1
        )
        hids = H[rows].argmax(axis=1)

        keep = (
            unique_idx[hids]
            & (np.count_nonzero(A[rows], axis=1) == 1)
            & (A[rows, hids] > 0)
        )
        rows = rows[keep]
        fcols = H.any(axis=0) & (~unique_idx)
        if rows.size and fcols.any():
            A[np.ix_(rows, fcols)] += float(fuzzy_unique)
    
        # if J:
        #     hsum = H.sum(1)
        #     hids = H.argmax(1)
        #     asum = A.sum(1)

        #     frows = (
        #         (hsum == 1)
        #         & unique_idx[hids]
        #         & (asum > 0)
        #         & (A[np.arange(N), hids] == asum)
        #     )
        #     fcols = H.any(0) & (~unique_idx)

        #     if frows.any() and fcols.any():
        #         A[np.ix_(frows, fcols)] += float(fuzzy_unique)

    if fuzzy_prior and fuzzy_idx.any() and J:
        A[fuzzy_idx] += float(fuzzy_prior)

    if fuzzy_global:
        A += float(fuzzy_global)

    # ============================================================
    # Normalization
    # ============================================================
    if normal in [None, False, 'none', 'count']:
        return A, La

    if normal in [True, 'row']:
        rs = A.sum(1, keepdims=True)
        A = np.divide(
            A,
            rs,
            out=np.zeros_like(A),
            where=rs > 0,
        )
        return A, La

    raise ValueError("normal must be one of {'count', 'row'}.")