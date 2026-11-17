"""Utilities for GWR slice weights and label-compatibility matrices."""

import numpy as np
import torch as th
from .neighbors_ensemble import Neighbors
from scipy.sparse import coo_matrix
from itertools import chain


def gwr_omega(
    L,
    Z=1.0,
    k=None,
    method="gaussian",
    scale=1.0,
    base=1.8,
    skew=0.0,
    df=3.0,
    shape=1.0,
):
    """
    Generate a row-normalized directional weight matrix between slices.

    Parameters
    ----------
    L : int
        Number of slices.

    Z : float, array-like or torch.Tensor, default=1.0
        Slice-axis geometry:

        - scalar: constant spacing;
        - length L-1: consecutive slice spacings;
        - length L: absolute slice positions;
        - shape (L, L): precomputed nonnegative distances.

    k : int or None, default=None
        Keep at most k slices on each side. None keeps all slices.

    method : str, default='gaussian'
        Distance kernel, where r is the direction-adjusted distance:

        gaussian
            exp(-r^2 / 2)

        exponential
            exp(-r)

        inverse
            1 / (1 + r)

        student
            (1 + r^2 / df)^(-(df + 1) / 2).
            Smaller df gives a heavier tail.

        gamma
            Gamma survival function Q(shape, r). It is monotonic from
            one to zero. shape=1 is equivalent to exponential.

        weibull
            exp(-r^shape). shape>1 suppresses distant slices faster;
            shape<1 produces a longer tail.

        log
            base^(1-r)

        adjacent
            1 / (1+r), restricted to immediately adjacent slices.

    scale : float, default=1.0
        Global bandwidth multiplier. The base bandwidth is

            median(adjacent slice distance) * scale.

    base : float, default=1.8
        Base used by method='log'. It must be greater than one.

    skew : float, default=0.0
        Directional log-bandwidth asymmetry. For omega[i, j]:

        - j < i: left/earlier slice;
        - j > i: right/later slice.

        The two bandwidths are

            h_left  = h * exp(-skew)
            h_right = h * exp( skew)

        Therefore:

        - skew > 0: left decays faster, right decays slower;
        - skew < 0: left decays slower, right decays faster;
        - skew = 0: symmetric kernel.

    df : float, default=3.0
        Degrees of freedom for method='student'. Must be positive.

    shape : float, default=1.0
        Shape parameter for method='gamma' or method='weibull'.
        Must be positive.

    Returns
    -------
    omega : torch.Tensor, shape (L, L)
        Row-normalized nonnegative slice weights with zero diagonal.
        omega[i, j] is the contribution of slice j to slice i.
    """
    L = int(L)
    if L < 2:
        raise ValueError("L must be at least 2.")

    Z = th.as_tensor(Z)
    if not Z.is_floating_point():
        Z = Z.float()

    index = th.arange(L, device=Z.device)
    offset = index[None, :] - index[:, None]  # j - i

    if Z.ndim == 0:
        z = index.to(Z.dtype) * Z
        distance = (z[:, None] - z[None, :]).abs()

    elif Z.ndim == 1:
        if Z.numel() == L - 1:
            z = th.cat((Z.new_zeros(1), Z.cumsum(0)))
        elif Z.numel() == L:
            z = Z
        else:
            raise ValueError("One-dimensional Z must have length L-1 or L.")

        distance = (z[:, None] - z[None, :]).abs()

    elif Z.ndim == 2 and Z.shape == (L, L):
        distance = Z

    else:
        raise ValueError(
            "Z must be a spacing, L-1 spacings, L positions, "
            "or an L x L distance matrix."
        )

    if not th.isfinite(distance).all() or (distance < 0).any():
        raise ValueError("Distances must be finite and nonnegative.")

    adjacent = th.cat((
        distance.diagonal(offset=1),
        distance.diagonal(offset=-1),
    ))
    adjacent = adjacent[adjacent > 0]

    if adjacent.numel() == 0:
        raise ValueError("Adjacent slice distances must contain positive values.")

    scale = th.as_tensor(
        scale,
        dtype=distance.dtype,
        device=distance.device,
    )
    if scale.numel() != 1 or not th.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be a positive scalar.")

    skew = th.as_tensor(
        skew,
        dtype=distance.dtype,
        device=distance.device,
    )
    if skew.numel() != 1 or not th.isfinite(skew):
        raise ValueError("skew must be a finite scalar.")

    bandwidth = adjacent.median() * scale

    # Positive skew:
    #   j < i -> exp(-skew), narrower and steeper
    #   j > i -> exp(+skew), wider and smoother
    direction = offset.sign().to(distance.dtype)
    directional_bandwidth = bandwidth * th.exp(skew * direction)
    ratio = distance / directional_bandwidth

    method = method.lower().replace("-", "_")

    if method == "gaussian":
        omega = th.exp(-0.5 * ratio.square())

    elif method == "exponential":
        omega = th.exp(-ratio)

    elif method == "inverse":
        omega = 1.0 / (1.0 + ratio)

    elif method in {"student", "student_t", "t"}:
        df = th.as_tensor(
            df,
            dtype=distance.dtype,
            device=distance.device,
        )
        if df.numel() != 1 or not th.isfinite(df) or df <= 0:
            raise ValueError("df must be a positive scalar.")

        omega = (
            1.0 + ratio.square() / df
        ).pow(-0.5 * (df + 1.0))

    elif method == "gamma":
        shape = th.as_tensor(
            shape,
            dtype=distance.dtype,
            device=distance.device,
        )
        if shape.numel() != 1 or not th.isfinite(shape) or shape <= 0:
            raise ValueError("shape must be a positive scalar.")

        # Regularized upper incomplete gamma Q(shape, ratio).
        # Unlike the Gamma PDF, this is maximal at zero and monotonic.
        omega = th.special.gammaincc(shape, ratio)

    elif method == "weibull":
        shape = th.as_tensor(
            shape,
            dtype=distance.dtype,
            device=distance.device,
        )
        if shape.numel() != 1 or not th.isfinite(shape) or shape <= 0:
            raise ValueError("shape must be a positive scalar.")

        omega = th.exp(-ratio.pow(shape))

    elif method == "log":
        base = th.as_tensor(
            base,
            dtype=distance.dtype,
            device=distance.device,
        )
        if base.numel() != 1 or not th.isfinite(base) or base <= 1:
            raise ValueError("base must be greater than one.")

        omega = base.pow(1.0 - ratio)

    elif method == "adjacent":
        mask = offset.abs() == 1
        omega = mask / (1.0 + ratio)

    else:
        raise ValueError(
            "method must be one of: gaussian, exponential, inverse, "
            "student, gamma, weibull, log or adjacent."
        )

    if k is not None:
        k = int(k)
        if k < 1:
            raise ValueError("k must be positive.")
        omega *= offset.abs() <= k

    omega.fill_diagonal_(0.0)

    row_sum = omega.sum(1, keepdim=True)
    if not th.isfinite(omega).all() or (row_sum <= 0).any():
        raise ValueError(
            "Some rows have no valid neighbors; increase k or scale."
        )

    return omega / row_sum


def label_onehots(
    Ls,
    Xs=None,
    La=None,
    knn=0,
    radius=None,
    knn_method="sknn",
    include_self=True,
    knn_qt=0.995,
    fuzzy_global=0,
    fuzzy_unique=0,
    fuzzy_label=None,
    fuzzy_prior=0,
    normal="row",
    dtype=np.float32,
    n_jobs=-1,
):
    """Build label-compatibility matrices for multiple slices.

    Parameters
    ----------
    Ls : sequence of array-like
        Labels for ``K`` slices. Each ``Ls[k]`` contains ``N_k`` samples.
        Each sample may hold a scalar label or a list of labels; a scalar
        item is treated as a one-element list, independently of ``knn``.
    Xs : sequence of array-like or None, default=None
        Coordinates or features for the ``K`` slices. ``Xs[k]`` must have
        shape ``(N_k, D)`` and is required when the corresponding ``knn`` is
        positive. ``None`` is expanded to one ``None`` value per slice.
    La : array-like or None, default=None
        Shared label vocabulary and column order. If omitted, labels are
        collected from all slices in first-appearance order, excluding
        samples containing ``fuzzy_label`` when a marker is provided.
    knn : int or sequence of int, default=0
        Number of non-self neighbors requested for each slice. A scalar is
        shared across slices; a sequence must have length ``K``. If positive,
        labels are aggregated over the neighbor graph. With
        ``include_self=True``, the query retains the sample itself in addition
        to as many as ``knn`` neighboring samples.
    radius : float, sequence or None, default=None
        Optional neighbor-search radius. A scalar is shared across slices; a
        sequence must have length ``K``. It is passed directly to
        ``Neighbors.transform``.
    knn_method : str, default='sknn'
        Neighbor-search backend passed to ``Neighbors(method=knn_method)``.
    include_self : bool, default=True
        Whether to retain the first returned neighbor, assumed to be the query
        sample itself. If ``False``, that first neighbor is removed.
    knn_qt : float or None, default=0.995
        Quantile cutoff applied to positive, finite neighbor distances. It is
        active when not ``None`` and smaller than 1. Distances above the pooled
        per-slice quantile are removed.
    fuzzy_global : float or sequence of float, default=0
        Additive compatibility assigned to every sample-label entry. A scalar
        is shared across slices; a list or tuple must have length ``K``.
    fuzzy_unique : float or sequence of float, default=0
        Additive compatibility used for labels appearing in at most one slice.
        For an unambiguous sample of such a label, this value is added to
        compatible non-unique labels present in the same slice. A scalar is
        shared across slices; a list or tuple must have length ``K``.
    fuzzy_label : scalar or None, default=None
        Marker for an unknown or ambiguous label. It is excluded from ``La``;
        its direct one-hot row is zero and can receive mass through
        ``fuzzy_prior`` and/or ``fuzzy_global``.
    fuzzy_prior : float or sequence of float, default=0
        Additive prior assigned across all columns of rows marked by
        ``fuzzy_label``. A scalar is shared across slices; a list or tuple must
        have length ``K``.
    normal : {True, False, 'row', 'count', 'none', None}, default='row'
        Output normalization. ``True`` or ``'row'`` divides each nonzero row by
        its sum. ``False``, ``None``, ``'none'`` or ``'count'`` returns raw
        compatibility counts and additive weights.
    dtype : numpy dtype, default=np.float32
        Data type of the returned compatibility matrices.
    n_jobs : int, default=-1
        Number of workers forwarded to ``label_onehot`` and retained for the
        neighbor interface.

    Returns
    -------
    As : list of numpy.ndarray
        ``K`` dense compatibility matrices. ``As[k]`` has shape
        ``(N_k, J)``, where ``J = len(La)``.
    La : numpy.ndarray
        Shared one-dimensional label vocabulary of length ``J`` in column
        order.

    Raises
    ------
    ValueError
        If a slice-wise argument has the wrong length, labels are
        inconsistent with ``La``, required coordinates are missing, or
        ``normal`` is unsupported.
    """
    Ls = list(Ls)
    K = len(Ls)

    Ls = [
        [v if isinstance(v, list) else [v] for v in L]
        for L in Ls
    ]
    Linfs = [
        np.asarray(
            list(chain.from_iterable(
                v for v in L
                if fuzzy_label is None or fuzzy_label not in v
            )),
            dtype=object,
        )
        for L in Ls
    ]

    if La is None:
        La = np.asarray(
            list(dict.fromkeys(chain.from_iterable(Linfs))),
            dtype=object,
        )
    else:
        La = np.asarray(La, dtype=object).reshape(-1)

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
    if isinstance(fuzzy_prior, (list, tuple)):
        assert len(fuzzy_prior) == K
    else:
        fuzzy_prior = [fuzzy_prior] * K

    if radius is None:
        radii = [None] * K
    elif np.isscalar(radius):
        radii = [radius] * K
    elif len(radius) == K:
        radii = list(radius)
    else:
        raise ValueError("radius must have length K.")

    if Xs is None:
        Xs = [None] * K
    elif len(Xs) != K:
        raise ValueError("Xs must have length K.")

    label_sets = [set(L) for L in Linfs]
    present = np.asarray(
        [[v in label_set for v in La] for label_set in label_sets],
        dtype=bool,
    )
    uu_ls = present & (present.sum(axis=0, keepdims=True) <= 1)

    As = []
    for k in range(K):
        A0, _ = label_onehot(
            Ls[k],
            La=La,
            unique_idx=uu_ls[k],
            knn=knns[k],
            radius=radii[k],
            knn_method=knn_method,
            include_self=include_self,
            knn_qt=knn_qt,
            fuzzy_global=fuzzy_global[k],
            fuzzy_unique=fuzzy_unique[k],
            fuzzy_label=fuzzy_label,
            fuzzy_prior=fuzzy_prior[k],
            X=Xs[k],
            normal=normal,
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
    knn_method="sknn",
    include_self=True,
    knn_qt=0.995,
    fuzzy_global=0,
    fuzzy_unique=0,
    fuzzy_label=None,
    fuzzy_prior=0,
    X=None,
    normal="row",
    dtype=np.float32,
    n_jobs=-1,
):
    """Build a dense one-hot or neighborhood label-compatibility matrix.

    Parameters
    ----------
    L : array-like
        Labels for ``N`` samples. Each sample may hold a scalar label or a
        list of labels; a scalar item is treated as a one-element list,
        independently of ``knn``. Repeated labels within one sample still
        produce a binary entry.
    La : array-like or None, default=None
        Label vocabulary defining the ``J`` output columns. It must contain
        every known label in ``L`` and must not contain duplicates or
        ``fuzzy_label``. If omitted, known labels are collected in
        first-appearance order.
    unique_idx : array-like of bool or None, default=None
        Boolean mask of length ``J`` identifying labels unique to one slice.
        It is required only when ``fuzzy_unique`` is nonzero. In normal use it
        is computed by ``label_onehots`` across all slices.
    knn : int, default=0
        Number of non-self neighbors to request. If ``knn <= 0``, return the
        direct binary matrix before fuzzy additions. If positive, aggregate
        neighbor labels through a sparse adjacency matrix. With
        ``include_self=True``, as many as ``knn + 1`` entries are retained per
        row, including the query sample.
    radius : float or None, default=None
        Optional radius passed directly to ``Neighbors.transform``.
    knn_method : str, default='sknn'
        Neighbor-search backend passed to ``Neighbors``.
    include_self : bool, default=True
        Whether the first returned neighbor, assumed to be the query sample,
        is retained. If ``False``, the first neighbor is discarded.
    knn_qt : float or None, default=0.995
        Pooled quantile cutoff for positive, finite neighbor distances. The
        cutoff is applied only when the value is not ``None`` and is below 1.
    fuzzy_global : float, default=0
        Additive compatibility assigned to every entry after KNN aggregation.
    fuzzy_unique : float, default=0
        Additive compatibility for an unambiguous sample whose label is marked
        by ``unique_idx``. The value is added to non-unique label columns that
        are present in this slice.
    fuzzy_label : bool, int, float, str or None, default=None
        Marker for unknown or ambiguous samples. It is excluded from ``La``.
        A single-label sample equal to this value, or a multi-label sample
        containing it, receives a zero direct one-hot row.
    fuzzy_prior : float, default=0
        Additive value assigned to every label column of rows marked by
        ``fuzzy_label``.
    X : array-like or None, default=None
        Coordinates or features of shape ``(N, D)`` used to construct the
        neighbor graph. Required when ``knn > 0``.
    normal : {True, False, 'row', 'count', 'none', None}, default='row'
        ``True`` or ``'row'`` normalizes each nonzero row to sum to one.
        ``False``, ``None``, ``'none'`` or ``'count'`` preserves raw counts and
        additive weights. Zero rows remain zero under row normalization.
    dtype : numpy dtype, default=np.float32
        Data type of the returned matrix.
    n_jobs : int, default=-1
        Reserved worker-count argument for API compatibility. The current
        function does not use it directly.

    Returns
    -------
    A : numpy.ndarray
        Dense compatibility matrix of shape ``(N, J)``. Depending on
        ``normal``, rows contain raw neighbor counts/additive weights or
        normalized label probabilities.
    La : numpy.ndarray
        One-dimensional label vocabulary of length ``J`` in output-column
        order.

    Raises
    ------
    ValueError
        If ``La`` is duplicated or incomplete,
        ``fuzzy_label`` is included in ``La``, ``unique_idx`` is unavailable or
        malformed, coordinates are unavailable or misaligned for KNN, or
        ``normal`` is unsupported.
    """
    if fuzzy_label is not None:
        assert type(fuzzy_label) in [bool, int, float, str, None]

    # TODO fuzzy_label: list

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
        raise ValueError(
            f"fuzzy_label '{fuzzy_label}' must not be included in La."
        )

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
        # A[fuzzy_idx] = 1.0 / float(J)
    else:
        if X is None:
            raise ValueError("X must be provided when knn > 0.")
        X = np.asarray(X)
        if X.shape[0] != N:
            raise ValueError(f"X.shape[0]={X.shape[0]} != len(L)={N}.")

        k = min(knn + 1, N)
        snn = Neighbors(method=knn_method)
        snn.fit(X)
        dist, idx = snn.transform(X, knn=k, radius=radius)  # radius TODO

        dist = np.asarray(dist)
        idx = np.asarray(idx)
        if include_self:
            dist, idx = dist[:, :k], idx[:, :k]
        else:
            dist, idx = dist[:, 1:], idx[:, 1:]

        keep = np.isfinite(dist) & (idx >= 0) & (idx < N)
        if knn_qt is not None and knn_qt < 1:
            dvalid = dist[keep & (dist > 0)]
            if dvalid.size:
                keep &= dist <= np.quantile(dvalid, knn_qt)

        rows, pos = np.nonzero(keep)
        cols = idx[rows, pos].astype(np.int64, copy=False)

        W = coo_matrix(
            (np.ones(rows.size, dtype=dtype), (rows, cols)),
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
            raise ValueError(
                "unique_idx is required when fuzzy_unique > 0."
            )

        unique_idx = np.asarray(unique_idx, dtype=bool).reshape(-1)
        if unique_idx.size != J:
            raise ValueError(
                f"len(unique_idx)={unique_idx.size} != len(La)={J}."
            )

        rows = np.flatnonzero(
            np.count_nonzero(H, axis=1) == 1
        )
        hids = np.nonzero(H[rows])[1]

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
    if normal in [None, False, "none", "count"]:
        return A, La

    if normal in [True, "row"]:
        rs = A.sum(1, keepdims=True)
        A = np.divide(
            A,
            rs,
            out=np.zeros_like(A),
            where=rs > 0,
        )
        return A, La

    raise ValueError("normal must be one of {'count', 'row'}.")
