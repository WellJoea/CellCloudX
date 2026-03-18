import numpy as np
from scipy.spatial import cKDTree
import scipy.sparse as ssp
from sklearn.utils import check_random_state


import numpy as np
import pandas as pd

from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy.sparse as ssp
import scipy as sci

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import scanpy as sc

import sys
sys.path.append('/gpfs/home/user19/JupyterCode/')
from CellCloudX_v126 import cellcloudx as cc

def pair_gl(X, Y, Xf, Yf, xlabels=None, knn=1, radius=None, ylabels=None, alpha=0.5, use_mnn=False, temp=[1,1], CI=1.0):
    from scipy import stats
    from sklearn.metrics.cluster import adjusted_rand_score
    from sklearn.metrics import adjusted_mutual_info_score
    from sklearn.metrics import normalized_mutual_info_score

    N, M = X.shape[0], Y.shape[0]
    X = np.array(X, dtype=np.float64)
    Y = np.array(Y, dtype=np.float64)

    src2, dst2, dist2 =coord_edges(X, Y, knn = knn, radius=radius,  method='sknn')
    if use_mnn:     
        src1, dst1, dist1 = coord_edges(Y, X, knn = knn, radius=radius, method='sknn')
        set1 = set(zip(src2, dst2))
        set2 = set(zip(dst1, src1))
        set3 = set1 & set2
        src2, dst2 = np.array(list(set3)).astype(np.int64).T
        dist2 = np.linalg.norm(X[src2] - Y[dst2], axis=1) ** 2

    if not CI is None:
        K  = int(len(dist2)*CI)
        kidx = np.argpartition(dist2, K,)[:K]
        # kidx2 = np.argsort(dist2)[:K]
        src2, dst2, dist2 = src2[kidx], dst2[kidx], dist2[kidx]

    sx = centerlize(np.concatenate([X, Y], axis=0))
    sf = center_normalize(np.concatenate([Xf, Yf], axis=0))
    # sf = normalize(np.concatenate([Xf, Yf], axis=0))
    
    spdist = np.linalg.norm(sx[:N][src2] - sx[N:][dst2], axis=1) ** 2
    ftdist = np.linalg.norm(sf[:N][src2] - sf[N:][dst2], axis=1) ** 2
    ftdist1 = (sf[:N][src2] * sf[N:][dst2]).sum(1)

    sigmap = np.median(spdist[spdist>0]) * temp[0]
    sigmaf = np.median(ftdist[ftdist>0]) * temp[1]

    sigmap = 1 * temp[0]
    sigmaf = 1 * temp[1]

    # fig, axs = plt.subplots(1,2, figsize=(7,3))
    # axs[0].hist(spdist/sigmap,bins=100)
    # axs[1].hist(ftdist/sigmaf,bins=100)
    # plt.show()

    score1 = spdist/sigmap + ftdist/sigmaf
    score2 = np.exp(- spdist/sigmap - ftdist/sigmaf ) ** 0.5
    score3 = (np.exp(- spdist/sigmap) + np.exp(- ftdist/sigmaf))/2
    score4 =  (spdist * ftdist) **0.5
    score5 =  ((spdist**alpha )* (ftdist**(1-alpha))) **0.5

    # df = pd.DataFrame(np.c_[score1, score2, score3, score4, score5],
    #                      columns=['score1', 'score2', 'score3', 'score4', 'score5'])

    # if not (xlabels is None or ylabels is None):
    #     types = xlabels.columns.tolist()
    #     methods = [['ARI','AMI', 'NMI'], [adjusted_rand_score, adjusted_mutual_info_score, normalized_mutual_info_score]]
    #     scores = []
    #     for itl in types:
    #         for inm, imd in zip(*methods):
    #             iscore = imd(xlabels[itl].values[src2], ylabels[itl].values[dst2])
    #             scores.append([f'{inm}({itl})', iscore])
    #     scores = pd.DataFrame(scores, columns=['types', 'score'])
    #     return df, scores
    # else:
    #     return df


def gl_score(X, Xf, labels=None, knn=15, radius=None, alpha=0.5, use_mnn=False, temp=[1,1], CI=0.98):
    from scipy import stats
    from sklearn.metrics.cluster import adjusted_rand_score
    from sklearn.metrics import adjusted_mutual_info_score
    from sklearn.metrics import normalized_mutual_info_score

    N = X.shape[0]
    X = np.array(X, dtype=np.float64)

    src2, dst2, dist2 =coord_edges(X, X, knn = knn, radius=radius,  method='sknn', keep_loops=False,)

    if not CI is None:
        K  = int(len(dist2)*CI)
        kidx = np.argpartition(dist2, K,)[:K]
        # kidx2 = np.argsort(dist2)[:K]
        src2, dst2, dist2 = src2[kidx], dst2[kidx], dist2[kidx]

    sx = centerlize(X)
    sf = center_normalize(Xf)

    spdist = np.linalg.norm(sx[src2] - sx[dst2], axis=1) ** 2
    ftdist = np.linalg.norm(sf[src2] - sf[dst2], axis=1) ** 2
    ftdist1 = (sf[src2] * sf[dst2]).sum(1)

    sigmap = np.median(spdist[spdist>0]) * temp[0]
    sigmaf = np.median(ftdist[ftdist>0]) * temp[1]
    sigmap = 1 * temp[0]
    sigmaf = 1 * temp[1]

    # fig, axs = plt.subplots(1,2, figsize=(7,3))
    # axs[0].hist(spdist/sigmap,bins=100)
    # axs[1].hist(ftdist/sigmaf,bins=100)
    # plt.show()

    score1 = spdist/sigmap + ftdist/sigmaf
    score2 = np.exp(- spdist/sigmap - ftdist/sigmaf ) ** 0.5
    score3 = (np.exp(- spdist/sigmap) + np.exp(- ftdist/sigmaf))/2
    score4 =  (spdist * ftdist) **0.5
    score5 =  np.exp(-score4)
    score6 =  spdist * score5/score5.sum()

    return [ score1, score2, score3, score4, score5, score6]

def coord_edges(coordx, coordy=None,
                knn=50,
                radius=None,
                
                max_neighbor = int(1e4),
                method='sknn' ,
                keep_loops= True,
                n_jobs = -1):
    if coordy is None:
        coordy = coordx
    
    cknn = cc.tl.Neighbors( method=method ,metric='euclidean', n_jobs=n_jobs)
    cknn.fit(coordx, radius_max= None,max_neighbor=max_neighbor)
    distances, indices = cknn.transform(coordy, knn=knn, radius = radius)

    src = np.concatenate(indices, axis=0).astype(np.int64)
    dst = np.repeat(np.arange(len(indices)), list(map(len, indices))).astype(np.int64)
    dist = np.concatenate(distances, axis=0)

    if (coordy is None) and (not keep_loops):
        mask = src != dst
        src = src[mask]
        dst = dst[mask]
        dist = dist[mask]
    print(f'mean edges: {dist.shape[0]/coordy.shape[0]}')
    return [src, dst, dist]

def center_normalize(X):
    if ssp.issparse(X): 
        X = X.toarray()

    X = X.copy()
    X -= X.mean(axis=0, keepdims=True)
    l2x = np.linalg.norm(X, ord=None, axis=1, keepdims=True)
    l2x[l2x == 0] = 1
    return X/l2x

def centerlize(X, Xm=None, Xs=None):
    if ssp.issparse(X): 
        X = X.toarray()
    X = X.copy()
    N,D = X.shape
    Xm = np.mean(X, 0)

    X -= Xm
    Xs = np.sqrt(np.sum(np.square(X))/(N*D/2)) if Xs is None else Xs
    X /= Xs

    return X




def robust_median_sq(vals, eps=1e-12):
    vals = np.asarray(vals).ravel()
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return 1.0
    m = np.median(vals)
    return float(max(m, eps))

def pairwise_sq_norm(a, b):
    # 返回 ||a-b||^2 的矩阵 (n,m)；注意可能消耗内存，需用于子采样规模
    a2 = (a*a).sum(1, keepdims=True)     # (n,1)
    b2 = (b*b).sum(1, keepdims=True).T   # (1,m)
    return a2 + b2 - 2.0 * (a @ b.T)

# -----------------------------
# 1) 构图：仅连接相邻切片
# -----------------------------
def build_cross_slice_edges(Xs, Fs=None, k_edge=8, lambda_f=1.0, use_feature_in_weight=True, random_state=0):
    """
    返回：
      - src, dst: 全局点索引的一维数组
      - w: 边权
      - dx2_list, df2_list: 边的平方距离（用于尺度估计）
      - idx_map: 每张切片在全局的起止索引
    """
    rng = check_random_state(random_state)
    offsets = np.cumsum([0] + [len(X) for X in Xs])
    N_total = int(offsets[-1])

    all_src, all_dst, all_w = [], [], []
    dx2_all, df2_all = [], []

    for s in range(len(Xs)-1):
        Xa, Xb = Xs[s], Xs[s+1]
        Na, Nb = len(Xa), len(Xb)

        tree = cKDTree(Xb)
        # 每个 a 点找 k 个 b 邻居
        dists, nn = tree.query(Xa, k=min(k_edge, max(1, Nb)))
        if dists.ndim == 1:  # k=1 时统一形状
            dists = dists[:, None]; nn = nn[:, None]

        # 空间平方距离
        dx2 = dists**2  # (Na, k)
        dx2_all.append(dx2.ravel())

        if Fs is not None and use_feature_in_weight:
            Fa, Fb = Fs[s], Fs[s+1]
            Fb_nn = Fb[nn]   # (Na, k, d)
            Fa_rep = Fa[:, None, :]  # (Na, 1, d)
            df2 = ((Fa_rep - Fb_nn)**2).sum(axis=2)  # (Na, k)
            df2_all.append(df2.ravel())
        else:
            df2 = np.zeros_like(dx2)

        # 稳健尺度
        sigx2 = robust_median_sq(dx2)
        sigf2 = robust_median_sq(df2) if Fs is not None and use_feature_in_weight else 1.0

        # 权重（位置+特征）
        w = np.exp(-dx2 / sigx2 - lambda_f * df2 / sigf2)

        # 组装全局索引
        src = offsets[s] + np.repeat(np.arange(Na), nn.shape[1])
        dst = offsets[s+1] + nn.ravel()
        all_src.append(src); all_dst.append(dst); all_w.append(w.ravel())

    src = np.concatenate(all_src) if all_src else np.array([], dtype=int)
    dst = np.concatenate(all_dst) if all_dst else np.array([], dtype=int)
    w   = np.concatenate(all_w)   if all_w   else np.array([], dtype=float)
    dx2_list = np.concatenate(dx2_all) if dx2_all else np.array([], dtype=float)
    df2_list = np.concatenate(df2_all) if df2_all else np.array([], dtype=float)

    return src, dst, w, dx2_list, df2_list, offsets

# -----------------------------
# 2) 拉普拉斯平滑分（位置/特征）
# -----------------------------
def laplacian_smoothness_scores(Xs, Fs, src, dst, w, dx2_list, df2_list):
    if len(w) == 0:
        return 0.0, 0.0

    # 坐标拼接
    X_all = np.vstack(Xs)
    # 位置能量（直接按边求，无需显式 L）
    dx2_edge = ((X_all[src] - X_all[dst])**2).sum(axis=1)  # (E,)
    # 平均边能量 & 指数映射
    sigx2 = robust_median_sq(dx2_list)
    Ex_bar = (w * dx2_edge).sum() / (w.sum() + 1e-12)
    S_lap_x = float(np.exp(-Ex_bar / sigx2))

    # 特征能量
    if Fs is None or (df2_list is None) or (len(df2_list) == 0):
        return S_lap_x, None
    F_all = np.vstack(Fs)
    df2_edge = ((F_all[src] - F_all[dst])**2).sum(axis=1)
    sigf2 = robust_median_sq(df2_list)
    Ef_bar = (w * df2_edge).sum() / (w.sum() + 1e-12)
    S_lap_f = float(np.exp(-Ef_bar / (sigf2 + 1e-12)))
    return S_lap_x, S_lap_f

# -----------------------------
# 3) 对称Chamfer分
# -----------------------------
def chamfer_score(Xs):
    if len(Xs) < 2:
        return 1.0
    cds = []
    for s in range(len(Xs)-1):
        Xa, Xb = Xs[s], Xs[s+1]
        ta = cKDTree(Xa); tb = cKDTree(Xb)
        da, _ = tb.query(Xa, k=1)
        db, _ = ta.query(Xb, k=1)
        CD = (da**2).mean() + (db**2).mean()
        cds.append(CD)
    sigx2 = robust_median_sq(cds)
    return float(np.exp(-(np.mean(cds)) / (sigx2 + 1e-12)))

# -----------------------------
# 4) 二阶差分弯曲分
# -----------------------------
def bending_score(Xs, src, dst, w, offsets):
    if len(Xs) < 3 or len(w) == 0:
        return None
    X_all = np.vstack(Xs)
    # 为每个 s 的点找去往 s+1 的“最强边” & 去往 s-1 的“最强边”
    S = len(Xs)
    best_plus = {}   # i -> (j_plus, w_plus)
    best_minus = {}  # i -> (j_minus, w_minus)

    for s in range(S-1):
        start_a, end_a = offsets[s], offsets[s+1]
        mask = (src >= start_a) & (src < end_a)  # 边从 s -> s+1
        sub_src, sub_dst, sub_w = src[mask], dst[mask], w[mask]
        # 对于每个 src 点，保留最大权重的 dst
        # 用 argsort 分桶实现
        order = np.lexsort((-sub_w, sub_src))  # 先按 src 分组，再按权重降序
        ss = sub_src[order]; dd = sub_dst[order]; ww = sub_w[order]
        # 保留每个 src 的第一条
        uniq, first_idx = np.unique(ss, return_index=True)
        pick = order[first_idx]
        for i, j, wi in zip(sub_src[pick], sub_dst[pick], sub_w[pick]):
            best_plus[i] = (j, wi)

    # 反向边（s+1 <- s）得到 “去往 s-1 的最强边”
    for s in range(S-1):
        start_b, end_b = offsets[s+1], offsets[s+2] if s+2 <= S-1 else offsets[s+1]
        mask = (src >= start_b) & (src < end_b)  # 这些是 (s+1) -> (s+2)，但我们要 (s) <- (s+1)
        # 更简单：直接扫描原边表，挑选 dst 属于 s 的边
    for s in range(1, S):
        start_prev, end_prev = offsets[s-1], offsets[s]
        mask = (dst >= start_prev) & (dst < end_prev)  # 边指向 s-1
        sub_src, sub_dst, sub_w = src[mask], dst[mask], w[mask]
        order = np.lexsort((-sub_w, sub_src))
        ss = sub_src[order]; dd = sub_dst[order]; ww = sub_w[order]
        uniq, first_idx = np.unique(ss, return_index=True)
        pick = order[first_idx]
        for i, j, wi in zip(sub_src[pick], sub_dst[pick], sub_w[pick]):
            # 这表示 i(在 s) 的“去往 s-1 的强边”为 j
            best_minus[i] = (j, wi)

    # 计算二阶差分（中间切片上的点）
    vals, weights = [], []
    for s in range(1, S-1):
        start, end = offsets[s], offsets[s+1]
        for i in range(start, end):
            if i in best_plus and i in best_minus:
                j_plus, w_plus = best_plus[i]
                j_minus, w_minus = best_minus[i]
                d2 = X_all[j_plus] - 2*X_all[i] + X_all[j_minus]
                w_i = min(w_plus, w_minus)
                vals.append((d2*d2).sum())
                weights.append(w_i)
    if len(vals) == 0:
        return None
    vals = np.asarray(vals); weights = np.asarray(weights)
    E_bend = (weights * vals).sum() / (weights.sum() + 1e-12)
    # 尺度用跨切片边的空间中位数
    sigx2 = robust_median_sq(vals)
    return float(np.exp(-E_bend / (sigx2 + 1e-12)))

# -----------------------------
# 5) 法向连续性分
# -----------------------------
def estimate_normals_pca(X, knn=16):
    if len(X) < 3:
        return np.zeros((len(X), 3))
    tree = cKDTree(X)
    _, nn = tree.query(X, k=min(knn, max(3, len(X))))
    if nn.ndim == 1: nn = nn[:, None]
    normals = np.zeros_like(X)
    for i in range(len(X)):
        pts = X[nn[i]]
        C = np.cov((pts - pts.mean(0)).T)
        w, V = np.linalg.eigh(C)
        normals[i] = V[:, np.argmin(w)]
    # 方向不唯一，后续取绝对余弦即可
    return normals

def normal_continuity_score(Xs, src, dst, w, offsets, knn_in_slice=16):
    if len(Xs) < 2 or len(w) == 0:
        return None
    normals = [estimate_normals_pca(X, knn=knn_in_slice) for X in Xs]
    N_offsets = np.cumsum([0] + [len(X) for X in Xs])
    N_all = sum(len(X) for X in Xs)
    n_all = np.zeros((N_all, 3))
    for s, n in enumerate(normals):
        n_all[offsets[s]:offsets[s+1]] = n
    cosv = np.abs((n_all[src] * n_all[dst]).sum(axis=1))  # |cos|
    # 只统计跨切片边
    if len(cosv) == 0:
        return None
    En = (1.0 - cosv)
    eta = 0.1
    return float(np.exp(-(w * En).sum() / ((w.sum() + 1e-12) * eta)))

# -----------------------------
# 6) Sinkhorn-OT & 循环一致性
# -----------------------------
def sinkhorn_uniform(a, b, C, eps=0.05, n_iter=200, tol=1e-9):
    # a,b 均匀也可直接传标量 1/n；此处传向量方便扩展
    K = np.exp(-C / max(eps, 1e-8))  # (n,m)
    u = np.ones_like(a)
    v = np.ones_like(b)
    for _ in range(n_iter):
        Ku = K.T @ u
        v_new = b / (Ku + 1e-12)
        Kv = K @ v_new
        u_new = a / (Kv + 1e-12)
        # 收敛判据
        if np.max(np.abs(u_new - u)) < tol and np.max(np.abs(v_new - v)) < tol:
            u, v = u_new, v_new
            break
        u, v = u_new, v_new
    P = (u[:, None] * K) * v[None, :]
    return P

def ot_and_cycle_scores(Xs, Fs, lambda_f=1.0, eps=0.05, ot_max_points=2000, random_state=0):
    if len(Xs) < 2:
        return None, None
    rng = check_random_state(random_state)
    ot_costs = []
    P_list = []   # 保存耦合以做循环一致性
    subs_idx = [] # 每对切片用到的子采样索引，便于 s->s+2 的直接 OT

    for s in range(len(Xs)-1):
        Xa, Xb = Xs[s], Xs[s+1]
        Fa = Fs[s] if Fs is not None else None
        Fb = Fs[s+1] if Fs is not None else None

        # 子采样（均匀）
        na = min(len(Xa), ot_max_points)
        nb = min(len(Xb), ot_max_points)
        ia = rng.choice(len(Xa), size=na, replace=False) if len(Xa) > na else np.arange(len(Xa))
        ib = rng.choice(len(Xb), size=nb, replace=False) if len(Xb) > nb else np.arange(len(Xb))

        Xa_, Xb_ = Xa[ia], Xb[ib]
        # 代价（含特征）
        Cx = pairwise_sq_norm(Xa_, Xb_)
        sigx2 = robust_median_sq(Cx)
        C = Cx / (sigx2 + 1e-12)
        if Fa is not None:
            Fa_, Fb_ = Fa[ia], Fb[ib]
            # 特征维可能不同，使用余弦距离或 CCA/线性投影可扩展；这里先统一维，或退化为欧氏在交集特征
            if Fa_.shape[1] == Fb_.shape[1]:
                Cf = pairwise_sq_norm(Fa_, Fb_)
                sigf2 = robust_median_sq(Cf)
                C = C + lambda_f * (Cf / (sigf2 + 1e-12))
        a = np.ones(len(Xa_)) / len(Xa_)
        b = np.ones(len(Xb_)) / len(Xb_)
        P = sinkhorn_uniform(a, b, C, eps=eps)
        cost = float((P * C).sum())
        ot_costs.append(cost)
        P_list.append(P)
        subs_idx.append((ia, ib))

    S_ot = float(np.exp(-np.mean(ot_costs))) if len(ot_costs) else None

    # 循环一致性：需要 s->s+2 的直接 OT 与合成 OT
    if len(Xs) >= 3:
        cyc_vals = []
        for s in range(len(Xs)-2):
            # 合成：P12 * P23
            P12 = P_list[s]
            P23 = P_list[s+1]
            P13_hat = P12 @ P23  # (na, nc)
            # 直接：在同一子采样上做 s vs s+2
            i1, _    = subs_idx[s]
            _,  i2   = subs_idx[s+1]
            X1 = Xs[s][i1]; X3 = Xs[s+2][i2]
            Cx = pairwise_sq_norm(X1, X3)
            sigx2 = robust_median_sq(Cx)
            C = Cx / (sigx2 + 1e-12)
            if Fs is not None and Fs[s].shape[1] == Fs[s+2].shape[1]:
                F1 = Fs[s][i1]; F3 = Fs[s+2][i2]
                Cf = pairwise_sq_norm(F1, F3)
                sigf2 = robust_median_sq(Cf)
                C = C + (Cf / (sigf2 + 1e-12))
            a = np.ones(len(X1)) / len(X1)
            b = np.ones(len(X3)) / len(X3)
            P13 = sinkhorn_uniform(a, b, C, eps=0.05)
            # L1 差
            diff = np.abs(P13 - P13_hat).sum()
            cyc_vals.append(diff)
        S_cycle = float(np.exp(-np.mean(cyc_vals))) if len(cyc_vals) else None
    else:
        S_cycle = None

    return S_ot, S_cycle

# -----------------------------
# 7) 总评估器
# -----------------------------
def evaluate_zstack_scores(
    Xs, Fs=None,
    k_edge=8, lambda_f=1.0,
    normal_knn=16,
    ot_eps=0.05, ot_max_points=2000,
    weights=None, random_state=0
):
    # 构图（相邻切片）
    src, dst, w, dx2_list, df2_list, offsets = build_cross_slice_edges(
        Xs, Fs, k_edge=k_edge, lambda_f=lambda_f, use_feature_in_weight=True, random_state=random_state
    )

    # Laplacian（位置/特征）
    S_lap_x, S_lap_f = laplacian_smoothness_scores(Xs, Fs, src, dst, w, dx2_list, df2_list)

    # Chamfer
    S_cd = chamfer_score(Xs)

    # 弯曲
    S_bend = bending_score(Xs, src, dst, w, offsets)

    # 法向
    S_normal = normal_continuity_score(Xs, src, dst, w, offsets, knn_in_slice=normal_knn)

    # OT & cycle
    S_ot, S_cycle = ot_and_cycle_scores(Xs, Fs, lambda_f=lambda_f, eps=ot_eps,
                                        ot_max_points=ot_max_points, random_state=random_state)

    # 组合
    parts = {
        'lap_pos': S_lap_x,
        'lap_feat': S_lap_f,
        'chamfer': S_cd,
        'bending': S_bend,
        'normal': S_normal,
        'ot': S_ot,
        'cycle': S_cycle
    }
    # 默认权重
    default_w = {
        'lap_pos': 1.0, 'lap_feat': 1.0, 'chamfer': 1.0,
        'bending': 1.0, 'normal': 0.5, 'ot': 1.0, 'cycle': 0.5
    }
    if weights is None:
        weights = default_w
    # 几何平均（跳过 None）
    num, den = 0.0, 0.0
    log_sum = 0.0
    for k, v in parts.items():
        if v is not None and np.isfinite(v) and v > 0:
            a = weights.get(k, 0.0)
            if a > 0:
                log_sum += a * np.log(v)
                den += a
    S_total = float(np.exp(log_sum / max(den, 1e-12))) if den > 0 else None

    return {'parts': parts, 'S_total': S_total}

import numpy as np
from time import time

# =========================
# 合成数据：基础 + 平滑几何与表型
# =========================
def make_synthetic_stack(S=6, N=4000, dz=1.0, feat_dim=8, random_state=42):
    """
    生成“对齐良好”的 z-stack:
      - 每个切片的 XY 来源于多高斯团，沿切片序号 s 加入小幅平滑的位姿扰动
      - z = s * dz + 小幅空间warp
      - 特征由若干 RBF 基函数生成，跨切片只做微小平滑相位变化 + 弱噪声
    """
    rng = np.random.default_rng(random_state)

    # 2D 混合高斯作为基础图形（3个团）
    centers = np.array([[0.0, 0.0], [1.2, 0.0], [0.6, 0.9]])
    comp = rng.integers(0, len(centers), size=N)
    XY0 = centers[comp] + 0.25 * rng.normal(size=(N, 2))
    # 标准化方便尺度稳定
    XY0 = (XY0 - XY0.mean(0)) / XY0.std(0)

    # RBF anchors（全局共享）
    K = feat_dim
    anchors = rng.normal(size=(K, 2)) * 1.0
    sigma2 = 0.8

    Xs, Fs = [], []
    for s in range(S):
        # 平滑的小位姿（随 s 正弦变化）
        rot = 0.12 * np.sin(2*np.pi*s / S)
        c, s_ = np.cos(rot), np.sin(rot)
        R = np.array([[c, -s_], [s_, c]])
        shift = np.array([0.20*np.sin(2*np.pi*s / S),
                          0.15*np.cos(2*np.pi*s / S)])

        XY = XY0 @ R.T + shift
        z = np.full((N, 1), s * dz) + 0.05 * np.sin(2*np.pi*XY[:, :1])
        X = np.hstack([XY, z]).astype(np.float64)
        Xs.append(X)

        # 特征：RBF + 轻微相位项 + 小噪声
        F = np.exp(-((XY[:, None, :] - anchors[None, :, :])**2).sum(-1) / (2*sigma2))
        F += 0.05 * np.sin(0.5*s + XY @ np.array([[0.2, -0.3]]).T)
        F += 0.03 * rng.normal(size=F.shape)
        Fs.append(F.astype(np.float64))

    return Xs, Fs

# =========================
# 退化：模拟配准差（旋转/平移/层抖动/特征扰动）
# =========================
def degrade_stack(Xs, Fs, trans_sigma=0.4, rot_sigma=0.4, z_jitter=0.2,
                  feat_noise=0.2, channel_shuffle=True, random_state=7):
    rng = np.random.default_rng(random_state)
    Xd, Fd = [], []
    for X, F in zip(Xs, Fs):
        # 随机旋转/平移（更差的对齐）
        theta = rng.normal(scale=rot_sigma)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        t = rng.normal(scale=trans_sigma, size=2)

        XY = X[:, :2] @ R.T + t
        z = X[:, 2:3] + z_jitter * rng.normal(size=(len(X), 1))
        X_bad = np.hstack([XY, z])

        F_bad = F.copy()
        if channel_shuffle:
            perm = rng.permutation(F.shape[1])
            F_bad = F_bad[:, perm]
        F_bad = F_bad + feat_noise * rng.normal(size=F.shape)

        Xd.append(X_bad.astype(np.float64))
        Fd.append(F_bad.astype(np.float64))
    return Xd, Fd

# =========================
# 打印工具
# =========================
def pretty_print_scores(name, out):
    print(f"\n=== {name} ===")
    parts = out['parts']
    for k in ['lap_pos', 'lap_feat', 'chamfer', 'bending', 'normal', 'ot', 'cycle']:
        v = parts.get(k, None)
        if v is not None:
            print(f"{k:>9s}: {v:8.5f}")
        else:
            print(f"{k:>9s}: None")
    print(f"  S_total: {out['S_total']:.6f}" if out['S_total'] is not None else "  S_total: None")

# =========================
# 主测试：好 vs 差 + 退化扫描
# =========================
def main():
    # 1) 合成“好”的 z-stack
    S, N, Df = 6, 4000, 8
    Xs_good, Fs_good = make_synthetic_stack(S=S, N=N, dz=1.0, feat_dim=Df, random_state=42)

    # 2) 评估“好”
    t0 = time()
    good = evaluate_zstack_scores(
        Xs_good, Fs_good,
        k_edge=6, lambda_f=0.7,
        normal_knn=12,
        ot_eps=0.10, ot_max_points=1000,  # 控制 OT 规模，跑得更快
        random_state=0
    )
    t1 = time()
    pretty_print_scores("GOOD (well-registered)", good)
    print(f"Time: {t1 - t0:.2f}s")

    # 3) 生成“差”的版本
    Xs_bad, Fs_bad = degrade_stack(
        Xs_good, Fs_good,
        trans_sigma=0.6, rot_sigma=0.6, z_jitter=0.30,
        feat_noise=0.25, channel_shuffle=True, random_state=24
    )

    # 4) 评估“差”
    t2 = time()
    bad = evaluate_zstack_scores(
        Xs_bad, Fs_bad,
        k_edge=6, lambda_f=0.7,
        normal_knn=12,
        ot_eps=0.10, ot_max_points=1000,
        random_state=1
    )
    t3 = time()
    pretty_print_scores("BAD (mis-registered)", bad)
    print(f"Time: {t3 - t2:.2f}s")

    # 5) 简单 sanity check：总分应当 好 > 差
    if good['S_total'] is not None and bad['S_total'] is not None:
        print("\nSanity check (S_total GOOD > BAD):",
              "PASS" if good['S_total'] > bad['S_total'] else "FAIL")

    # 6) 退化强度扫描（查看趋势）
    print("\n--- Degradation sweep ---")
    levels = [0.0, 0.2, 0.4, 0.6]
    results = []
    for lv in levels:
        Xs_deg, Fs_deg = degrade_stack(
            Xs_good, Fs_good,
            trans_sigma=0.2 + lv,
            rot_sigma=0.2 + lv,
            z_jitter=0.10 + 0.3*lv,
            feat_noise=0.10 + 0.2*lv,
            channel_shuffle=True, random_state=123 + int(lv*10)
        )
        out = evaluate_zstack_scores(
            Xs_deg, Fs_deg,
            k_edge=6, lambda_f=0.7,
            normal_knn=12,
            ot_eps=0.10, ot_max_points=1000,
            random_state=2
        )
        results.append(out['S_total'])
        print(f"level={lv:.1f} -> S_total={out['S_total']:.6f}")

    # 趋势提示
    if all(r is not None for r in results):
        monotone = all(results[i] >= results[i+1] - 1e-6 for i in range(len(results)-1))
        print("Monotonic descending w.r.t. degradation:", "YES" if monotone else "NO")
