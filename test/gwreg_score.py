import numpy as np
from scipy.spatial import cKDTree
from scipy import sparse
import scipy.sparse as ssp
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy import sparse
import scipy.sparse as ssp
from tqdm import tqdm
import matplotlib.pyplot as plt

def slice_knn(Xs, knn=8, ks=1, radius=None,kd_method='annoy',CI=0.99, disable=True):
    offsets = np.cumsum([0] + [len(X) for X in Xs])
    N_total = int(offsets[-1])

    all_src, all_dst, all_d = [], [], []
    L = len(Xs)
    pbar = tqdm(range(L-1), total=L-1, desc=f'knn {L}', disable=disable)

    for s in pbar:
        for j in range(s+1, min(s+ks+1, L)):
            Xa, Xb = Xs[s], Xs[j]
            Na, Nb = len(Xa), len(Xb)

            src1, dst1, dist1 = coord_edges(Xa, Xb, knn = knn, radius=radius, method=kd_method)
            src2, dst2, dist2 = coord_edges(Xb, Xa, knn = knn, radius=radius, method=kd_method)

            all_src.append(offsets[s] + src1)
            all_src.append(offsets[j] + src2)
            all_dst.append(offsets[j] + dst1)
            all_dst.append(offsets[s] + dst2)
            all_d.append(dist1)
            all_d.append(dist2)

    src = np.concatenate(all_src) 
    dst = np.concatenate(all_dst)
    dist = np.concatenate(all_d)

    K  = int(len(dist)*CI)
    kidx = np.argpartition(dist, K,)[:K]
    src = src[kidx]
    dst = dst[kidx]
    
    w   = np.ones(len(src))
    W = sparse.csr_matrix((w, (dst, src)), shape=(N_total, N_total))
    return W

def GMD(Xs, Fs, labels=None, W=None, alpha=0.5, knn=8, 
        normalx= True, normalf=True,
        temp=[1,1], agg='mean', use_xy=True,
        radius=None,kd_method='annoy',  CI=0.97):
    from scipy import stats
    from sklearn.metrics.cluster import adjusted_rand_score
    from sklearn.metrics import adjusted_mutual_info_score
    from sklearn.metrics import normalized_mutual_info_score


    if W is None:
        W = slice_knn(Xs, knn=knn, radius=radius, kd_method=kd_method, CI=CI)
    src, dst = W.nonzero()

    sx = np.concatenate(Xs, axis=0)
    sf = np.concatenate(Fs, axis=0)

    if use_xy:
        sx = sx[:, :2]
    if normalx:
        sx = scaler(sx)


    if normalf:
        sf = normalize(sf)
    
    spdist = np.linalg.norm(sx[src] - sx[dst], axis=1) ** 2
    ftdist = np.linalg.norm(sf[src] - sf[dst], axis=1) ** 2
    # ftdist1 = (sf[src] * sf[dst]).sum(1)

    sigmap = np.median(spdist[spdist>0]) * temp[0]
    sigmaf = np.median(ftdist[ftdist>0]) * temp[1]

    sigmap = 1 * temp[0]
    sigmaf = 1 * temp[1]


    score1 = spdist/sigmap + ftdist/sigmaf
    score2 = np.exp(- spdist/sigmap - ftdist/sigmaf ) ** 0.5
    score3 = (np.exp(- spdist/sigmap) + np.exp(- ftdist/sigmaf))/2
    score4 =  (spdist * ftdist) **0.5
    score5 =  ((spdist**alpha )* (ftdist**(1-alpha))) **0.5

    df = pd.DataFrame(np.c_[score1, score2, score3, score4, score5],
                         columns=['score1', 'score2', 'score3', 'score4', 'score5'])

    if agg == 'mean':
        mdf = df.mean(0)
        mdf['score4'] = df['score4'].sum(0) /(sx.shape[0]**0.5)
        mdf['score5'] = df['score5'].sum(0) /(sx.shape[0]**0.5)
    elif agg == 'sum':
        mdf = df.sum(0)

    if not (labels is None):
        types = labels.columns.tolist()
        methods = [['ARI','AMI', 'NMI'], [adjusted_rand_score, adjusted_mutual_info_score, normalized_mutual_info_score]]
        methods = [['ARI'], [adjusted_rand_score]]
        scores = []
        for itl in types:
            for inm, imd in zip(*methods):
                iscore = imd(labels[itl].values[src], labels[itl].values[dst])
                scores.append([f'{inm}({itl})', iscore])
        scores = pd.DataFrame(scores, columns=['types', 'score'])
        mdf = pd.concat([mdf, scores.set_index('types').iloc[:,0]])
        for itl in types:
            iscore = labels[itl].values[src] == labels[itl].values[dst]
            isoore = iscore.sum() / (~iscore).sum()
            mdf[f'rs({itl})'] = isoore
    return mdf
    
def laplacian_smoothness_scores(Xs, Fs=None, W=None, knn=8, radius=None,kd_method='annoy',
                                bf = 10, 
                                normalx=True, normalf=True, 
                                sigx2=1,sigf2=1, lambdax=1):
    if W is None:
        W = slice_knn(Xs, knn=knn, radius=radius, kd_method=kd_method)
    src, dst = W.nonzero()

    Xa = np.vstack(Xs)
    if normalx:
        Xa = scaler(Xa) * bf
    dx2 = np.linalg.norm(Xa[src] - Xa[dst], axis=1) **2
    V1 = np.exp(-dx2 / sigx2)
    V2 = np.exp(-dx2)
  
    if Fs is not None:
        Fa = np.vstack(Fs)
        if normalf:
            Fa = normalize(Fa)
        df2 = np.linalg.norm(Fa[src] - Fa[dst], axis=1) ** 2
        V1 = np.exp(-dx2 / (sigx2*lambdax)  - df2 / sigf2)
        V2 = np.exp(- (dx2*df2) **0.5 )
        V3 = np.exp(-df2)
        V4 = ((Fa[src]*Fa[dst]).sum(1) + 1)/2
    
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 2, figsize=(5, 3))
    axs[0,0].hist(dx2, bins=100)
    axs[0,0].set_title('dx2')
    if Fs is not None:
        axs[0,1].hist(df2, bins=100)
        axs[0,1].set_title('df2')
    axs[1,0].hist(V1, bins=100)
    axs[1,0].set_title('V1')
    axs[1,1].hist(V2, bins=100)
    axs[1,1].set_title('V2')
    plt.show()

    W1 = sparse.csr_matrix((V1, (src, dst)), shape=W.shape)
    L1 = sparse.diags(np.array(W1.sum(1)).ravel()) - W1

    W2  = sparse.csr_matrix((V2, (src, dst)), shape=W.shape)
    L2 = sparse.diags(np.array(W2.sum(1)).ravel()) - W2

    W3 = sparse.csr_matrix((V3, (src, dst)), shape=W.shape)
    L3 = sparse.diags(np.array(W3.sum(1)).ravel()) - W3

    W4 = sparse.csr_matrix((V4, (src, dst)), shape=W.shape)
    L4 = sparse.diags(np.array(W4.sum(1)).ravel()) - W4

    Ex1 = (V1 * dx2).sum()/Xa.shape[0]
    Ex2 = (V2 * dx2).sum()/Xa.shape[0]
    Ex3 = np.trace(Xa.T @ (L1 @ Xa))/Xa.shape[0]
    Ex4 = np.trace(Xa.T @ (L2 @ Xa))/Xa.shape[0]
    Ex5 = np.trace(Xa.T @ (L3 @ Xa))/Xa.shape[0]
    Ex6 = np.trace(Xa.T @ (L4 @ Xa))/Xa.shape[0]
    Ex7 = Ex1/V1.sum()
    Ex8 = Ex2/V2.sum()

    if Fs is not None:
        Ef1 = (V1 * df2).sum()/Xa.shape[0]
        Ef2 = (V2 * df2).sum()/Xa.shape[0]
        Ef3 = np.trace(Fa.T @ (L1 @ Fa))/Xa.shape[0]
        Ef4 = np.trace(Fa.T @ (L2 @ Fa))/Xa.shape[0]
        Ef5 = np.trace(Fa.T @ (L3 @ Fa))/Xa.shape[0]    
        Ef6 = np.trace(Fa.T @ (L4 @ Fa))/Xa.shape[0]
        Ef7 = Ef1/V1.sum()
        Ef8 = Ef2/V2.sum()

    R = np.array([[Ex1, Ex2, Ex3, Ex4, Ex5, Ex6, Ex7, Ex8], [Ef1, Ef2, Ef3, Ef4, Ef5, Ef6, Ef7, Ef8]])
    print(R)
    return R

def gene_smoothness_scores(Xs, G, Fs= None, W=None, knn=8, radius=None,kd_method='annoy',
                                 CI=0.98,
                                sigx2=1,sigf2=1, lambdax=1):
    if W is None:
        W = slice_knn(Xs, knn=knn, radius=radius, kd_method=kd_method, CI=CI)
    src, dst = W.nonzero()

    Xa = np.vstack(Xs)
    dx2 = np.linalg.norm(Xa[src] - Xa[dst], axis=1) **2
    V1 = np.exp(-dx2 / sigx2)
    V2 = np.exp(-dx2)
  
    if Fs is not None:
        Fa = np.vstack(Fs)
        df2 = np.linalg.norm(Fa[src] - Fa[dst], axis=1) ** 2
        V1 = np.exp(-dx2 / (sigx2*lambdax)  - df2 / sigf2)
        V2 = np.exp(- (dx2*df2) **0.5 )

    W1 = sparse.csr_matrix((V1, (src, dst)), shape=W.shape)
    L1 = sparse.diags(np.array(W1.sum(1)).ravel()) - W1

    W2  = sparse.csr_matrix((V2, (src, dst)), shape=W.shape)
    L2 = sparse.diags(np.array(W2.sum(1)).ravel()) - W2

    Ex1 = (V1 * G[dst]).sum()/G.shape[0]
    Ex2 = (V2 * G[dst]).sum()/G.shape[0]
    Ex3 = np.trace(G[:,None].T @ (L1 @ G[:,None]))/G.shape[0]
    Ex4 = np.trace(G[:,None].T @ (L2 @ G[:,None]))/G.shape[0]
    Ex5 = np.trace(G[:,None].T @ (W @ G[:,None]))/G.shape[0]

    R = np.array([Ex1, Ex2, Ex3, Ex4, Ex5])
    return R

gwCOLORS = ['#b240ce', '#b6b51f', '#0780cf', '#765005', '#fa6d1d', 
          '#2f3ea8', '#da1f18', '#701866', '#f47a75', '#009db2', 
          '#024b51', '#0780cf', '#765005', '#6beffc', '#3b45dd', 
          '#ad94ec', '#00749d', '#6ed0a7', '#0e2c82', '#706c01', 
          '#9be4ff', '#d70000']

def gwplt_hist(dfs, figsize=(6, 10), y_lim = (0,100), ys_min = None, ys_max = None, 
             scores= [2,3,4,5,6],
             palette= gwCOLORS,
             linewidth=1, save=None):
    import seaborn as sns

    fig, axs = plt.subplots( len(scores), 1, figsize=figsize,
                            constrained_layout=False, sharex=False, sharey=False)
    for i, iscore in enumerate(scores):
        if len(scores) == 1:
            iax = axs
        else:
            iax = axs[i]
        idf = dfs.loc[:, iscore].copy()

        sns.barplot( dfs, y=dfs.index, x =iscore,  linewidth=linewidth, 
                    palette =palette[:dfs.shape[0]], orient= 'h',
                    ax=iax)
        iax.grid(False)
        iax.set_title(iscore)
        # legend = iax.get_legend()
        # legend.set_bbox_to_anchor((1.05, 0.5))
        # legend.set_loc('center left')  

    fig.tight_layout()
    if save:
        fig.savefig(f'{save}.gw.score.hist.pdf',transparent=True, format='pdf')
        dfs.to_csv(f'{save}.gw.score.hist.csv')
    plt.show()

def gwplt_hist1(dfs, figsize=(6, 10), y_lim = (0,100), ys_min = None, ys_max = None, 
             scores= [2,3,4,5,6],
             palette= gwCOLORS[:4],
             linewidth=1, save=None, **kargs):
    import seaborn as sns

    fig, axs = plt.subplots( len(scores), 1, figsize=figsize,
                            constrained_layout=False, sharex=False, sharey=False)
    for i, iscore in enumerate(scores):
        sns.barplot( dfs, y='Time', x =iscore, hue='Group', linewidth=linewidth, 
                    palette =palette, orient= 'h',
                    ax=axs[i], legend='auto', **kargs)
        axs[i].grid(False)
        axs[i].set_title(iscore)
        legend = axs[i].get_legend()
        legend.set_bbox_to_anchor((1.05, 0.5))
        legend.set_loc('center left')  

    fig.tight_layout()
    if save:
        fig.savefig(f'{save}.gw.score.hist.pdf',transparent=True, format='pdf')
        dfs.to_csv(f'{save}.gw.score_hu.hist.csv')
    plt.show()


def gwplt_hist2(dfs, groupyb='Time', y='Group', scores= [3,4,5],
             figsize=(6, 10),
             palette= gwCOLORS[:4], 
             sharex=True, sharey=False,
             linewidth=1, save=None, **kargs):
    import seaborn as sns
    groups = np.unique(dfs[groupyb])
    for iscore in scores:
        fig, axs = plt.subplots( len(groups), 1, figsize=figsize,
                                constrained_layout=False, sharex=sharex, sharey=sharey)
        for i, igroup in enumerate(groups):
            idf = dfs[ (dfs[groupyb]==igroup) ].copy()

            sns.barplot( idf, y=y, x =iscore,  linewidth=linewidth, 
                        palette =palette, orient= 'h',
                        ax=axs[i], legend='auto', **kargs)
            axs[i].grid(False)
            # axs[i].set_title(iscore)
            axs[i].set_ylabel(igroup)
            # legend = axs[i].get_legend()
            # legend.set_bbox_to_anchor((1.05, 0.5))
            # legend.set_loc('center left')  

        fig.tight_layout()
        if save:
            fig.savefig(f'{save}.gw.score_{iscore}.hist.pdf',transparent=True, format='pdf')
            # dfs.to_csv(f'{save}.gw.score_{iscore}.hist.csv')
        plt.show()

def gwplt_hist_cap(dfs, figsize=(4.5, 2), y_lim = (0,100),
             scores= ['score4'],
             COLOR = ['#e41a1c','#4daf4a','#009db2', 'violet', 'cyan', 'gold'],
             cap_ratio =None,
             cap =None,
             cap_space = 0.05,  d = 5, wspace=0.08,
             break_ratio =0.18,
             break_line=8,
             linewidth=1, save=None):
    import seaborn as sns
    palette = COLOR
    
    for i, iscore in enumerate(scores):
        idf = dfs.loc[:,iscore].copy()
        fig, (ax_left, ax_right) = plt.subplots(ncols=2, nrows=1, sharey=True, 
                                              figsize=figsize,
                                              width_ratios=[1-break_ratio, break_ratio],
                                              gridspec_kw={'wspace':wspace})
        sns.barplot( idf, linewidth=linewidth, 
                    palette =palette, orient= 'h',
                    ax=ax_left, legend=False)
        sns.barplot( idf,  linewidth=linewidth, 
                    palette =palette, orient= 'h',
                    ax=ax_right, legend=False)

        ax_left.grid(False)
        ax_right.grid(False)
        # ax_left.set_xlabel('')
        # ax_right.set_xlabel('')
        ax_left.set_title(iscore)

        L = idf.max()
        if cap is None:
            cap_ratio = cap_ratio or 0.75
            cap = L * cap_ratio
        else:
            capl = min(cap, L*0.98)
        capp = cap_space * L

        ax_right.set_xlim(left=capl+capp, right=L+capp)   # those limits are fake
        ax_left.set_xlim(0,capl)

        ax_left.spines.right.set_visible(False)
        ax_right.spines.left.set_visible(False)

        ax_left.yaxis.tick_left()
        ax_left.tick_params(axis='y', which='both', right=False, labelright=False)
        ax_right.yaxis.tick_right()
        ax_right.tick_params(axis='y', which='both', right=False, labelright=False)
        # sns.despine(ax=ax_left)
        # sns.despine(ax=ax_right, right=True)
        # # ax_bottom.legend_.remove()

        d  # proportion of vertical to horizontal extent of the slanted line
        kwargs = dict(marker=[(-1, -d), (1, d)], markersize=break_line,
                      linestyle="none", color='k', mec='k', mew=1, clip_on=False)
        ax_left.plot([1, 1], [0, 0], transform=ax_left.transAxes, **kwargs)
        ax_right.plot([0, 0], [0, 0], transform=ax_right.transAxes, **kwargs)
        ax_left.plot([1, 1], [1, 1], transform=ax_left.transAxes, **kwargs)
        ax_right.plot([0, 0], [1, 1], transform=ax_right.transAxes, **kwargs)

        # fig.tight_layout()
        if save:
            fig.savefig(f'{save}.{iscore}.hist.cap.pdf',transparent=True, format='pdf')
            dfs.to_csv(f'{save}.{iscore}.hist.cap.csv')
        plt.show()

def gwplt_hist_cap1(dfs, x='score4', y ='knn', hue='group', 
                   figsize=(4.5, 2), y_lim = (0,100),
             COLOR = ['#e41a1c','#4daf4a','#009db2', 'violet', 'cyan', 'gold'],
             cap_ratio =None,
             cap =None,
             cap_space = 0.05,  d = 5, wspace=0.08,
             break_ratio =0.18,
             break_line=8,
             linewidth=1, save=None,  **kargs):
    import seaborn as sns
    palette = COLOR

    fig, (ax_left, ax_right) = plt.subplots(ncols=2, nrows=1, sharey=True, 
                                          figsize=figsize,
                                          width_ratios=[1-break_ratio, break_ratio],
                                          gridspec_kw={'wspace':wspace})
    sns.barplot( dfs, 
                x=x, y =y, hue=hue,
                 linewidth=linewidth, 
                palette =palette, orient= 'h',
                ax=ax_left, legend=False, **kargs)
    sns.barplot( dfs,  x=x, y =y, hue=hue, 
                linewidth=linewidth, 
                palette =palette, orient= 'h',
                ax=ax_right, legend=False,  **kargs)

    ax_left.grid(False)
    ax_right.grid(False)
    # ax_left.set_xlabel('')
    # ax_right.set_xlabel('')
    ax_left.set_title(x)

    L = dfs[x].max()
    if cap is None:
        cap_ratio = cap_ratio or 0.75
        cap = L * cap_ratio
    else:
        capl = min(cap, L*0.98)
    capp = cap_space * L

    ax_right.set_xlim(left=capl+capp, right=L+capp)   # those limits are fake
    ax_left.set_xlim(0,capl)

    ax_left.spines.right.set_visible(False)
    ax_right.spines.left.set_visible(False)

    ax_left.yaxis.tick_left()
    ax_left.tick_params(axis='y', which='both', right=False, labelright=False)
    ax_right.yaxis.tick_right()
    ax_right.tick_params(axis='y', which='both', right=False, labelright=False)
    # sns.despine(ax=ax_left)
    # sns.despine(ax=ax_right, right=True)
    # # ax_bottom.legend_.remove()

    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=break_line,
                  linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    ax_left.plot([1, 1], [0, 0], transform=ax_left.transAxes, **kwargs)
    ax_right.plot([0, 0], [0, 0], transform=ax_right.transAxes, **kwargs)
    ax_left.plot([1, 1], [1, 1], transform=ax_left.transAxes, **kwargs)
    ax_right.plot([0, 0], [1, 1], transform=ax_right.transAxes, **kwargs)

    # fig.tight_layout()
    if save:
        fig.savefig(save,transparent=True, format='pdf')
    plt.show()


def pca_normals_curv(X, knn=8, radius=None,kd_method='annoy'):
    distances, indices = coord_edges(X, knn = knn, radius=radius, method=kd_method, return_array=True)
    nb = indices[:,1:]
    N = len(X)
    Nrm = np.zeros_like(X)
    Curv = np.zeros(N)
    for i in range(N):
        P = X[nb[i]] - X[nb[i]].mean(0, keepdims=True)
        C = (P.T @ P) / max(1, len(P)-1)
        w, V = np.linalg.eigh(C)   # ascending
        Nrm[i] = V[:,0]
        Curv[i] = w[0] / (w.sum() + 1e-12)
    # canonical orientation (optional)
    Nrm /= (np.linalg.norm(Nrm, axis=1, keepdims=True)+1e-12)
    return Nrm, Curv

def planarity_score(curv, tau=0.1):
    print('curv:', np.sum(curv), )
    med = float(np.median(curv))
    return float(np.exp(-med / max(tau,1e-9)))

def normal_tv_score(Nrm, A):
    A = A.tocoo()
    cosv = np.abs(np.sum(Nrm[A.row] * Nrm[A.col], axis=1))
    print('cosv:', np.sum(cosv))
    entv = float(np.mean(1.0 - cosv))
    return float(1.0 - entv)  # in [0,1]

def build_cross_slice_edges(Xs, Fs=None, knn=8, radius=None, lambda_x=1.0,
                            kd_method='sknn',
                             sigx2=1,sigf2=1, random_state=0):
    """
    返回：
      - src, dst: 全局点索引的一维数组
      - w: 边权
      - dx2_list, df2_list: 边的平方距离（用于尺度估计）
      - idx_map: 每张切片在全局的起止索引
    """
    offsets = np.cumsum([0] + [len(X) for X in Xs])
    N_total = int(offsets[-1])

    all_src, all_dst, all_w = [], [], []
    dx2_all, df2_all = [], []

    for s in range(len(Xs)-1):
        Xa, Xb = Xs[s], Xs[s+1]
        Fa, Fb = Fs[s], Fs[s+1]
        Na, Nb = len(Xa), len(Xb)

        src1, dst1, dist1 = coord_edges(Xa, Xb, knn = knn, radius=radius, method=kd_method)
        dst2, src2, dist2 = coord_edges(Xb, Xa, knn = knn, radius=radius, method=kd_method)

        src = np.concatenate([src1, src2])
        dst = np.concatenate([dst1, dst2])
        dx2 = np.linalg.norm(Xa[src] - Xb[dst], axis=1) **2
        df2 = np.linalg.norm(Fa[src] - Fb[dst], axis=1) ** 2

        print(1111,  np.mean(dx2), np.median(dx2),  np.mean(df2), np.median(df2), )
        a = -dx2 / (sigx2*lambda_x) - df2 / sigf2
        b = - (dx2 * df2)**0.5
        print(2222, np.mean(a), np.median(a), np.min(a), np.max(a), np.mean(b), np.median(b), np.min(b), np.max(b), )
        w = np.exp(-dx2 / (sigx2*lambda_x) - df2 / sigf2)
        w = np.exp(- (dx2 * df2)**0.5 )

        src += offsets[s]
        dst += offsets[s+1]
        all_src.append(src); all_dst.append(dst); all_w.append(w.ravel())
        dx2_all.append(dx2); df2_all.append(df2)

    src = np.concatenate(all_src) 
    dst = np.concatenate(all_dst)
    w   = np.concatenate(all_w)
    W = sparse.csr_matrix((w, (src, dst)), shape=(N_total, N_total))

    dx2_list = np.concatenate(dx2_all)
    df2_list = np.concatenate(df2_all)
    print(w.max(), w.min(), dx2_list.shape, df2_list.shape)

    return W, dx2_list, df2_list

def laplacian_dirichlet_score(X, F, W= None, sigma2=1.0, knn=8, radius=None,kd_method='annoy',):
    if W is None:
        src, dst, dist2 = coord_edges(X, knn = knn, radius=radius, method=kd_method)
        w  = np.ones(len(src))
        W = sparse.csr_matrix((w, (dst, src)), shape=(X.shape[0], X.shape[0]))
    src, dst = W.nonzero()
    V = np.exp( -((X[src] - X[dst])**2).sum(axis=1)/sigma2)
    V1 = np.exp( -((F[src] - F[dst])**2).sum(axis=1) )
    V2 = ((F[src] * F[dst]).sum(1) + 1)/2


    d = np.array(W.sum(1)).ravel()
    L = sparse.diags(d) - W
    Ex = np.trace(F.T @ (L @ F))

    W1 = sparse.csr_matrix((V, (src, dst)), shape=W.shape)
    d1 = np.array(W1.sum(1)).ravel()
    L1 = sparse.diags(d1) - W1
    Ex1 = np.trace(X.T @ (L1 @ X))
    print(Ex)
    return Ex

    W1 = sparse.csr_matrix((V1, (src, dst)), shape=W.shape)
    d1 = np.array(W1.sum(1)).ravel()
    L1 = sparse.diags(d1) - W1
    Ex1 = np.trace(X.T @ (L1 @ X))

    W2 = sparse.csr_matrix((V2, (src, dst)), shape=W.shape)
    d2 = np.array(W2.sum(1)).ravel()
    L2 = sparse.diags(d2) - W2
    Ex2 = np.trace(X.T @ (L2 @ X))

    print(Ex, Ex1, Ex2)
    return Ex, Ex1, Ex2

def laplacian_dirichlet_score(X, F, W= None, sigma2=1.0, knn=8, radius=None,kd_method='annoy',):
    if W is None:
        src, dst, dist2 = coord_edges(X, knn = knn, radius=radius, method=kd_method)
        w  = np.ones(len(src))
        W = sparse.csr_matrix((w, (dst, src)), shape=(X.shape[0], X.shape[0]))
    src, dst = W.nonzero()
    V = np.exp( -((X[src] - X[dst])**2).sum(axis=1)/sigma2)
    W = sparse.csr_matrix((V, (src, dst)), shape=W.shape)
    d = np.array(W.sum(1)).ravel()
    L = sparse.diags(d) - W
    Ex = np.trace(F.T @ (L @ F))
    print(Ex)
    return Ex


def LSS(Xs, Fs, W, dx2_list, df2_list):

    src, dst = W.nonzero()
    w = np.array(W[src, dst]).ravel()

    d = np.array(W.sum(1)).ravel()
    L = sparse.diags(d) - W
    # 坐标拼接
    X_all = np.vstack(Xs)
    # 位置能量（直接按边求，无需显式 L）
    dx2_edge = ((X_all[src] - X_all[dst])**2).sum(axis=1)  # (E,)

    Ex = (w * dx2_edge).sum() 
    Ex1 = np.trace(X_all.T @ (L @ X_all))
    print(Ex,Ex/w.sum(), Ex1)

    # Ex_bar = (w * dx2_edge).sum() / (w.sum() + 1e-12)
    # S_lap_x = float(np.exp(-Ex_bar / sigx2))


    # F_all = np.vstack(Fs)
    # df2_edge = ((F_all[src] - F_all[dst])**2).sum(axis=1)
    # sigf2 = robust_median_sq(df2_list)
    # Ef_bar = (w * df2_edge).sum() / (w.sum() + 1e-12)
    # S_lap_f = float(np.exp(-Ef_bar / (sigf2 + 1e-12)))
    # return S_lap_x, S_lap_f

def coord_edges(coordx, coordy=None,
                knn=50,
                radius=None,
                
                max_neighbor = int(1e4),
                method='sknn' ,
                keep_loops= True,
                return_array = False,
                n_jobs = -1):

    import sys
    sys.path.append('/gpfs/home/user19/JupyterCode/')
    from CellCloudX_v126 import cellcloudx as cc

    if coordy is None:
        coordy = coordx
    
    cknn = cc.tl.Neighbors( method=method ,metric='euclidean', n_jobs=n_jobs)
    cknn.fit(coordx, radius_max= None,max_neighbor=max_neighbor)
    distances, indices = cknn.transform(coordy, knn=knn, radius = radius)

    if return_array:
        return distances, indices

    src = np.concatenate(indices, axis=0).astype(np.int64)
    dst = np.repeat(np.arange(len(indices)), list(map(len, indices))).astype(np.int64)
    dist = np.concatenate(distances, axis=0)

    if (coordy is None) and (not keep_loops):
        mask = src != dst
        src = src[mask]
        dst = dst[mask]
        dist = dist[mask]
    # print(f'mean edges: {dist.shape[0]/coordy.shape[0]}')
    return [src, dst, dist]


# ---------- graph utils ----------
def knn_graph(X, k=16):
    tree = cKDTree(X)
    d, nb = tree.query(X, k=k+1)  # include self
    nb = nb[:,1:]
    src = np.repeat(np.arange(len(X)), k)
    dst = nb.reshape(-1)
    # symmetrize
    A = sparse.coo_array((np.ones_like(src), (src, dst)), shape=(len(X), len(X)))
    A = A + A.T
    A = A.tocsr()
    A.data[:] = 1.0
    return A

def gaussian_weights(X, A, F= None,  s2=None):
    A = A.tocoo()
    diff = X[A.row] - X[A.col]
    d2 = np.sum(diff*diff, axis=1)

    if s2 is None:
        s2 = np.median(d2) + 1e-12
    print(1111111, np.mean(d2), np.median(d2))
    w = np.exp(-d2 / s2)

    if F is not None:
        F2 = np.linalg.norm(F[A.row] - F[A.col], axis=1)**2
        # w = np.exp(-d2 / s2 + F2)
        print(11111, (d2*F2).sum() )
        w = np.exp(-d2 / s2 *F2) **0.5
    W = sparse.coo_array((w, (A.row, A.col)), shape=A.shape).tocsr()
    return W, s2



def laplacian_energy_score(X, W,  c=1.0):
    d = np.array(W.sum(1)).ravel()
    L = sparse.diags(d) - W
    E = float(np.trace(X.T @ (L @ X))) 
    # print(E/len(X),  float(np.exp(- E / len(X))) )
    # S = float(np.exp(- E / (c*len(X)*s2 + 1e-12)))
    N,D = X.shape
    S = float(np.exp(- E / (N*D/2)) ) 
    return S

def feature_smoothness_score(F, W, tau=0.1):
    if F is None: return np.nan
    d = np.array(W.sum(1)).ravel()
    L = sparse.diags(d) - W
    E = float(np.trace(F.T @ (L @ F))) 
    N,D = F.shape
    # varF = float(np.mean((F - F.mean(0))**2)) + 1e-12
    # print(E, E/F.shape[0], varF)
    return float(np.exp(- E / (N*D/2)))



def mean_curvature_proxy_score(X, L, tau=0.1):
    # node-wise (Lx)_i magnitude^2 median
    Lx = np.stack([(L @ X[:,j]) for j in range(3)], axis=1)
    R2 = np.sum(Lx*Lx, axis=1)
    med = float(np.median(R2))
    return float(np.exp(- med / max(tau,1e-9)))


def spectral_highfreq_ratio_score(X, L, t=0.1, m=5, tau=0.2):
    I = sparse.eye(L.shape[0], format='csr')
    H = (I - t*L).tocsr()   # one-step heat kernel approx
    Xsm = X.copy()
    for _ in range(m):
        Xsm = H @ Xsm
    num = float(np.sum((X - Xsm)**2))
    den = float(np.sum((X - X.mean(0))**2))
    R = num / den
    print(R, num , den)
    return float(np.exp(- R / max(tau,1e-9)))

def box_counting_dimension_score(X, bins_list=(16,24,32,48,64), tau=0.5):
    # voxel occupancy across multiple resolutions
    mins = X.min(0) - 1e-9
    maxs = X.max(0) + 1e-9
    rng = (maxs - mins)
    eps = []
    Ns  = []
    for B in bins_list:
        # map to [0,B)
        g = ((X - mins) / rng * B).clip(0, B-1-1e-9)
        keys = np.floor(g).astype(int)
        # unique voxels
        if len(keys)==0: continue
        uniq = np.unique(keys, axis=0)
        Ns.append(len(uniq))
        eps.append(1.0/B)
    if len(Ns) < 3: return np.nan
    x = np.log(1/np.array(eps)); y = np.log(np.array(Ns))
    D, b = np.polyfit(x, y, 1)   # slope is dimension
    return float(np.exp(- abs(D - 2.0) / max(tau,1e-9)))

def geo_feat_concordance_score(X, F, k=12, max_eval=50000):
    if F is None: return np.nan
    N = len(X)
    rs = np.random.RandomState(0)
    idx = np.arange(N)
    if N > max_eval:
        idx = rs.choice(N, max_eval, replace=False)
    tree = cKDTree(X)
    nb = tree.query(X[idx], k=k+1)[1][:,1:]
    rhos = []
    for i, neigh in zip(idx, nb):
        dx = np.linalg.norm(X[neigh] - X[i], axis=1)
        df = np.linalg.norm(F[neigh] - F[i], axis=1)
        # Spearman via ranks
        rx = np.argsort(np.argsort(dx))
        rf = np.argsort(np.argsort(df))
        rx = (rx - rx.mean()) / (rx.std()+1e-9)
        rf = (rf - rf.mean()) / (rf.std()+1e-9)
        rho = float(np.mean(rx*rf))
        rhos.append(rho)
    rho_m = float(np.mean(rhos))
    return 0.5*(1.0 + rho_m)

def scaler( X):
    return (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8)

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

def normalize(X):
    if ssp.issparse(X): 
        X = X.toarray()

    X = X.copy()
    l2x = np.linalg.norm(X, ord=None, axis=1, keepdims=True)
    l2x[l2x == 0] = 1
    return X/l2x #*((self.DF/2.0)**0.5)
