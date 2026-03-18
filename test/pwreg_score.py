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

def ground_truth_state(X, Y, Xf, Yf, xlabels=None, knn=1, ylabels=None, alpha=0.5, 
                       kd_method='sknn', use_mnn=False, temp=[1,1], CI=1.0):
    from scipy import stats
    from sklearn.metrics.cluster import adjusted_rand_score
    from sklearn.metrics import adjusted_mutual_info_score
    from sklearn.metrics import normalized_mutual_info_score

    N, M = X.shape[0], Y.shape[0]
    X = np.array(X, dtype=np.float64)
    Y = np.array(Y, dtype=np.float64)

    src2, dst2, dist2 =coord_edges(X, Y, knn = knn,  method=kd_method)
    if use_mnn:     
        src1, dst1, dist1 = coord_edges(Y, X, knn = knn,  method=kd_method)
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

    df = pd.DataFrame(np.c_[score1, score2, score3, score4, score5],
                         columns=['score1', 'score2', 'score3', 'score4', 'score5'])

    if not (xlabels is None or ylabels is None):
        types = xlabels.columns.tolist()
        methods = [['ARI','AMI', 'NMI'], [adjusted_rand_score, adjusted_mutual_info_score, normalized_mutual_info_score]]
        scores = []
        for itl in types:
            for inm, imd in zip(*methods):
                iscore = imd(xlabels[itl].values[src2], ylabels[itl].values[dst2])
                scores.append([f'{inm}({itl})', iscore])
        scores = pd.DataFrame(scores, columns=['types', 'score'])
        return df, scores
    else:
        return df

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

def scaler( X):
    return (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8)

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

    return [src, dst, dist]

def normalize(X):
    if ssp.issparse(X): 
        X = X.toarray()

    X = X.copy()
    l2x = np.linalg.norm(X, ord=None, axis=1, keepdims=True)
    l2x[l2x == 0] = 1
    return X/l2x #*((self.DF/2.0)**0.5)

def pwreg_scores(adata12, MT, MD, knn=1, temp=[1e-2,1], CI=0.97, use_mnn=False, agg='mean',
                 alpha=0.5, labels=None,
                COLOR = ['gray','#e41a1c','#4daf4a','#009db2', 'violet', 'cyan', 'gold']
                    ):
    # sc.pp.scale(adata12, max_value=10)
    # sc.tl.pca(adata12, n_comps=40)

    X_data = adata12[(adata12.obs['Batch'] == 'X'), :]
    Y_data = adata12[(adata12.obs['Batch'] == 'Y'), :]

    # X_feat = X_data.X
    # Y_feat = Y_data.X

    X_feat = X_data.obsm['X_pca']
    Y_feat = Y_data.obsm['X_pca']
    if labels is not None:
        xlabels = X_data.obs[labels]
        ylabels = Y_data.obs[labels]

    # MT = ['raw', 'PASTE', 'PASTE2', 'SANTO', 'SPACEL', 'spateo', 'CCF']
    # MD = [None, paste_l, paste2_l, santo_l,spacel_l,spateo_l, ccf_l]

    dfs, txs, tys= [], [], []
    dfs1, scores1 = [], []
    for i, (ilc, iname) in enumerate(zip(MD, MT)):
        if iname == 'CCF':
            TX = X_data.obsm['ccf_e']
            TY = Y_data.obsm['ccf_e']
        elif iname in X_data.obsm.keys():
            TX = X_data.obsm[iname]
            TY = Y_data.obsm[iname]

        elif iname == 'raw':
            TX = X_data.obsm['spatial']
            TY = Y_data.obsm['spatial']

        else:
            ilc.index = ilc.index.astype(str)
            if 'Z' in ilc.columns:
                TX = ilc.loc[X_data.obs_names,['X', 'Y', 'Z']].values
                TY = ilc.loc[Y_data.obs_names,['X', 'Y', 'Z']].values
            else:
                TX = ilc.loc[X_data.obs_names,['X', 'Y']].values
                TY = ilc.loc[Y_data.obs_names,['X', 'Y']].values
        idf = ground_truth_state(TX, TY, X_feat, Y_feat, knn=knn, temp=temp, CI=CI, alpha=alpha, use_mnn=use_mnn)

        if agg == 'mean':
            mdf = idf.mean(0)
            mdf['score4'] = idf['score4'].sum(0) /(idf.shape[0]**0.5)
            mdf['score5'] = idf['score5'].sum(0) /(idf.shape[0]**0.5)
            mdf.name = iname
        elif agg == 'sum':
            mdf = idf.sum(0)
            mdf.name = iname
        else:
            mdf = idf
        
        if labels is not None:
            idf0, iscore = ground_truth_state0(TX, TY,  X_feat, Y_feat,
                                        xlabels=xlabels, ylabels=ylabels, CI=CI)
            iscore['method'] = iname
            idf0['method'] = iname
            dfs1.append(idf0)
            scores1.append(iscore)
            
        dfs.append(mdf)
        txs.append(TX)
        tys.append(TY)

    dfs = pd.concat(dfs, axis=1)

    if labels is not None:
        dfs1 = pd.concat(dfs1, axis=0)
        dfs1['method'] = pd.Categorical(dfs1['method'], categories=MT)

        sims = pd.concat(scores1, axis=0)
        sims = sims.pivot(columns='types', index='method', values='score')

        stats = pd.concat([dfs.T, dfs1.groupby('method').mean(0), sims], axis=1)
        stats['distance(mean)'] = -stats['distance(mean)']
        stats

        return dfs, txs, tys, stats
    else:
        return dfs, txs, tys

def transpos(adata, pos_df):
    # adata = ccf_l
    # pos_df = paste_l

    fidx = pos_df.group=='f'
    midx = pos_df.group=='m'

    if 'Z' in pos_df.columns:
        xyz = ['X', 'Y', 'Z']
        xyz0 = ['X0', 'Y0', 'Z0']
    else:
        xyz = ['X', 'Y']
        xyz0 = ['X0', 'Y0']
    tx = pos_df[fidx][xyz]
    ty = pos_df[midx][xyz]

    rx = pos_df[fidx][xyz0]
    ry = pos_df[midx][xyz0]

    xids = adata.obs['Batch'] == 'X'
    yids = adata.obs['Batch'] == 'Y'

    rxa = adata[xids].obsm['spatial']
    rya = adata[yids].obsm['spatial']

    rx0 = adata[pos_df[fidx].index].obsm['spatial']
    ry0 = adata[pos_df[midx].index].obsm['spatial']

    print( 111, np.array(rx0 - rx).sum(), np.array(rx0 - rx).sum() )

    tmaty = cc.tf.homotransform_estimate(ry, ty, transformer='rigid')
    tmatx = cc.tf.homotransform_estimate(rx, tx, transformer='rigid')

    ny =  cc.tf.homotransform_point(rya, tmaty)
    nx =  cc.tf.homotransform_point(rxa, tmatx)

    pos_new = pd.DataFrame( np.c_[ np.r_[nx, ny], np.r_[rxa, rya] ], 
                           columns = [*xyz, *xyz0],
                            index = np.r_[adata.obs_names[xids], adata.obs_names[yids], ] )
    pos_new['group'] = np.repeat(['f','m'], repeats=[nx.shape[0], ny.shape[0]])

    print(222,  (pos_new.loc[ pos_df.index, [*xyz, *xyz0] ] - 
             pos_df[[*xyz, *xyz0]]).values.sum() )
    return pos_new

def plt_dens(dfs, figsize=(4.5, 4.5), y_lim = (0,200), ys_lim = [-1,1], linewidth=1, save=None):
    import seaborn as sns
    palette = COLOR
    dfs = dfs.reset_index(drop=True)
    #['#009db2', '#765005']
    fig, axs = plt.subplots(1, 3, figsize=figsize,
                             constrained_layout=False, sharex=False, sharey=False)

    sns.kdeplot(hue='method', y='distance(mean)',  linewidth=linewidth, cut=2,
                palette =palette,
                data=dfs, ax=axs[0], legend=False)
    axs[0].grid(False)
    # axs[0].get_xaxis().set_visible(False)
    # axs[0].get_yaxis().set_visible(False)
    axs[0].set_ylim(*y_lim)

    sns.kdeplot(hue='method', y='cosine(mean)',  linewidth=linewidth, cut=2,
                palette =palette,
                data=dfs, ax=axs[1], legend=False)
    axs[1].grid(False)
    # axs[1].get_xaxis().set_visible(False)
    # axs[1].get_yaxis().set_visible(False)
    axs[1].set_ylim(*ys_lim)

    ax2 = sns.kdeplot(hue='method', y='pearson(mean)',  linewidth=linewidth, cut=2,
                palette =palette,
                data=dfs,ax=axs[2], legend=True)
    axs[2].grid(False)
    # axs[2].get_xaxis().set_visible(False)
    # axs[2].get_yaxis().set_visible(False)
    axs[2].set_ylim(*ys_lim)

    fig.tight_layout()
    if save:
        fig.savefig(f'{save}.score.pdf',  transparent=True, format='pdf' )
        dfs.to_csv(f'{save}.score.csv')
    plt.show()

def P_value(df, groupy, stats):
    from scipy.stats import mannwhitneyu, ttest_ind
    pars = df[groupy].unique()
    p_vals = []
    for i in range(len(pars)-1):
        xdf = df[(df[groupy] == pars[i])]
        ydf = df[(df[groupy] == pars[i+1])]
        for ist in stats:
            #p1 = mannwhitneyu(xdf[ist], ydf[ist], method="exact")[1]
            ix = xdf[ist].values
            iy = ydf[ist].values
            ix = ix[~pd.isna(ix)]
            iy = iy[~pd.isna(iy)]

            p2 = ttest_ind(ix, iy)[1]
            ps = [pars[i], pars[i+1], ist, p2]
            p_vals.append(ps)
    return p_vals

def plt_hist0(dfs, figsize=(4.5, 6), y_lim = (0,100), ys_min = None, ys_max = None,
             linewidth=1, save=None):
    import seaborn as sns
    palette = COLOR
    idf = dfs.groupby('method').mean(0).reset_index()
    fig, axs = plt.subplots( 3,1, figsize=figsize,
                             constrained_layout=False, sharex=False, sharey=False)

    sns.barplot( y='method', x='distance(mean)',  linewidth=linewidth, 
                palette =palette,
                data=idf, ax=axs[0], legend=False)
    axs[0].grid(False)
    # axs[0].get_xaxis().set_visible(False)
    # axs[0].get_yaxis().set_visible(False)
    axs[0].set_xlim(0, idf['distance(mean)'].max())
    axs[0].axvline(idf['distance(mean)'].min(), c='gray', linestyle='-.')

    sns.barplot( y='method', x='cosine(mean)',  linewidth=linewidth, 
                palette =palette,
                data=idf, ax=axs[1], legend=False)
    axs[1].grid(False)
    # axs[1].get_xaxis().set_visible(False)
    # axs[1].get_yaxis().set_visible(False)
    ys_lim = [idf['cosine(mean)'].min() if ys_min is None else ys_min, 
              idf['cosine(mean)'].max() if ys_max is None else ys_max]
    axs[1].set_xlim(*ys_lim)

    ax2 = sns.barplot( y='method', x='pearson(mean)',  linewidth=linewidth, 
                palette =palette,
                data=idf,ax=axs[2], legend=True)
    axs[2].grid(False)
    # axs[2].get_xaxis().set_visible(False)
    # axs[2].get_yaxis().set_visible(False)

    ys_lim = [idf['pearson(mean)'].min() if ys_min is None else ys_min, 
              idf['pearson(mean)'].max() if ys_max is None else ys_max]
    axs[2].set_xlim(*ys_lim)

    fig.tight_layout()
    if save:
        fig.savefig(f'{save}.score.hist.pdf',transparent=True, format='pdf')
        idf.to_csv(f'{save}.score.hist.csv')
    plt.show()

def plt_hist(dfs, figsize=(4.5, 4), y_lim = (0,100), ys_min = None, ys_max = None, 
             scores= ['score3', 'score4'],
             COLOR = ['#e41a1c','#4daf4a','#009db2', 'violet', 'cyan', 'gold'],
             linewidth=1, save=None):
    import seaborn as sns
    palette = COLOR
    
    fig, axs = plt.subplots( len(scores), 1, figsize=figsize,
                            constrained_layout=False, sharex=False, sharey=False)
    for i, iscore in enumerate(scores):
        idf = dfs.iloc[:, 1:].loc[iscore].copy()

        sns.barplot( idf,  linewidth=linewidth, 
                    palette =palette, orient= 'h',
                    ax=axs[i], legend=False)
        axs[i].grid(False)
        axs[i].set_title(iscore)

    fig.tight_layout()
    if save:
        fig.savefig(f'{save}.score.hist.pdf',transparent=True, format='pdf')
        dfs.to_csv(f'{save}.score.hist.csv')
    plt.show()

def plt_hist_cap(dfs, figsize=(4.5, 2), y_lim = (0,100),
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
        idf = dfs.iloc[:, 1:].loc[iscore].copy()
        fig, (ax_left, ax_right) = plt.subplots(ncols=2, nrows=1, sharey=True, 
                                              figsize=figsize,
                                              width_ratios=[1-break_ratio, break_ratio],
                                              gridspec_kw={'wspace':wspace})
        sns.barplot( idf,  linewidth=linewidth, 
                    palette =palette, orient= 'h',
                    ax=ax_left, legend=False)
        sns.barplot( idf,  linewidth=linewidth, 
                    palette =palette, orient= 'h',
                    ax=ax_right, legend=False)

        ax_left.grid(False)
        ax_right.grid(False)
        ax_left.set_xlabel('')
        ax_right.set_xlabel('')

        L = idf.max()
        if cap is None:
            cap_ratio = cap_ratio or 0.75
            cap = L * cap_ratio
        else:
            capl = min(cap, L*0.98)
        capp = cap_space * L

        ax_right.set_xlim(left=capl+capp)   # those limits are fake
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

        fig.tight_layout()
        if save:
            fig.savefig(f'{save}.{iscore}.hist.pdf',transparent=True, format='pdf')
            dfs.to_csv(f'{save}.{iscore}.hist.csv')
        plt.show()

def plt_hist1(dfs, figsize=(4.5, 6), y_lim = (0,100), ys_min = None, ys_max = None, error =0,
             linewidth=1, save=None):
    import seaborn as sns
    palette = COLOR
    dfs = dfs.copy().reset_index(drop=True)

    idf = dfs.groupby('method').mean(0).reset_index()
    fig, axs = plt.subplots( 3,1, figsize=figsize,
                             constrained_layout=False, sharex=False, sharey=False)

    g = sns.barplot( y='method', x='distance(mean)',  linewidth=linewidth, 
                palette =palette,  errorbar=('ci', 95),
                data=dfs.copy(), ax=axs[0], legend=False)

    axs[0].grid(False)
    # axs[0].get_xaxis().set_visible(False)
    # axs[0].get_yaxis().set_visible(False)
    axs[0].set_xlim(y_lim[0], idf['distance(mean)'].max() if y_lim[1] is None else y_lim[1])
    axs[0].axvline(idf['distance(mean)'].min(), c='gray', linestyle='-.')

    g1 = sns.barplot( y='method', x='cosine(mean)',  linewidth=linewidth, 
                palette =palette, errorbar=('ci', 95),
                data=dfs.copy(), ax=axs[1], legend=False)
    axs[1].grid(False)
    # axs[1].get_xaxis().set_visible(False)
    # axs[1].get_yaxis().set_visible(False)
    ys_lim = [idf['cosine(mean)'].min() if ys_min is None else ys_min, 
              idf['cosine(mean)'].max()+error if ys_max is None else ys_max]
    axs[1].set_xlim(*ys_lim)

    g2 = sns.barplot( y='method', x='pearson(mean)',  linewidth=linewidth, 
                palette =palette, errorbar=('ci', 95),
                data=dfs, ax=axs[2], legend=False)
    axs[2].grid(False)
    # axs[2].get_xaxis().set_visible(False)
    # axs[2].get_yaxis().set_visible(False)
    ys_lim = [idf['pearson(mean)'].min() if ys_min is None else ys_min, 
              idf['pearson(mean)'].max()+error if ys_max is None else ys_max]
    axs[2].set_xlim(*ys_lim)

    fig.tight_layout()
    if save:
        fig.savefig(f'{save}.score.hist.pdf',transparent=True, format='pdf')
        idf.to_csv(f'{save}.score.hist.csv')
    plt.show()

def plt_scatter( txs, tys, save=None, size=1.5, wspace=0, hspace=0, sharey=False,sharex=False,
                rasterized=True, alpha=None, dpi=1000,
                MTs = ['raw', 'PASTE', 'PASTE2', 'SANTO', 'SPACEL', 'spateo', 'CCF'],
                COLOR = ['gray', '#e41a1c','#4daf4a','#009db2', 'violet', 'cyan', 'gold'],
                invert_xaxis=False, invert_yaxis=False, figsize=(18,5),):
    import matplotlib.ticker as ticker
    palette= COLOR
    txsn, tysn = [],[]
    for itx,ity in zip(txs,tys):
        shift = itx.mean(0)
        txsn.append(itx-shift)
        tysn.append(ity-shift)
    
    amin = np.r_[ txsn[0], tysn[0]].min(0) 
    amax = np.r_[ txsn[0], tysn[0]].max(0)
   
    bmin = np.concatenate([*txsn[1:], *tysn[1:]], axis=0).min(0)
    bmax = np.concatenate([*txsn[1:], *tysn[1:]], axis=0).max(0)
        
    fig, ax = plt.subplots(1,len(MTs), figsize=figsize, sharey=sharey,sharex=sharex)
    fig.subplots_adjust(wspace=wspace, hspace=hspace)
    for i, (ilc, iname) in enumerate(zip(palette, MTs)):
        ialpha = alpha if i>0 else 1
        if i ==0:
            if invert_xaxis:
                ax[i].invert_xaxis()
            if invert_yaxis:
                ax[i].invert_yaxis()
        iX = txsn[i].astype(np.float64)
        iY = tysn[i].astype(np.float64)

        # shift = iX.mean(0)
        # iX -= shift
        # iY -= shift

        ax[i].scatter(iX[:,0], iX[:,1], s=size, edgecolor = 'none', rasterized=rasterized, c='#b39ddf')
        ax[i].scatter(iY[:,0], iY[:,1], s=size, edgecolor = 'none', rasterized=rasterized, alpha=ialpha, c=palette[i])
        ax[i].grid(False)
        ax[i].set_aspect('equal', adjustable='box', anchor='C')
        ax[i].set_title(iname)
        if i == 0:
            ax[i].set_xlim(amin[0], amax[0])
            ax[i].set_ylim(amin[1], amax[1])
        else:
            ax[i].set_xlim(bmin[0], bmax[0])
            ax[i].set_ylim(bmin[1], bmax[1])

        # ax[i].set_axis_off()
        ax[i].xaxis.set_major_locator(ticker.NullLocator())
        ax[i].yaxis.set_major_locator(ticker.NullLocator())
    fig.tight_layout()
    if save:
        plt.savefig(f'{save}.scatter.pdf', dpi=dpi, transparent=True, format='pdf')
        plt.savefig(f'{save}.scatter.svg', dpi=dpi, transparent=True)
    plt.show()

def plt_scatter0( txs, tys, save=None, size=1.5, wspace=0, hspace=0, sharey=True,sharex=True,
                rasterized=True, alpha=None, dpi=1000,
                MTs = ['raw', 'PASTE', 'PASTE2', 'SANTO', 'SPACEL', 'spateo', 'CCF'],
                COLOR = ['gray', '#e41a1c','#4daf4a','#009db2', 'violet', 'cyan', 'gold'],
                invert_xaxis=False, invert_yaxis=False, figsize=(18,5),):
    import matplotlib.ticker as ticker
    palette = COLOR
    fig, ax = plt.subplots(1,len(MTs), figsize=figsize, sharey=sharey,sharex=sharex)
    fig.subplots_adjust(wspace=wspace, hspace=hspace)
    for i, (ilc, iname) in enumerate(zip(palette, MTs)):
        ialpha = alpha if i>0 else 1
        if i ==0:
            if invert_xaxis:
                ax[i].invert_xaxis()
            if invert_yaxis:
                ax[i].invert_yaxis()
        iX = txs[i].astype(np.float64)
        iY = tys[i].astype(np.float64)
        shift = iX.mean(0)

        iX -= shift
        iY -= shift
        ax[i].scatter(iX[:,0], iX[:,1], s=size, edgecolor = 'none', rasterized=rasterized, c='#b39ddf')
        ax[i].scatter(iY[:,0], iY[:,1], s=size, edgecolor = 'none', rasterized=rasterized, alpha=ialpha, c=palette[i])
        ax[i].grid(False)
        ax[i].set_aspect('equal', adjustable='box', anchor='C')
        ax[i].set_title(iname)
        # ax[i].set_axis_off()
        ax[i].xaxis.set_major_locator(ticker.NullLocator())
        ax[i].yaxis.set_major_locator(ticker.NullLocator())
    fig.tight_layout()
    if save:
        plt.savefig(f'{save}.scatter.pdf', dpi=dpi, transparent=True, format='pdf')
        plt.savefig(f'{save}.scatter.svg', dpi=dpi, transparent=True)
    plt.show()

def plt_scatter_ad(adatas, bases, color, save=None, size=None, wspace=0, hspace=0, sharey=True,sharex=True,
                rasterized=True, alpha=None, dpi=1000, figscale=5,
                invert_xaxis=False, invert_yaxis=False, figsize=None,):
    import matplotlib.ticker as ticker
    if figsize is None:
        figsize = (figscale*len(bases), figscale)
    fig, ax = plt.subplots(1, len(bases), figsize=figsize, sharey=sharey,sharex=sharex)
    fig.subplots_adjust(wspace=wspace, hspace=hspace)
    for i, basis in enumerate(bases):
        if len(bases) == 1:
            ax = [ax]
        ialpha = alpha if i>0 else 1
        if i ==0:
            if invert_xaxis:
                ax[i].invert_xaxis()
            if invert_yaxis:
                ax[i].invert_yaxis()
        legend_loc = 'right margin' if i == len(bases) -1 else None
        sc.pl.embedding(adatas,  color=color, basis=basis, size=size, legend_loc=legend_loc, show=False, ax=ax[i])
        ax[i].grid(False)
        ax[i].set_aspect('equal', adjustable='box', anchor='C')
        # ax[i].set_axis_off()
        ax[i].xaxis.set_major_locator(ticker.NullLocator())
        ax[i].yaxis.set_major_locator(ticker.NullLocator())

    fig.tight_layout()
    if save:
        plt.savefig(f'{save}.scatter.{color}.pdf', dpi=dpi, transparent=True, format='pdf')
        plt.savefig(f'{save}.scatter.{color}.svg', dpi=dpi, transparent=True)
    plt.show()

def scatter3d_mpl(adata, groupby, basis='spatial',
                  colors=None, transparent=True,
                  use_raw =False,
                  figsize=(5,5), size= 0.4,
                  rasterized=True, alpha =1, facecolor=None,
                  edgecolors='none' ,frame_type=4, xyz=[0,1,2],
                  show_legend=True, loc="center left", lncol=1, lfs=7, 
                  axis_off =True, cmap='viridis', vmin=None, vmax=None,
                  sharex=True, sharey=True, save=None, show=True, dpi=None,
                  markerscale=10, scatterpoints=1, bbox_to_anchor=(0.93, 0, 0.5, 1),
                  invert_xaxis=False, invert_yaxis=False, invert_zaxis=False,
                  elev=None, azim=None, roll=None, ax=None, saveargs={},):
    from matplotlib.lines import Line2D

    if use_raw:
        idata = adata.raw
    else:
        idata = adata

    if groupby in adata.obs_names:
        gdata = idata.obs[groupby]
    else:
        gdata = idata[:, groupby].X
        if ssp.issparse(gdata):
            gdata = gdata.toarray()
        gdata = gdata.flatten()
    
    gdtype = cc.ut._arrays.vartype(gdata)

    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize,
                                 sharex=sharex, sharey=sharey,  facecolor= facecolor,
                                 subplot_kw=dict(projection="3d"))
        plt.subplots_adjust(wspace=0,hspace=0)

    ax.view_init(elev=elev, azim=azim, roll=roll)

    if gdtype == 'continuous': #PASS
        colors1 = gdata #need normal
        mapdata= adata.obsm[basis][:,xyz]
    
        ax.scatter(mapdata[:, 0], mapdata[:, 1], mapdata[:, 2], 
                s=size, 
                c=colors1, 
                edgecolors=edgecolors,
                rasterized=rasterized, 
                cmap= cmap,
                vmin=vmin, vmax=vmax,
                alpha=alpha)
    else:
        colors = adata.uns[f'{groupby}_colors'] if colors is None else colors
        labels = adata.obs[groupby].cat.remove_unused_categories().cat.categories.tolist()
        colors1 = adata.obs[groupby].copy().cat.rename_categories(colors)
        mapdata= adata.obsm[basis][:,xyz]

        ax.scatter(mapdata[:, 0], mapdata[:, 1], mapdata[:, 2], 
                s=size, 
                c=colors1, 
                edgecolors=edgecolors,
                rasterized=rasterized, 
                alpha=alpha)
    

    # for i, c, label in zip(range(len(labels)), colors, labels):
    #     widx = (adata.obs[groupby] == label)
    #     imap = mapdata[widx,:]
    #     ax.scatter(imap[:, 0], imap[:, 1], imap[:, 2], 
    #             s=size, 
    #             c=c, 
    #             label=label, 
    #             edgecolors=edgecolors,
    #             rasterized=rasterized, 
    #             alpha=alpha)

    cc.pl.add_frame(ax, frame_linewidth=0.3,
                    #xlims=(-1,10), ylims=(0,10),zlims=(-1,15),
                    frame_type=frame_type)
    ax.set_aspect('equal', 'box')

    if facecolor is not None:
        ax.set_facecolor(facecolor)
    if axis_off:
        ax.set_axis_off()

    if invert_xaxis:
        ax.invert_xaxis()
    if invert_yaxis:
        ax.invert_yaxis()
    if invert_zaxis:
        ax.invert_zaxis()

    if gdtype == 'discrete':
        legend_elements = [
                    Line2D([0], [0], marker='o', color=icl, label=ila,
                            linewidth=0,
                            markerfacecolor=icl, markersize=size)
        for icl, ila in zip(colors, labels) ]

        if show_legend:
            ax.legend( #title=groupby, 
                        handles=legend_elements, 
                        loc=loc, ncol=lncol,
                        prop={'size':lfs},
                        #alignment='left',
                        #scatterpoints=scatterpoints,
                        bbox_to_anchor=bbox_to_anchor,
                        #frameon=frameon,
                        #mode=mode,
                        markerscale=markerscale)
    try:
        fig.tight_layout()
    except:
        pass

    if save:
        fig.savefig(save, dpi=dpi, transparent=True, **saveargs)
    if show is None:
        return fig, ax
    elif show is True:
        plt.show()
    else:
        plt.close()


def scatterstack(*Xs, figsize=(8,8), samples=None, size=1, sharex=True, sharey=True,  color=None):
    L = len(Xs)
    if isinstance(size, (list, tuple)):
        pointsize = size
    else:
        pointsize = [size for _ in range(L)]

    if color is None or len(color) ==0:
        color = cc.pl.color_palette(L)

    fig, ax = plt.subplots(1,1, sharex=sharex, sharey=sharey,  figsize=figsize)
    for i in range(L):
        ax.scatter(Xs[i][:,0], Xs[i][:,1], s=pointsize[i], c=color[i],  edgecolors='None',)
    ax.set_aspect('equal', 'box')
    plt.show()


def spider(df, id_column=None, columns=None, max_values=None, 
            title=None, alpha=0.15, 
            color_bg='#A0A0A0', alpha_bg=0.05, 
            colors=None, fs='xx-small', fs_format = '.3f',
            padding=1.05, figsize=(5,5), rotate_label=True,
             show_legend=True, bbox_to_anchor=(0.1, 0.1),
            saveargs={}, show=True, save=None, ax=None,
            **kargs):
    columns = df._get_numeric_data().columns.tolist() if columns is None else columns
    data = df[columns].to_dict(orient='list')
    ids = df.index.tolist() if id_column is None else df[id_column].tolist()
    if max_values is None:
        max_values = {key: padding*max(value) for key, value in data.items()}

    normalized_data = {key: np.array(value) / max_values[key] for key, value in data.items()}
    num_vars = len(data.keys())
    tiks = list(data.keys())
    tiks += tiks[:1]
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist() + [0]
    degrees = [ np.degrees(angle-np.pi if angle>np.pi else angle ) - 90 for angle in angles]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    for i, model_name in enumerate(ids):
        values = [normalized_data[key][i] for key in data.keys()]
        actual_values = [data[key][i] for key in data.keys()]
        values += values[:1]  # Close the plot for a better look
        icolor = None if colors is None else colors[i]

        ax.plot(angles, values, c=icolor, label=model_name, **kargs)
        ax.fill(angles, values, c=icolor, alpha=alpha)
        for _x, _y, t, r in zip(angles, values, actual_values, degrees):
            t = f'{t :{fs_format}}' if isinstance(t, float) else str(t)
            ax.text(_x, _y, t, size=fs, rotation=r,
                    rotation_mode='anchor')

    if not color_bg is None:
        ax.fill(angles, np.ones(num_vars + 1), alpha=alpha_bg)
    ax.set_yticklabels([])
    ax.set_xticks(angles)
    ax.set_xticklabels(tiks)
    # ax.grid(linewidth=3)

    if rotate_label:
        for label, degree in zip(ax.get_xticklabels(), degrees):
            x,y = label.get_position()
            lab = ax.text(x,y, label.get_text(), transform=label.get_transform(),
                          ha=label.get_ha(), va=label.get_va())
            lab.set_rotation(degree)
        ax.set_xticklabels([])

    if show_legend: ax.legend(loc='upper right', bbox_to_anchor=bbox_to_anchor)
    if title is not None: plt.suptitle(title)

    try:
        fig.tight_layout()
    except:
        pass

    if save:
        fig.savefig(save, **saveargs)
    if show is None:
        return fig, ax
    elif show is True:
        plt.show()
    else:
        plt.close()

def plot_regist_2d(posreg, save=None, figsize=(10,5), size=0.5, sharex=True, 
                   slice_colors = ['#e41a1c','#377eb8','#4daf4a','#984ea3']):
    X = posreg[(posreg['group'] == 'f')]
    Y = posreg[(posreg['group'] == 'm')]

    fig, ax = plt.subplots(1,2, figsize=figsize, sharex=sharex,)
    ax[0].scatter(X['X'], X['Y'], s=size, edgecolors='none', c='#b39ddf')
    ax[0].scatter(Y['X'], Y['Y'], s=size, edgecolors='none', c='c')
    ax[0].grid(False)
    ax[0].set_aspect('equal', adjustable='box')

    ax[1].scatter(X['X0'], X['Y0'], s=size, edgecolors='none', c='#b39ddf')
    ax[1].scatter(Y['X0'], Y['Y0'], s=size, edgecolors='none', c='c')
    ax[1].grid(False)
    ax[1].set_aspect('equal', adjustable='box')
    fig.tight_layout()
    if save:
        plt.savefig(save)
    plt.show()

def plt_hist01(dfs,  figsize=(4.5, 4.5), y_lim = (0,100), ys_lim = [-1,1], linewidth=1, save=None):
    import seaborn as sns
    palette = ['#e41a1c','#4daf4a','#009db2', 'gold']
    #['#009db2', '#765005']
    fig, axs = plt.subplots(1, 3, figsize=figsize,
                             constrained_layout=False, sharex=False, sharey=False)

    sns.kdeplot(hue='method', y='distance(mean)',  linewidth=linewidth, cut=2,
                palette =palette,
                data=dfs, ax=axs[0], legend=False)
    axs[0].grid(False)
    axs[0].get_xaxis().set_visible(False)
    axs[0].get_yaxis().set_visible(False)
    axs[0].set_ylim(*y_lim)
    
    sns.kdeplot(hue='method', y='cosine(mean)',  linewidth=linewidth, cut=2,
                palette =palette,
                data=dfs, ax=axs[1], legend=False)
    axs[1].grid(False)
    axs[1].get_xaxis().set_visible(False)
    axs[1].get_yaxis().set_visible(False)
    axs[1].set_ylim(*ys_lim)
    
    ax2 = sns.kdeplot(hue='method', y='pearson(mean)',  linewidth=linewidth, cut=2,
                palette =palette,
                data=dfs,ax=axs[2], legend=True)
    axs[2].grid(False)
    axs[2].get_xaxis().set_visible(False)
    axs[2].get_yaxis().set_visible(False)
    axs[2].set_ylim(*ys_lim)
    
    fig.tight_layout()
    if save:
        fig.savefig(f'{save}.score.pdf', format='pdf')
        dfs.to_csv(f'{save}.score.csv')
    plt.show()

def ground_truth_state0(TX, TY, X_feat, Y_feat, xlabels=None, knn=1, ylabels=None, CI=0.95):
    from scipy import stats
    from sklearn.metrics.cluster import adjusted_rand_score
    from sklearn.metrics import adjusted_mutual_info_score
    from sklearn.metrics import normalized_mutual_info_score

    src2, dst2, dist2 = cc.tl.coord_edges(TX, TY, knn = knn,  method='sknn')

    if not CI is None:
        K  = int(len(dist2)*CI)
        kidx = np.argpartition(dist2, K,)[:K]
        # kidx2 = np.argsort(dist2)[:K]
        src2, dst2, dist2 = src2[kidx], dst2[kidx], dist2[kidx]

    # normd = np.sum(np.square(TX_n[src2] - TY_n[dst2]), axis=1)
    # normd = np.exp(-0.5* normd)

    cos2 = cc.tl.mtx_similarity(X_feat, Y_feat, method='cosine', pairidx=[src2, dst2])
    # pea2 = cc.tl.mtx_similarity(X_feat, Y_feat, method='pearson', pairidx=[src2, dst2])
    pea2 = stats.pearsonr(X_feat[src2], Y_feat[dst2], axis=1).statistic

    df = pd.DataFrame(np.c_[dist2, cos2, pea2],
                         columns=['distance(mean)', 'cosine(mean)', 'pearson(mean)'])

    if not (xlabels is None or ylabels is None):
        types = xlabels.columns.tolist()
        methods = [['ARI','AMI', 'NMI'], [adjusted_rand_score, adjusted_mutual_info_score, normalized_mutual_info_score]]
        scores = []
        for itl in types:
            for inm, imd in zip(*methods):
                iscore = imd(xlabels[itl].values[src2], ylabels[itl].values[dst2])
                scores.append([f'{inm}({itl})', iscore])
        scores = pd.DataFrame(scores, columns=['types', 'score'])
        return df, scores
    else:
        return df