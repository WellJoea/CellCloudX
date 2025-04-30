import numpy as np
import collections
import scipy as sci
import matplotlib.pyplot as plt
from scipy.sparse import csr_array
from typing import List, Optional
from joblib import Parallel, delayed

from ...tools._exp_edges import mtx_similarity
from ...tools._decomposition import dualPCA
from ...tools._neighbors import Neighbors
from ...tools._search import searchidx
from ...plotting._imageview import drawMatches

class SSWNN():
    def __init__(self, latent, groups, splocs=None, levels = None, verbose=1):

        self.latent = np.array(latent)
        self.splocs = splocs
        self.groups = groups
        self.N = latent.shape[0]
        self.D = None
        self.lables = np.arange(self.N)
        self.cellid = np.arange(self.N)
    
        assert len(groups) == self.N
        if not splocs is None:
            assert splocs.shape[0] == self.N
            self.D = splocs.shape[0]

        if levels is None:
            try:
                self.order = groups.cat.remove_unused_categories().cat.categories
            except:
                self.order = np.unique(groups)
        else:
            self.order = levels

        self.mtx_similarity = mtx_similarity
        self.verbose = verbose

    def build(self,
                dpca_npca = 50,
                method='hnsw',
                spmethod=None,
                metric='euclidean',
                n_jobs=-1,
                root=None, 
                regist_pair=None,
                full_pair=False,
                step=1,
                show_tree=False, 
                keep_self=True,

                **kargs):

        self.align_pair, self.trans_pair = searchidx(len(self.order), 
                                                        root=root,
                                                        regist_pair=regist_pair,
                                                        full_pair=full_pair,
                                                        keep_self=keep_self,
                                                        step=step,
                                                        show_tree=show_tree)

        groupidx = collections.OrderedDict()
        for igroup in self.order:
            groupidx[igroup] = [self.groups == igroup, self.cellid[(self.groups == igroup)]]
        self.groupidx = groupidx

        enns = {}
        for sid in self.order:
            idx = self.groupidx[sid][0]
            enn = Neighbors(method=method, metric=metric, n_jobs=n_jobs)
            enn.fit(self.latent[idx], **kargs)
            enns[sid] = enn
        self.enns = enns

        if not self.splocs is None:
            snns = {}
            if spmethod is None:
                if self.splocs.shape[1] >=10:
                    spmethod = method
                else:
                    spmethod = 'sknn'

            for sid in self.order:
                idx = self.groupidx[sid][0]
                snn = Neighbors(method=spmethod, metric=metric, n_jobs=n_jobs)
                snn.fit(self.splocs[idx], **kargs)
                snns[sid] = snn
            self.snns = snns
        else:
            self.snns = None

        self.largs = dict(method=method, metric=metric, n_jobs=n_jobs, 
                          dpca_npca=dpca_npca, kargs=kargs)

    def kdtree_dpca(self, rsid, qsid, qdata = None, knn= 11, 
                    dpca_npca=None, dpca_scale=True, **kargs):
        ridx, rlabel = self.groupidx[rsid]
        qidx, qlabel = self.groupidx[qsid]

        rdata = self.latent[ridx]
        qdata = self.latent[qidx] if qdata is None else qdata
        rdata, qdata = dualPCA(rdata, qdata, 
                                n_comps = self.largs.get('dpca_npca', dpca_npca),
                                scale=self.largs.get('dpca_scale', dpca_scale),
                                axis=0,
                                zero_center=True)
        kdnn = Neighbors(method=self.largs.get('method', 'annoy'), 
                         metric=self.largs.get('metric', 'euclidean'), 
                         n_jobs=self.largs.get('n_jobs', -1))

        kdnn.fit(rdata, **self.largs.get('kargs', {}))
        cdkout = kdnn.transform(qdata, knn=knn, **kargs)
        return kdnn, cdkout

    @staticmethod
    def translabel(ckdout, rsize=None, rlabel=None, qlabel=None, return_type='raw'):
        nnidx = ckdout[1]
        # minrnum = np.int64(nnidx.max()- nnidx.min()+1)
        minrnum = np.unique(nnidx).shape[0]
        if not rlabel is None:
            assert len(rlabel) >= minrnum
        if not rsize is None:
            assert rsize >= minrnum
        if not qlabel is None:
            assert len(qlabel) == len(nnidx)
        
        if return_type == 'raw':
            return [ckdout[0], nnidx]
        elif return_type == 'raw_lable':
            rlabel = np.asarray(rlabel)
            return [ckdout[0], rlabel[nnidx]]
        elif return_type in ['lists', 'lists_label', 'sparse', 'sparseidx']:
            try:
                src = nnidx.flatten('C')
                dst = np.repeat(np.arange(nnidx.shape[0]), nnidx.shape[1])
                dist = ckdout[0].flatten('C')
            except:
                src = np.concatenate(nnidx, axis=0)
                dst = np.repeat(np.arange(len(nnidx)), list(map(len, nnidx)))
                dist = np.concatenate(ckdout[0], axis=0)

            if return_type in ['sparse', 'sparseidx']:
                rsize = rsize or (None if rlabel is None else len(rlabel)) or minrnum ## set fixed value
                if return_type == 'sparseidx':
                    dist = np.ones_like(dst)
                adj = csr_array((dist, (dst, src)), shape=(nnidx.shape[0], rsize))
                if not adj.has_sorted_indices:
                    adj.sort_indices()
                adj.eliminate_zeros()
                return adj
            elif return_type in ['lists_label']:
                if not rlabel is None:
                    src = np.asarray(rlabel)[src]
                if not qlabel is None:
                    dst = np.asarray(qlabel)[dst]
                return [src, dst, dist]
            elif return_type in ['lists']:
                return [src, dst, dist]
        else:
            raise ValueError('return_type must be one of "raw", "raw_lable", "lists", "lists_label",  "sparse", "sparseidx"')

    def query(self, rsid, qsid, slot = 'enn', qdata = None,
               use_dpca=True, dpca_npca=50,  knn=11, return_type='raw', **kargs):
        ridx, rlabel = self.groupidx[rsid]
        qidx, qlabel = self.groupidx[qsid]

        if use_dpca and (slot == 'enn'):
            kdnn, cdkout = self.kdtree_dpca( rsid, qsid, qdata = qdata, dpca_npca=dpca_npca, knn= knn, **kargs)
        elif slot == 'enn':
            kdnn = self.enns[rsid]
            qdata = self.latent[qidx] if qdata is None else qdata
            cdkout = kdnn.transform(qdata, knn=knn, **kargs)
        elif slot == 'snn':
            kdnn = self.snns[rsid]
            qdata = self.splocs[qidx] if qdata is None else qdata
            cdkout = kdnn.transform(qdata, knn=knn, **kargs)
        return self.translabel(cdkout, rlabel=rlabel, qlabel=qlabel, return_type=return_type)

    def simi_pair(self, rsid, qsid, method = 'cosine', pairidx = None):
        ridx, _ = self.groupidx[rsid]
        qidx, _ = self.groupidx[qsid]

        rdata = self.latent[ridx]
        qdata = self.latent[qidx]

        return self.mtx_similarity(rdata, qdata, method=method, pairidx=pairidx)

    def selfsnns(self, sids=None, o_neighbor = 60, s_neighbor =30, show_simi = False):
        rrnns = {}
        if sids is None:
            sids = self.order
            #self.align_pair
        for sid in sids:
            if self.splocs is None:
                rrnn = self.query(sid, sid, slot='enn', knn=o_neighbor+1, return_type='sparseidx', use_dpca=False)
            else:
                if o_neighbor and s_neighbor:
                    rrenn = self.query(sid, sid, slot='enn', knn=o_neighbor+1, return_type='sparseidx', use_dpca=False)
                    rrsnn = self.query(sid, sid, slot='snn', knn=s_neighbor+1, return_type='sparseidx')
                    rrnn = rrenn.multiply(rrsnn)
                elif (not o_neighbor) and s_neighbor:
                    rrnn = self.query(sid, sid, slot='snn', knn=s_neighbor+1, return_type='sparseidx')
                elif (not s_neighbor) and o_neighbor:
                    rrnn = self.query(sid, sid, slot='enn', knn=o_neighbor+1, return_type='sparseidx', use_dpca=False)

            if show_simi:
                mridx, mqidx = rrnn.nonzero()
                simis  = self.simi_pair(sid, sid, pairidx=[mridx, mqidx])
                import matplotlib.pylab as plt
                fig, ax = plt.subplots(1,3, figsize=(12,4))
                ax[0].hist(rrnn.sum(0), bins=s_neighbor, facecolor='b', label = f'{np.mean(rrnn.sum(0))}')
                ax[1].hist(rrnn.sum(1), bins=s_neighbor, facecolor='b', label = f'{np.mean(rrnn.sum(1))}')
                ax[2].hist(simis, bins=100, facecolor='b', label = f'{np.mean(simis)}')
                plt.show()
            rrnns[sid] = rrnn
        return rrnns

    def swnnscore(self, ssnn, n_neighbor = None, lower = 0.01, upper = 0.995):
        mhits = ssnn.data
        mhits = np.sqrt(mhits)
        # plt.hist(mhits, bins=100)
        # plt.show()

        n_neighbor = min(n_neighbor or max(mhits), max(mhits))
        mhits = np.float64(mhits)/np.float64(n_neighbor)

        min_score = np.quantile(mhits, lower)
        max_score = np.quantile(mhits, upper)
        mhits = (mhits-min_score)/(max_score-min_score)
        mhits  = np.clip(mhits, 0, 1)
        # print(min_score, max_score, n_neighbor, lower, upper)
        # plt.hist(mhits, bins=100)
        # plt.show()
        return mhits

    def swmnn(self, rsid, qsid, rrnn=None, qqnn = None, m_neighbor=6, 
              e_neighbor =30, s_neighbor =30, o_neighbor = 50, 
              lower = 0.01, upper = 0.995, 
              use_dpca=True, 
              drawmatch =False, 
                line_width=0.1, line_alpha=0.35,
                line_sample=None,
                size=1,
                fsize=7,
                **kargs):
        qrnna = self.query(rsid, qsid, slot='enn', knn=e_neighbor, return_type='raw', use_dpca=use_dpca, sort_dist=True, **kargs) #shape[0] == q
        rqnna = self.query(qsid, rsid, slot='enn', knn=e_neighbor, return_type='raw', use_dpca=use_dpca, sort_dist=True, **kargs) #shape[0] == r

        if (rrnn is None):
            rrnn = self.selfsnns(sids=[rsid], o_neighbor = o_neighbor, s_neighbor =s_neighbor)[rsid]
        if qqnn is None:
            qqnn = self.selfsnns(sids=[qsid], o_neighbor = o_neighbor, s_neighbor =s_neighbor)[qsid]

        qrnn = self.translabel(qrnna, rsize=rrnn.shape[0], return_type='sparseidx')
        rqnn = self.translabel(rqnna, rsize=qqnn.shape[0], return_type='sparseidx')

        qrmnn = [qrnna[0][:,:m_neighbor], qrnna[1][:,:m_neighbor] ]
        qrmnn = self.translabel(qrmnn, rsize=rrnn.shape[0], return_type='sparseidx')
        rqmnn = [rqnna[0][:,:m_neighbor], rqnna[1][:,:m_neighbor] ]
        rqmnn = self.translabel(rqmnn, rsize=qqnn.shape[0], return_type='sparseidx')

        ssnn = (rqnn.dot(qqnn.transpose())).multiply(rrnn.dot(qrnn.transpose()))
        mnn = rqmnn.multiply(qrmnn.transpose())
        ssnn = ssnn.multiply(mnn) #
        if not ssnn.has_sorted_indices:
            ssnn.sort_indices()

        mridx, mqidx = ssnn.nonzero()
        mhits = self.swnnscore(ssnn, n_neighbor=min(e_neighbor, s_neighbor),
                               lower = lower, upper = upper)
        keepidx = mhits > 0 
        mridx = mridx[keepidx].astype(np.int64)
        mqidx = mqidx[keepidx].astype(np.int64)
        mhits = mhits[keepidx].astype(np.float64)

        ssnn = csr_array((mhits, (mridx, mqidx)), shape=ssnn.shape)
        ssnn.sort_indices()
        ssnn.eliminate_zeros()

        if drawmatch and (not self.splocs is None):
            ridx, rlabel = self.groupidx[rsid]
            qidx, qlabel = self.groupidx[qsid]
            rposall = self.splocs[ridx]
            qposall = self.splocs[qidx]
            drawMatches( (rposall[mridx], qposall[mqidx]),
                        bgs =(rposall, qposall),
                        line_color = ('r'), ncols=2,
                        pairidx=[(0,1)], fsize=fsize,
                        titles= [rsid, qsid],
                        line_sample=line_sample,
                        size=size,
                        line_width=line_width, line_alpha=line_alpha)
        return ssnn

    def swmnns(self, m_neighbor=6, e_neighbor =30, s_neighbor =30, 
                o_neighbor = 50,
                use_dpca=True, 
                merge_edges=False,
            **kargs):

        rrnns = self.selfsnns(s_neighbor = s_neighbor, o_neighbor=o_neighbor)
        paris, scores = [], []
        matches = {}
        for i, (rid, qid) in enumerate(self.align_pair):
            if self.verbose>=1:
                print(f'match: {rid} -> {qid}')
            rsid = self.order[rid]
            qsid = self.order[qid]
            ridx, rlabel = self.groupidx[rsid]
            qidx, qlabel = self.groupidx[qsid]

            ssnn = self.swmnn(rsid, qsid, rrnn=rrnns[rsid], qqnn = rrnns[qsid], 
                              use_dpca = use_dpca,
                              m_neighbor=m_neighbor, e_neighbor =e_neighbor, 
                              s_neighbor =s_neighbor, **kargs)
            if merge_edges:
                mridx, mqidx = ssnn.nonzero()
                mhits = ssnn.data
                mridx = rlabel[mridx]
                mqidx = qlabel[mqidx]
                paris.append(np.array([mridx, mqidx]))
                scores.append(mhits)
            else:
                matches[(rid, qid)] = ssnn
        if merge_edges:
            paris = np.concatenate(paris, axis=1, dtype=np.int64)
            scores = np.concatenate(scores, axis=0, dtype=np.float64)
            matches = csr_array((scores, (paris[0], paris[1])), shape=(self.N, self.N))
            matches = [matches, self.align_pair]
        return matches

    def nnmatch(self, rsid, qsid, knn=6, **kargs):

        qrnn = self.query(rsid, qsid, knn=knn, return_type='lists', **kargs)
        rqnn = self.query(qsid, rsid, knn=knn, return_type='lists', **kargs)
        rqnn = zip(rqnn[1], rqnn[0])
        qrnn = zip(qrnn[0], qrnn[1])

        mnn = set(qrnn) & set(rqnn)
        mnn = np.array(list(mnn))

        return mnn

    def negative_self0(self, kns=10, seed = None, exclude_edge_index = None):
        nnn_idx = []
        for rsid in self.order:
            ridx, rlabel = self.groupidx[rsid]
            nnn = self.negative_sampling(rlabel, kns=kns, seed=seed)
            nnn_idx.extend(nnn)
        if not exclude_edge_index is None:
            nnn_idx = list(set(nnn_idx) - set(exclude_edge_index))
        return np.array(nnn_idx)

    def negative_self(self, kns=10, seed = None, exclude_edge_index = None):
        nnn_idx = []
        for rsid in self.order:
            rlabel = self.kdls[rsid]
            nnn = self.negative_sampling(rlabel, kns=kns, seed=seed)
            nnn_idx.extend(nnn)
        if not exclude_edge_index is None:
            nnn_idx = list(set(nnn_idx) - set(exclude_edge_index))
        return np.array(nnn_idx)

    def negative_hself(self, edge_index, kns=None, seed = None):
        nnn_idx = []
        for rsid in self.order:
            rlabel = self.kdls[rsid]
            iposidx = np.isin(edge_index[1], rlabel) & np.isin(edge_index[0], rlabel) #src ->dst
            nnn = self.negative_hsampling(edge_index[:, iposidx], rlabel, kns=kns, seed=seed)
            nnn_idx.append(nnn)
        return np.concatenate(nnn_idx, axis=1)
    
    def pairmnn(self, knn=10, cross=True, return_dist=False, direct=False, **kargs):
        mnn_idx = []
        for i, (ridx, qidx) in enumerate(self.align_pair):
            rsid = self.order[ridx]
            qsid = self.order[qidx]
            imnn = self.nnmatch(rsid, qsid, knn=knn, cross=cross,
                                direct=direct,
                                return_dist=return_dist, **kargs)
            recol = [1,0,2] if return_dist else [1,0]
            imnn = np.vstack([imnn, imnn[:,recol]])
            mnn_idx.append(imnn)
        return np.concatenate(mnn_idx, axis=0)
    
    @staticmethod
    def negative_sampling(labels, kns=10, seed = None, exclude_edge_index = None):
        n_nodes = len(labels)
        rng = np.random.default_rng(seed=seed)
        idx = rng.integers(0, high = n_nodes, size=[n_nodes,kns])
        nnn = [ (labels[v], labels[k]) for k in range(n_nodes) for v in idx[k]] #src->dst
        if not exclude_edge_index is None:
            nnn = list(set(nnn) - set(exclude_edge_index))
        else:
            nnn = list(set(nnn))
        return (nnn)

def sswnn_pair( X, Y, X_feat, Y_feat,
                pair_name = None,
                kd_method='annoy', sp_method = 'sknn',

                use_dpca = False, dpca_npca = 60, 
                max_pairs = None, min_pairs=100, score_threshold=0.05,
                m_neighbor=10, e_neighbor =30, s_neighbor =30,
                o_neighbor = None,
                lower = 0.01, upper=0.995,
                verbose = 1, 
                point_size=1,
                drawmatch=False,  line_sample=None,
                line_width=0.5, line_alpha=0.5, **kargs):

    assert X.shape[0] == X_feat.shape[0]
    assert Y.shape[0] == Y_feat.shape[0]

    verbose and print('Compute sswnn...')
    if pair_name is None:
        pair_name = ['X', 'Y']
    hData = np.concatenate([X_feat, Y_feat], axis=0)
    position = np.concatenate([X, Y], axis=0)
    groups = np.repeat(pair_name, [X.shape[0], Y.shape[0]])

    ssnn = SSWNN(hData, groups,
                splocs=position,
                levels=pair_name,)
    ssnn.build( method=kd_method,
                spmethod=sp_method,
                root=0,
                regist_pair=None,
                step=1,
                full_pair=False)

    nnmatch = ssnn.swmnn(pair_name[0], pair_name[1],
                         m_neighbor=m_neighbor, 
                         e_neighbor =e_neighbor, 
                        s_neighbor =s_neighbor, 
                        o_neighbor =o_neighbor,

                        lower = lower, 
                        upper = upper, 
                        use_dpca = use_dpca,
                        dpca_npca = dpca_npca,
                        drawmatch=drawmatch,
                        line_width=line_width,
                        line_alpha=line_alpha,
                        line_sample=line_sample,
                        size=point_size,
                        **kargs)

    mridx, mqidx = nnmatch.nonzero()
    mscore = nnmatch.data
    verbose and print(f'raw sswnn pairs: {mscore.shape[0]}...')
    
    t_idx = np.where(mscore>=score_threshold)[0]
    # if (not min_pairs is None): # TODO
    #     if (mscore.shape[0] >= min_pairs) and (t_idx.shape[0] >= min_pairs):

    #     print(f'warning: min pairs is lower than {min_pairs}.')
    verbose and print(f'sswnn pairs filtered by score_threshold: {t_idx.shape[0]}...')
    
    if (max_pairs is not None):
        max_pairs = min(max_pairs, t_idx.shape[0])
        mscore1 = mscore[t_idx]
        ind = np.argpartition(mscore1, -max_pairs)[-max_pairs:]
        t_idx = t_idx[ind]
        verbose and print(f'sswnn pairs filtered by max_pairs: {t_idx.shape[0]}...')
    mridx = mridx[t_idx]
    mqidx = mqidx[t_idx]
    mscore = mscore[t_idx]
    pairs = np.stack((mridx, mqidx), axis=1, dtype=np.int64)
    return pairs, mscore

def sswnn_match( hData, groups, position=None,
                levels = None,  
                kd_method='hnsw', sp_method = 'sknn',
                root=None, regist_pair=None, full_pair=False, step=1,

                use_dpca = False, dpca_npca = 60,
                m_neighbor=6, e_neighbor =30, s_neighbor =30,
                o_neighbor = 30,
                lower = 0.01, upper = 0.995,

                point_size=1,
                drawmatch=False,  line_sample=None,
                merge_edges=False,
                line_width=0.5, line_alpha=0.5, verbose=0, **kargs):

    ssnn = SSWNN(hData, groups,
                splocs=position,
                levels=levels,
                verbose=verbose)
    ssnn.build( method=kd_method,
                spmethod=sp_method,
                root=root,
                regist_pair=regist_pair,
                step=step,
                full_pair=full_pair)
    return ssnn.swmnns(m_neighbor=m_neighbor,
                                e_neighbor =e_neighbor,
                                s_neighbor =s_neighbor,
                                o_neighbor =o_neighbor,
                                use_dpca = use_dpca,
                                dpca_npca = dpca_npca,
                                lower = lower,
                                upper = upper,
                                drawmatch=drawmatch,
                                line_width=line_width,
                                line_alpha=line_alpha,
                                line_sample=line_sample,
                                fsize=4,
                                size=point_size,
                                merge_edges = merge_edges,
                                **kargs)