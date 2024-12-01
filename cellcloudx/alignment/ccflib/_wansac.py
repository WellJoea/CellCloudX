import numpy as np
import collections
import skimage.transform as skitf
import skimage as ski
import scipy as sci
import matplotlib.pyplot as plt


from ...plotting._imageview import drawMatches
from ...transform import homotransform_point, homotransform_points, homoreshape
from ...utilis._arrays import isidentity, list_iter
from .autothreshold_ransac import atransac
from ._sswnn import SSWNN

class wansac(SSWNN):
    TRANS = {
        'rigid':skitf.EuclideanTransform, #3
        'euclidean':skitf.EuclideanTransform, #3
        'similarity':skitf.SimilarityTransform, #4
        'affine':skitf.AffineTransform, #6
        'projective':skitf.ProjectiveTransform, # 8
        'homography':skitf.ProjectiveTransform,
        'piecewise-affine':skitf.PiecewiseAffineTransform,
        'fundamental': skitf.FundamentalMatrixTransform,
        'essential': skitf.EssentialMatrixTransform,
        'polynomial': skitf.PolynomialTransform,
    }
    def __init__(self, *arg, **kargs):
        super().__init__(*arg, **kargs)

        self.atransac = atransac
        self.transform_image = homoreshape

    def fitmodel(self, rsid, qsid,
                   model_class='rigid',
                   m_neighbor=6, e_neighbor =30, s_neighbor =30,
                   lower = 0.01, upper = 0.9, use_weight=False,
                   stop_merror = 1e-3,
                    min_samples=40, residual_threshold=1., 
                    residual_trials=100, max_trials=700, CI = 0.95,
                    drawmatch=False, verbose=False,
                    line_alpha=0.35,
                    line_sample=None,
                    line_width=0.5,
                    size=1,
                    fsize=5,
                    seed=491001,
                    titles = None,
                    hide_axis=False,
                    equal_aspect=True,
                    invert_xaxis=False,
                    invert_yaxis=False,
                    line_color = ('r'), 
                    ncols=2,
                    pargs={},
                  **kargs):

        model_class = self.TRANS[model_class] if model_class in self.TRANS else model_class
        if titles is None:
            titles = [rsid, qsid]

        # mnnk = self.nnmatch(rsid, qsid, knn=knn,
        #                     rdata = rdata, qdata=qdata,
        #                      cross=cross, return_dist=True, **kargs)
        print(f'Match pairs: {rsid} <-> {qsid}')
        print('Match Features...')
        ssnn = self.swmnn(rsid, qsid, rrnn=self.rrnns[rsid], 
                            qqnn = self.rrnns[qsid], 
                            m_neighbor=m_neighbor, e_neighbor =e_neighbor, 
                            s_neighbor =s_neighbor,lower = lower, upper = upper,
                             **kargs)
        mridx, mqidx = ssnn.nonzero()
        mscore = ssnn.data
        ridx, rlabel = self.groupidx[rsid]
        qidx, qlabel = self.groupidx[qsid]
        mridx = rlabel[mridx]
        mqidx = qlabel[mqidx]

        rpos = self.splocs[mridx]
        qpos = self.splocs[mqidx]
        print('Evaluate Transformation...')
        inliers, model = self.atransac(rpos, qpos, model_class,
                                            min_samples=min_samples,
                                            data_weight = (mscore if use_weight else None),
                                            max_trials=max_trials,
                                            CI=CI,
                                            residual_trials=residual_trials,
                                            verbose=verbose,
                                            seed=seed,
                                            stop_merror=stop_merror,
                                            residual_threshold=residual_threshold)
        src_pts = rpos[inliers]
        dst_pts = qpos[inliers]

        rposall = self.splocs[ridx]
        qposall = self.splocs[qidx]
        dst_mov = homotransform_point(qposall, model, inverse=False)

        keepidx = np.zeros_like(mridx)
        keepidx[inliers] = 1
        anchors = np.vstack([mridx, mqidx, keepidx, mscore]).T

        if drawmatch:
            ds3 = homotransform_point(dst_pts, model, inverse=False)
            drawMatches( (src_pts, ds3), bgs =(rposall, dst_mov),
                        line_color = line_color, ncols=ncols,
                        pairidx=[(0,1)], fsize=fsize,
                        titles= titles,
                        size=size,
                        equal_aspect = equal_aspect,
                        hide_axis=hide_axis,
                        invert_xaxis=invert_xaxis,
                        invert_yaxis=invert_yaxis,
                        line_sample=line_sample,
                        line_alpha=line_alpha,
                        line_width=line_width,
                        **pargs)
        return model, anchors

    def regists(self, m_neighbor=6, e_neighbor =30, s_neighbor =30,
               method='rigid', CIs = 0.95, o_neighbor = 60, 
               broadcast = True,  stop_merror=1e-3, lower=0.05, residual_trials=100,
               drawmatch=False, line_width=0.5, line_alpha=0.5, **kargs):

        CIs = list_iter(CIs)
        method = list_iter(method)
        lowers = list_iter(lower)
        m_neighbors = list_iter(m_neighbor)
        residual_trials = list_iter(residual_trials)

        self.rrnns = self.selfsnns(o_neighbor = o_neighbor, s_neighbor=s_neighbor)
        tforms = [np.identity(3)] * len(self.order)
        self.matches = {}
        for i, (ridx, qidx) in enumerate(self.align_pair):
            rsid = self.order[ridx]
            qsid = self.order[qidx]
            model, anchors = self.fitmodel(rsid, qsid, 
                                         drawmatch=drawmatch, 
                                         model_class=method[i],
                                         CI=CIs[i],
                                         lower=lowers[i],
                                         m_neighbor = m_neighbors[i], 
                                         e_neighbor = e_neighbor, 
                                         s_neighbor = s_neighbor,
                                         line_width=line_width,
                                         line_alpha=line_alpha,
                                         stop_merror = stop_merror,
                                         residual_trials=residual_trials[i],
                                         **kargs)
            tforms[qidx] = model
            self.matches[(rsid, qsid)] = anchors

        self.tmats = tforms
        self.tforms = self.update_tmats(self.trans_pair, tforms) if broadcast else tforms

    def transform_points(self, moving=None, tforms=None, inverse=False):
        tforms = self.tforms if tforms is None else tforms
        if moving is None:
            mov_out = self.splocs.astype(np.float64).copy()
            for i,sid in enumerate(self.order):
                itform = tforms[i]
                icid = self.groupidx[sid][1]
                iloc = self.splocs[icid]
                nloc = homotransform_point(iloc, itform, inverse=inverse)
                mov_out[icid,:] = nloc
        else:
            mov_out = homotransform_points(moving, tforms, inverse=inverse)
        self.mov_out = mov_out

    @staticmethod
    def transform_images(
                   images = None, 
                   tform =None,
                   sf=None,
                   rescale = None,
                   padsize = None, **kargs):
        if (images is None):
            return 

        mov_imgs = []
        mov_tmats = []
        for i, (img) in enumerate(images):
            img, itam = homoreshape(img, 
                                    tform = list_iter(tform)[i], 
                                    sf = list_iter(sf)[i],
                                    rescale = list_iter(rescale)[i],
                                    padsize = list_iter(padsize)[i], **kargs)
            mov_imgs.append(img)
            mov_tmats.append(itam)
        return mov_imgs, mov_tmats

    @staticmethod
    def update_tmats(trans_pair, raw_tmats):
        try:
            new_tmats = raw_tmats.copy()
        except:
            new_tmats = [ imt.copy() for imt in raw_tmats]

        for ifov, imov in trans_pair:
            new_tmats[imov] = np.matmul(new_tmats[imov], new_tmats[ifov])
        return new_tmats

    @staticmethod
    def split(groups):
        try:
            Order = groups.cat.remove_unused_categories().cat.categories
        except:
            Order = np.unique(groups)

        idxdict = collections.OrderedDict()
        for igroup in Order:
            idxdict[igroup] = (groups == igroup)
        return idxdict

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

    @staticmethod
    def negative_hsampling( edge_index, labels, kns=None, seed = 200504):
        pos_lab, counts = np.unique(edge_index[1],return_counts=True)  #src -> dst
        nev_set = []
        seed = [seed] if isinstance(seed, int) else seed
        for i in range(len(pos_lab)):
            ipos = pos_lab[i]
            isize = kns or counts[i]
            iset =  edge_index[0][edge_index[1] == ipos]
            nevs =  list(set(labels) - set(iset) -set([ipos]) )
            rng = np.random.default_rng(seed=[i, *seed])
            inev = rng.choice(nevs, size=isize, replace=False, shuffle=False)
            nev_set.append(inev)
        nev_set= np.concatenate(nev_set, axis=0)
        if kns is None:
            src_set= edge_index[1]
        else:
            src_set= np.repeat(pos_lab, kns)
        neg_sam = np.array([src_set, nev_set])
        return (neg_sam)

    @staticmethod
    def icp_transform( A, B):
        assert A.shape == B.shape

        # get number of dimensions
        m = A.shape[1]

        # translate points to their centroids
        centroid_A = np.mean(A, axis=0)
        centroid_B = np.mean(B, axis=0)
        AA = A - centroid_A
        BB = B - centroid_B

        # rotation matrix
        H = np.dot(AA.T, BB)
        U, S, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)

        # special reflection case
        if np.linalg.det(R) < 0:
            Vt[m-1,:] *= -1
            R = np.dot(Vt.T, U.T)

        # translation
        t = centroid_B.T - np.dot(R,centroid_A.T)

        # homogeneous transformation
        T = np.identity(m+1)
        T[:m, :m] = R
        T[:m, m] = t

        return T, R, t

def catsc(position, feauture,  groups, 
                ckd_method='hnsw', sp_method = 'sknn',
                use_dpca = False,
                dpca_npca = 60,
                root=None, regist_pair=None, full_pair=False, step=1,
                m_neighbor= 6, 
                e_neighbor= 30,
                s_neighbor= 30,
                o_neighbor = 30, 
                lower = 0.01,
                stop_merror = 1e-3,
                reg_method = 'rigid', point_size=1,
                CIs = 0.93, 
                drawmatch=False,  line_sample=None,
                line_width=0.5, line_alpha=0.5,**kargs):
    ws = wansac()
    ws.build(feauture, groups,
                splocs=position,
                method=ckd_method,
                spmethod=sp_method,
                dpca_npca = dpca_npca,
                root=root,
                regist_pair=regist_pair,
                step=step,
                full_pair=full_pair)
    ws.regists(m_neighbor=m_neighbor,
                e_neighbor =e_neighbor,
                s_neighbor =s_neighbor,
                o_neighbor = o_neighbor,
                use_dpca = use_dpca,
                lower = lower,
                method=reg_method,
                CIs = CIs,
                broadcast = True,
                drawmatch=drawmatch,
                line_width=line_width,
                line_alpha=line_alpha,
                line_sample=line_sample,
                fsize=4,
                size=point_size,
                stop_merror=stop_merror,
                **kargs)
    ws.transform_points()
    return ws


def cats_wrap(adata, groupby, 
              basis='spatial',
              latent = 'glatent',
              add_align = 'align',
              norm_latent = False,
              root=None,
              regist_pair=None,
              step=1,
              drawmatch=True,

              use_dpca = True,
              dpca_npca = 60,
              ckd_method ='hnsw',
              reg_method = 'rigid',
              m_neighbor= 2,
              e_neighbor = 30,
              s_neighbor = 30,
              o_neighbor = 60,
              lower = 0.05,
              CIs = 0.95,
              residual_trials=20,
              line_width=0.5,
              line_alpha=0.5,
              line_sample=None,
              point_size=1,
              stop_merror=1e-4,

              trans_img = False,
              rescale = None,
              padsize = None,
              order = None,
              img_key="hires",
              img_add_key = 'tres',


              verbose=1, **kargs):
    groups = adata.obs[groupby]
    position = adata.obsm[basis]
    Hs = adata.obsm[latent]

    if norm_latent:
        Hnorm = Hs / np.linalg.norm(Hs, axis=1, keepdims=True)
        # Hnorm = F.normalize(torch.FloatTensor(Hs), dim=1).numpy()
    else:
        Hnorm = Hs
    
    sargs = dict(
                root=root,
                regist_pair=regist_pair,
                step = step,
                m_neighbor = m_neighbor,
                e_neighbor =e_neighbor,
                s_neighbor =s_neighbor,
                o_neighbor =o_neighbor,
                use_dpca = use_dpca,
                dpca_npca = dpca_npca,
                lower = lower,
                ckd_method=ckd_method,
                reg_method = reg_method,
                point_size = point_size,
                CIs = CIs,
                stop_merror=stop_merror,
                drawmatch=drawmatch,
                line_sample=line_sample,
                line_width=line_width,
                line_alpha=line_alpha,
                residual_trials=residual_trials,
                verbose=verbose)
    sargs.update(kargs)
    wr = catsc( position, Hnorm, groups,  **sargs)

    tforms, tmats, new_pos, matches, Order = wr.tforms, wr.tmats, wr.mov_out, wr.matches, wr.order
    adata.obsm[add_align] = new_pos
    adata.uns[f'{add_align}_tforms'] = tforms
    adata.uns[f'{add_align}_tmats'] = tmats
    adata.uns[f'{add_align}_matches'] = matches
    adata.uns[f'{add_align}_order'] = Order
    print(f'finished: added to `.obsm["{add_align}"]`')
    print(f'          added to `.uns["{add_align}_tforms"]`')


    if trans_img:
        mtforms = dict(zip( Order, tforms))
        groups = adata.obs[groupby]
        try:
            order = groups.cat.remove_unused_categories().cat.categories
        except:
            order = np.unique(groups)
    
        for igroup in order:

            itam = mtforms.get(igroup, np.eye(3))

            if not igroup in adata.uns[basis].keys():
                adata.uns[basis][igroup] = {}
            adata.uns[basis][igroup][f'{img_add_key}_postmat'] = itam

            try:
                iimg = adata.uns[basis][igroup]['images'][img_key]
    
                isf = adata.uns[basis][igroup]['scalefactors'].get(f'tissue_{img_key}_scalef',1)
                iimg, itam_sf =  homoreshape(iimg, tform=itam, scalef=isf, rescale=rescale, padsize=padsize, order=order)
                re_isf = rescale if rescale else isf

                adata.uns[basis][igroup]['images'][img_add_key] = iimg
                adata.uns[basis][igroup][f'{img_add_key}_imgtmat'] = itam_sf
                adata.uns[basis][igroup]['scalefactors'][f'tissue_{img_add_key}_scalef'] = re_isf
            except:
                if verbose >1:
                    print(f'No image was found in `.uns[{basis}][{igroup}]["images"][{img_key}]`.')
                    print(f'pass images registration.')

        verbose and print(f'finished: added to `.obsm["{add_align}"]`')
        verbose and print(f'          added to `.uns["{basis}"][<group>]"]`')

