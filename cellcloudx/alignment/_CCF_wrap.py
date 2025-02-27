import numpy as np
import collections
import skimage.transform as skitf
import skimage as ski
import scipy as sci
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from ..tools._search import searchidx
from ..plotting._imageview import drawMatches
from ..transform import ccf_transform_point, homoreshape
from ..utilis._arrays import isidentity, list_iter
from ..utilis._clean_cache import clean_cache
from ._CCF import ccf, ccf_registration
from ..io._logger import logger

class ccf_wrap():
    def __init__(self, 
                pts=None, 
                features=None,
                groups=None, 
                levels=None, **kargs): #TODO
        self.pts = np.asarray(pts, dtype=np.float64)
        self.features = features

        self.N = pts.shape[0]
        self.D = pts.shape[1]
        assert pts.shape[0] == len(groups)

        self.levels = levels
        self.cellid = np.arange(self.N)
        if levels is None:
            try:
                self.levels = groups.cat.remove_unused_categories().cat.categories
            except:
                self.levels = np.unique(groups)
        self.levels = list(self.levels)
        self.nlevel = len(self.levels)
        assert self.nlevel >1

        self.groups = collections.OrderedDict()
        for igroup in self.levels:
            self.groups[igroup] = self.cellid[(groups == igroup)]

    def buildidx(self,
                root=None, 
                regist_pair=None,
                full_pair=False,
                step=1,
                self_pair=False,
                show_tree=False):
        if not root is None:
            root = self.levels.index(root)
        if not regist_pair is None:
            regist_pair = [ (self.levels.index(r), self.levels.index(q)) 
                            for r,q in regist_pair ]
        
        align_pair, trans_pair = searchidx(self.nlevel, 
                                        root=root,
                                        regist_pair=regist_pair,
                                        full_pair=full_pair,
                                        self_pair=self_pair,
                                        step=step,
                                        show_tree=show_tree)
        
        align_pair = [ (self.levels[r], self.levels[q]) 
                                    for r,q in align_pair ]
        # trans_pair = [ (self.levels[r], self.levels[q]) 
        #                             for r,q in trans_pair ]
        return align_pair, trans_pair

    def regist(self, rid, qid, pts=None, method=['ansac', 'ccd'], 
                      transformer=['rigid', 'rigid'],
                      rsample=None, qsample=None,
                      seed = 200504,
                      tran_args = {},
                      **kargs):
        np.random.seed(seed)
        xidx, yidx =  self.groups[rid], self.groups[qid]
        if not rsample is None:
            if rsample < 1:
                rsample = int(rsample*len(xidx))
            rsample = min(rsample, len(xidx))
            xidxi = np.random.choice(xidx, size=rsample, replace=False)
        else:
            xidxi = xidx
        if not qsample is None:
            if qsample < 1:
                qsample = int(qsample*len(yidx))
            qsample = min(qsample, len(yidx))
            yidxi = np.random.choice(yidx, size=qsample, replace=False)
        else:
            yidxi = yidx
        if pts is None:
            pts = self.pts
        X = pts[xidxi]
        Y = pts[yidxi]
        X_feat = None if self.features is None else self.features[xidxi] 
        Y_feat = None if self.features is None else self.features[yidxi]
        model = ccf_registration(X, Y, X_feat=X_feat, Y_feat=Y_feat, 
                                 method=method, transformer=transformer, **kargs)
        paras  = model.paras
        tforms = model.tforms
        TY = model.transform_point(self.pts[yidx], **tran_args)
        return paras, tforms, TY

    def regists(self, root=None, regist_pair=None, full_pair=False,
                step=1,
                method=['ansac', 'ccd'], transformer=['rigid', 'rigid'],
                self_pair=False,
                show_tree=False, 
                stack=False,
                n_jobs=1, **kargs):
        align_pair, trans_pair = self.buildidx(root=root, 
                                                regist_pair=regist_pair,
                                                full_pair=full_pair,
                                                step=step,
                                                self_pair=self_pair,
                                                show_tree=show_tree)
        self.align_pair = align_pair
        self.trans_pair = trans_pair

        self.TY = np.array(self.pts, dtype=np.float64)
        self.paras = []
        self.tforms = []

        if stack: #TODO
            for rid, qid in self.align_pair:
                logger.info(f"register: {rid} <- {qid}")
                xidx, yidx =  self.groups[rid], self.groups[qid]
                iparas, itform, iTY = self.regist(rid, qid, pts= self.TY,
                                            method=method, transformer=transformer,
                                            **kargs)
                self.TY[yidx] = iTY
                self.paras.append(iparas)
                self.tforms.append(itform)
                clean_cache()
        else:
            n_jobs = 1
            if n_jobs > 1:
                models = Parallel( n_jobs=n_jobs, verbose=1 )(delayed(ccf_registration)
                                    (self.pts[self.groups[rid]], 
                                    self.pts[self.groups[qid]],
                                    X_feat= None if self.features is None else self.features[self.groups[rid]],
                                    Y_feat= None if self.features is None else self.features[self.groups[qid]],
                                    method=method, transformer=transformer,
                                    **kargs) 
                                for rid, qid in self.align_pair)
            else:
                models = [self.regist(rid, qid, 
                                      method=method, transformer=transformer,
                                      **kargs) for rid, qid in self.align_pair]
    
            for i, (rid, qid) in enumerate(self.align_pair):
                qidx = self.levels.index(qid)
                iyidx = self.groups[qid]
                self.TY[iyidx] = models[i][2]
                self.paras.append(models[i][0])
                self.tforms.append(models[i][1])

        clean_cache()
        return self

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

    # def transform_points(self, moving=None, tforms=None, inverse=False):
    #     tforms = self.tforms if tforms is None else tforms
    #     if moving is None:
    #         mov_out = self.loc.astype(np.float64).copy()
    #         for i,sid in enumerate(self.order):
    #             itform = tforms[i]
    #             icid = self.groupidx[sid][1]
    #             iloc = self.loc[icid]
    #             nloc = homotransform_point(iloc, itform, inverse=inverse)
    #             mov_out[icid,:] = nloc
    #     else:
    #         mov_out = homotransform_points(moving, tforms, inverse=inverse)
    #     self.mov_out = mov_out

