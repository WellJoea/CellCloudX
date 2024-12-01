import numpy as np
import collections
import skimage.transform as skitf
import skimage as ski
import scipy as sci
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from ..tools._search import searchidx
from ..plotting._imageview import drawMatches
from ..transform import homotransform_point, homotransform_points, homoreshape
from ..utilis._arrays import isidentity, list_iter
from ._CCF import ccf, ccf_registration

class ccf_wrap():
    def __init__(self, pts, features, groups, levels=None, **kargs):
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
        self.nlevel = len(levels)
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

    def regist(self, rid, qid, method=['ansac', 'ccd'], 
                      transformer=['rigid', 'rigid'],
                      **kargs):
        xidx, yidx =  self.groups[rid], self.groups[qid]
        X = self.pts[xidx]
        Y = self.pts[yidx]
        X_feat = None if self.features is None else self.features[xidx] 
        Y_feat = None if self.features is None else self.features[yidx] 

        model = ccf_registration(X, Y, X_feat=X_feat, Y_feat=Y_feat, 
                                 method=method, transformer=transformer, **kargs)
        return model

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
        self.models = []
        self.TY = np.array(self.pts, dtype=np.float64)
        self.paras = []
        self.tforms = []

        if stack:
            for rid, qid in self.align_pair:
                xidx, yidx =  self.groups[rid], self.groups[qid]
                X = self.TY[xidx]
                Y = self.TY[yidx]
                X_feat = None if self.features is None else self.features[xidx] 
                Y_feat = None if self.features is None else self.features[yidx] 
                model = ccf_registration(X, Y, X_feat=X_feat, Y_feat=Y_feat,
                                         method=method, transformer=transformer,
                                          **kargs)
                self.TY[yidx] = model.TY
                self.tforms.append(model.tforms)
                self.paras.append(model.paras)
                self.models.append(model)

        else:
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
            self.models = models

            for i, (rid, qid) in enumerate(self.align_pair):
                qidx = self.levels.index(qid)
                iyidx = self.groups[qid]
                imodel = models[i]
                self.TY[iyidx] = imodel.TY
                self.tforms.append(imodel.tforms)
                self.paras.append(imodel.paras)
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

