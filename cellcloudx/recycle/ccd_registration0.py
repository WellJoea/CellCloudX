"""
CCD was Modified from: https://github.com/siavashk/pycpd
"""
import numpy as np
import pandas as pd
import collections

from .rigid_registration import RigidRegistration
from .affine_registration import AffineRegistration
from .deformable_registration import DeformableRegistration
from .constrained_deformable_registration import ConstrainedDeformableRegistration
from ...tools._search import searchidx
from ...utilis._arrays import isidentity, list_iter
from ...transform import homotransform_point, homotransform_points, homoreshape

class ccd:
    TRANS = {
        'rigid':RigidRegistration, 
        'euclidean':RigidRegistration,
        'similarity':RigidRegistration, 
        'affine':AffineRegistration, 
        'deformable':DeformableRegistration,
        'constraineddeformable':ConstrainedDeformableRegistration,
    }
    def __init__(self, position, groups, features=None, lables=None, levels=None):
        self.position = position
        self.groups = groups
        self.features = features
        self.lables = lables
        self.levels = levels
        self.N = position.shape[0]
        assert position.shape[0] == len(groups)

    def build(self,
               root=None, 
                regist_pair=None,
                full_pair=False,
                step=1,
                show_tree=False, 
                keep_self=True):
        if self.lables is None:
            try:
                self.lables = self.groups.index.values
            except:
                self.lables = np.arange(self.N)

        self.cellid = np.arange(self.N)
        if self.levels is None:
            try:
                self.levels = self.groups.cat.remove_unused_categories().cat.categories
            except:
                self.levels = np.unique(self.groups)

        self.align_pair, self.trans_pair = searchidx(len(self.levels), 
                                                root=root,
                                                regist_pair=regist_pair,
                                                full_pair=full_pair,
                                                keep_self=keep_self,
                                                step=step,
                                                show_tree=show_tree)
        groupidx = collections.OrderedDict()
        for igroup in self.levels:
            groupidx[igroup] = [self.groups == igroup, self.cellid[(self.groups == igroup)]]
        self.groupidx = groupidx

    def regist(self, method='affine', broadcast = True, **kargs):
        method = list_iter(method)
        tforms = [np.identity(3)] * len(self.levels)
        self.matches = {}
        for i, (ridx, qidx) in enumerate(self.align_pair):
            rsid = self.levels[ridx]
            qsid = self.levels[qidx]
            ridx, rlabel = self.groupidx[rsid]
            qidx, qlabel = self.groupidx[qsid]

            rX = self.position[ridx]
            qY = self.position[qidx]
            rF = self.features[ridx] if self.features is not None else None
            qF = self.features[qidx] if self.features is not None else None

            model = self.regist_pair(rX, qY, X_feat=rF, Y_feat=qF, method=method[i], **kargs)
            tforms[qidx] = model

        self.tmats = tforms
        self.tforms = self.update_tmats(self.trans_pair, tforms) if broadcast else tforms

    @staticmethod
    def regist( X, Y, X_feat=None, Y_feat=None, method='affine', 
                    source_id = None, target_id= None,**kwargs):
        model = ccd.TRANS[method]
        if method in ['rigid', 'euclidean']:
            fargs = {'scale': False}
        elif method in ['similarity']:
            fargs = {'scale': True}
        elif method in ['deformable']:
            if (not source_id is None) and (not target_id is None):
                model = ccd.TRANS['constraineddeformable']
        elif method in ['constraineddeformable']:
            assert (not source_id is None) and (not target_id is None)
        else:
            fargs = {}
        kwargs.update(fargs)
        model.register(X, Y, X_feat=X_feat, Y_feat=Y_feat, **kwargs)
        return 

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